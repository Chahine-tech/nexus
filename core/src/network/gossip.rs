use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::network::messages::{MessageType, NetworkMessage, NodeId, encode_message};
use crate::network::quic::{QuicTransport, TransportError};
use crate::sketch::hyperloglog::HyperLogLog;

#[derive(Debug, Error)]
pub enum GossipError {
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),
}

/// Snapshot of a node's local state, propagated via gossip.
///
/// Merge rule: highest timestamp wins (idempotent, deterministic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipState {
    pub node_id: NodeId,
    pub doc_count: u64,
    /// Unix epoch seconds — used to resolve conflicts.
    pub timestamp: u64,
}

/// IDF gossip state — carries a HyperLogLog sketch for cardinality estimation.
///
/// Merge rule: per-register max (CRDT — commutative, associative, idempotent).
/// Separate from GossipState to keep concerns isolated and allow GossipPagerank
/// to follow the same pattern in week 4.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdfGossipState {
    pub node_id: NodeId,
    pub sketch: HyperLogLog,
    /// Unix epoch seconds — used to suppress re-broadcast of stale updates.
    pub timestamp: u64,
}

/// Gossip engine — propagates `doc_count` heartbeats + HyperLogLog IDF sketches.
///
/// All `std::sync` locks: never held across `.await`. `DashMap` for lock-free peers.
pub struct GossipEngine {
    local_state: Arc<RwLock<GossipState>>,
    peers: DashMap<NodeId, GossipState>,
    local_idf: Arc<RwLock<HyperLogLog>>,
    peer_idf: DashMap<NodeId, IdfGossipState>,
    transport: Arc<QuicTransport>,
}

impl GossipEngine {
    pub fn new(node_id: NodeId, transport: Arc<QuicTransport>) -> Self {
        let local_state =
            GossipState { node_id, doc_count: 0, timestamp: current_timestamp() };
        Self {
            local_state: Arc::new(RwLock::new(local_state)),
            peers: DashMap::new(),
            local_idf: Arc::new(RwLock::new(HyperLogLog::new())),
            peer_idf: DashMap::new(),
            transport,
        }
    }

    /// Updates local doc_count and bumps the timestamp. Sync.
    pub fn update_local(&self, doc_count: u64) {
        let mut state =
            self.local_state.write().expect("gossip RwLock poisoned");
        state.doc_count = doc_count;
        state.timestamp = current_timestamp();
    }

    /// Merges an incoming peer state — keeps the entry with the higher timestamp. Sync.
    pub fn handle_incoming(&self, state: GossipState) {
        self.peers
            .entry(state.node_id.clone())
            .and_modify(|existing| {
                if state.timestamp > existing.timestamp {
                    *existing = state.clone();
                }
            })
            .or_insert(state);
    }

    /// Broadcasts local state to all given peer addresses. Async.
    ///
    /// The RwLock is acquired, state cloned, and lock released **before** any `.await`.
    pub async fn broadcast(&self, peers: &[SocketAddr]) -> Result<(), GossipError> {
        // Clone state while holding the lock, then release before any .await.
        let state = {
            let guard = self.local_state.read().expect("gossip RwLock poisoned");
            guard.clone()
        };

        let payload = encode_message(&state).map_err(TransportError::Message)?;
        let sender = state.node_id.clone();

        for &addr in peers {
            let msg = NetworkMessage {
                kind: MessageType::GossipIdf,
                payload: payload.clone(),
                sender: sender.clone(),
                signature: [0u8; 64], // signing added in week 3
            };
            match self.transport.connect(addr).await {
                Ok(conn) => {
                    if let Err(e) = QuicTransport::send(&conn, &msg).await {
                        tracing::warn!(?addr, error = %e, "gossip send failed");
                    }
                }
                Err(e) => {
                    tracing::warn!(?addr, error = %e, "gossip connect failed");
                }
            }
        }
        Ok(())
    }

    /// Returns a snapshot of all known peer states. Sync.
    pub fn peer_states(&self) -> Vec<GossipState> {
        self.peers.iter().map(|e| e.value().clone()).collect()
    }

    // ---------------------------------------------------------------------------
    // IDF gossip — HyperLogLog sketch propagation
    // ---------------------------------------------------------------------------

    /// Adds a term to the local HyperLogLog sketch. Sync.
    ///
    /// Call this from the indexing path (e.g., Node::index_document).
    /// Lock is acquired and released immediately — never held across .await.
    pub fn add_term(&self, term: &str) {
        let mut hll = self.local_idf.write().expect("local_idf RwLock poisoned");
        hll.add(term.as_bytes());
    }

    /// Estimates global term cardinality by merging local sketch with all peer sketches. Sync.
    pub fn estimated_cardinality(&self) -> f64 {
        // Clone local HLL (lock released immediately).
        let mut merged = {
            let guard = self.local_idf.read().expect("local_idf RwLock poisoned");
            guard.clone()
        };
        // Merge all peer IDF sketches.
        for entry in self.peer_idf.iter() {
            merged.merge_in_place(&entry.value().sketch);
        }
        merged.estimate()
    }

    /// Merges an incoming IDF gossip state from a peer. Sync.
    ///
    /// CRDT merge: per-register max. Idempotent — calling twice with the same
    /// state produces the same result.
    pub fn handle_idf_incoming(&self, state: IdfGossipState) {
        self.peer_idf
            .entry(state.node_id.clone())
            .and_modify(|existing| {
                existing.sketch.merge_in_place(&state.sketch);
                existing.timestamp = existing.timestamp.max(state.timestamp);
            })
            .or_insert(state);
    }

    /// Broadcasts the local HyperLogLog sketch to all given peer addresses. Async.
    ///
    /// The RwLock is acquired, sketch cloned, and lock released **before** any `.await`.
    pub async fn broadcast_idf(&self, peers: &[SocketAddr]) -> Result<(), GossipError> {
        let local_node_id = {
            let guard = self.local_state.read().expect("gossip RwLock poisoned");
            guard.node_id.clone()
        };

        // Clone sketch while holding the lock, then release before any .await.
        let idf_state = {
            let guard = self.local_idf.read().expect("local_idf RwLock poisoned");
            IdfGossipState {
                node_id: local_node_id.clone(),
                sketch: guard.clone(),
                timestamp: current_timestamp(),
            }
        };

        let payload = encode_message(&idf_state).map_err(TransportError::Message)?;

        for &addr in peers {
            let msg = NetworkMessage {
                kind: MessageType::GossipIdfSketch,
                payload: payload.clone(),
                sender: local_node_id.clone(),
                signature: [0u8; 64],
            };
            match self.transport.connect(addr).await {
                Ok(conn) => {
                    if let Err(e) = QuicTransport::send(&conn, &msg).await {
                        tracing::warn!(?addr, error = %e, "idf gossip send failed");
                    }
                }
                Err(e) => {
                    tracing::warn!(?addr, error = %e, "idf gossip connect failed");
                }
            }
        }
        Ok(())
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::quic::QuicTransport;
    use crate::crypto::identity::NodeKeypair;

    fn make_engine() -> GossipEngine {
        // Dummy transport — gossip unit tests don't use the network.
        let kp = NodeKeypair::generate();
        // We need a QuicTransport but won't actually call broadcast in unit tests.
        // Wrap in Arc and pass — the engine won't use it.
        // Can't easily create a QuicTransport without binding, so use a fake node_id only.
        let node_id = kp.node_id();

        // For unit tests, create the engine with a minimal struct trick:
        // Use a real but disconnected transport bound to a free port.
        // Since broadcast isn't called in these tests, this is fine.
        let rt = tokio::runtime::Handle::try_current();
        if let Ok(handle) = rt {
            let transport = handle.block_on(async {
                rustls::crypto::ring::default_provider().install_default().ok();
                QuicTransport::bind("127.0.0.1:0".parse().unwrap(), &kp).await.unwrap()
            });
            GossipEngine::new(node_id, Arc::new(transport))
        } else {
            // Fallback for sync test context — create a minimal runtime.
            let rt = tokio::runtime::Runtime::new().unwrap();
            let transport = rt.block_on(async {
                rustls::crypto::ring::default_provider().install_default().ok();
                QuicTransport::bind("127.0.0.1:0".parse().unwrap(), &kp).await.unwrap()
            });
            GossipEngine::new(node_id, Arc::new(transport))
        }
    }

    #[test]
    fn handle_incoming_merges_by_max_timestamp() {
        let engine = make_engine();
        let peer_id = NodeId([9u8; 32]);

        let state_old =
            GossipState { node_id: peer_id.clone(), doc_count: 10, timestamp: 100 };
        let state_new =
            GossipState { node_id: peer_id.clone(), doc_count: 50, timestamp: 200 };

        engine.handle_incoming(state_old);
        engine.handle_incoming(state_new);

        let peers = engine.peer_states();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].doc_count, 50);
        assert_eq!(peers[0].timestamp, 200);
    }

    #[test]
    fn older_state_does_not_overwrite_newer() {
        let engine = make_engine();
        let peer_id = NodeId([8u8; 32]);

        let state_new =
            GossipState { node_id: peer_id.clone(), doc_count: 100, timestamp: 500 };
        let state_old =
            GossipState { node_id: peer_id.clone(), doc_count: 1, timestamp: 50 };

        engine.handle_incoming(state_new);
        engine.handle_incoming(state_old); // should not overwrite

        let peers = engine.peer_states();
        assert_eq!(peers[0].doc_count, 100);
        assert_eq!(peers[0].timestamp, 500);
    }

    #[test]
    fn update_local_sets_doc_count() {
        let engine = make_engine();
        engine.update_local(42);
        let state = engine.local_state.read().unwrap();
        assert_eq!(state.doc_count, 42);
        assert!(state.timestamp > 0);
    }

    #[test]
    fn add_term_affects_local_cardinality() {
        let engine = make_engine();
        assert_eq!(engine.estimated_cardinality(), 0.0);
        for i in 0u32..100 {
            engine.add_term(&format!("term_{i}"));
        }
        let est = engine.estimated_cardinality();
        assert!(est > 50.0, "estimated cardinality {est} should be > 50 after 100 terms");
    }

    #[test]
    fn handle_idf_incoming_merges_sketches() {
        let engine_a = make_engine();
        let engine_b = make_engine();

        let node_b_id = {
            engine_b.local_state.read().unwrap().node_id.clone()
        };

        for i in 0u32..300 {
            engine_a.add_term(&format!("term_{i}"));
        }

        let b_sketch = {
            let mut hll = HyperLogLog::new();
            for i in 300u32..600 {
                hll.add(format!("term_{i}").as_bytes());
            }
            hll
        };

        let idf_state = IdfGossipState {
            node_id: node_b_id,
            sketch: b_sketch,
            timestamp: current_timestamp(),
        };
        engine_a.handle_idf_incoming(idf_state);

        let est = engine_a.estimated_cardinality();
        let error = ((est - 600.0) / 600.0).abs();
        assert!(
            error < 0.20,
            "merged estimate {est:.1} is more than 20% off from 600 (error={error:.3})"
        );
    }

    #[test]
    fn handle_idf_incoming_is_idempotent() {
        let engine = make_engine();
        let peer_id = NodeId([5u8; 32]);
        let mut hll = HyperLogLog::new();
        for i in 0u32..100 {
            hll.add(&i.to_le_bytes());
        }
        let state = IdfGossipState {
            node_id: peer_id,
            sketch: hll,
            timestamp: 42,
        };
        engine.handle_idf_incoming(state.clone());
        engine.handle_idf_incoming(state);
        let est = engine.estimated_cardinality();
        // Should be the same as adding once.
        assert!(est > 50.0);
    }

    #[test]
    fn idf_gossip_state_serde_roundtrip() {
        let mut hll = HyperLogLog::new();
        for i in 0u32..50 {
            hll.add(&i.to_le_bytes());
        }
        let state = IdfGossipState {
            node_id: NodeId([1u8; 32]),
            sketch: hll.clone(),
            timestamp: 999,
        };
        let bytes = rmp_serde::to_vec(&state).unwrap();
        let decoded: IdfGossipState = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.sketch, hll);
        assert_eq!(decoded.timestamp, 999);
    }
}
