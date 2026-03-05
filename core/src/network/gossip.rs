use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::network::messages::{MessageType, NetworkMessage, NodeId, encode_message};
use crate::network::quic::{QuicTransport, TransportError};

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

/// Minimal gossip engine for week 2 — propagates `doc_count` + `timestamp`.
///
/// `local_state` uses `std::sync::RwLock` because no `.await` is held while
/// the lock is active. `peers` uses `DashMap` for lock-free concurrent access.
pub struct GossipEngine {
    local_state: Arc<RwLock<GossipState>>,
    peers: DashMap<NodeId, GossipState>,
    transport: Arc<QuicTransport>,
}

impl GossipEngine {
    pub fn new(node_id: NodeId, transport: Arc<QuicTransport>) -> Self {
        let local_state =
            GossipState { node_id, doc_count: 0, timestamp: current_timestamp() };
        Self {
            local_state: Arc::new(RwLock::new(local_state)),
            peers: DashMap::new(),
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
}
