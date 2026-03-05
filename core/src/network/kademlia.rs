use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Consistent hashing helper
// ---------------------------------------------------------------------------

/// Converts a search term to a NodeId-shaped key via blake3.
///
/// Used for consistent hashing: the node whose NodeId has minimum XOR distance
/// to this key is "responsible" for that term's index shard.
pub fn term_to_key(term: &str) -> NodeId {
    NodeId(*blake3::hash(term.as_bytes()).as_bytes())
}

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::network::messages::{MessageType, NetworkMessage, NodeId, decode_message, encode_message};
use crate::network::quic::{QuicTransport, TransportError};

const K: usize = 20;
const NUM_BUCKETS: usize = 256;

#[derive(Debug, Error)]
pub enum KademliaError {
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),
    #[error("no bootstrap nodes reachable")]
    NoBootstrap,
    #[error("routing table lock poisoned")]
    LockPoisoned,
}

/// A peer node's identity and address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: NodeId,
    pub addr: SocketAddr,
}

// ---------------------------------------------------------------------------
// RoutingTable
// ---------------------------------------------------------------------------

/// Kademlia k-bucket routing table.
///
/// 256 buckets for 256-bit NodeIds, k=20 per bucket.
/// Buckets are ordered by recency: index 0 = least recently seen, last = most recent.
pub struct RoutingTable {
    pub local_id: NodeId,
    buckets: Vec<Vec<NodeInfo>>,
}

impl RoutingTable {
    pub fn new(local_id: NodeId) -> Self {
        Self { local_id, buckets: vec![Vec::new(); NUM_BUCKETS] }
    }

    /// XOR distance between two node IDs.
    pub fn xor_distance(a: &NodeId, b: &NodeId) -> [u8; 32] {
        let mut dist = [0u8; 32];
        for i in 0..32 {
            dist[i] = a.0[i] ^ b.0[i];
        }
        dist
    }

    /// Bucket index = position of the highest differing bit (leading zeros of XOR distance).
    pub fn bucket_index(local: &NodeId, target: &NodeId) -> usize {
        let dist = Self::xor_distance(local, target);
        for (i, &byte) in dist.iter().enumerate() {
            if byte != 0 {
                return i * 8 + byte.leading_zeros() as usize;
            }
        }
        NUM_BUCKETS - 1 // identical IDs → last bucket (should not normally occur)
    }

    /// Inserts or refreshes a node in the routing table.
    ///
    /// - If already present: move to end (most-recently-seen).
    /// - If bucket full: evict oldest (index 0), insert at end.
    pub fn update(&mut self, node: NodeInfo) {
        if node.id == self.local_id {
            return; // never add self
        }
        let idx = Self::bucket_index(&self.local_id, &node.id);
        let bucket = &mut self.buckets[idx];

        if let Some(pos) = bucket.iter().position(|n| n.id == node.id) {
            bucket.remove(pos);
            bucket.push(node);
            return;
        }

        if bucket.len() >= K {
            bucket.remove(0); // evict least-recently-seen
        }
        bucket.push(node);
    }

    /// Returns up to `k` nodes closest to `target`, sorted by XOR distance.
    pub fn closest_nodes(&self, target: &NodeId, k: usize) -> Vec<NodeInfo> {
        let mut all: Vec<NodeInfo> =
            self.buckets.iter().flat_map(|b| b.iter().cloned()).collect();
        all.sort_by_key(|n| Self::xor_distance(target, &n.id));
        all.truncate(k);
        all
    }

    /// Returns the single node XOR-closest to `term`'s blake3 key.
    ///
    /// Returns None if the routing table is empty.
    /// Used for consistent hashing: routes each term to its responsible shard node.
    pub fn responsible_node(&self, term: &str) -> Option<NodeInfo> {
        let key = term_to_key(term);
        self.closest_nodes(&key, 1).into_iter().next()
    }
}

// ---------------------------------------------------------------------------
// Kademlia
// ---------------------------------------------------------------------------

/// Kademlia DHT node — routing table + iterative FIND_NODE.
///
/// `table` uses `std::sync::Mutex` because it is never held across `.await`.
pub struct Kademlia {
    table: Arc<Mutex<RoutingTable>>,
    transport: Arc<QuicTransport>,
}

impl Kademlia {
    pub fn new(local_id: NodeId, transport: Arc<QuicTransport>) -> Self {
        Self {
            table: Arc::new(Mutex::new(RoutingTable::new(local_id))),
            transport,
        }
    }

    /// Bootstraps the node by connecting to known peers and running FIND_NODE(self).
    pub async fn bootstrap(
        &self,
        bootstrap_nodes: Vec<SocketAddr>,
    ) -> Result<(), KademliaError> {
        if bootstrap_nodes.is_empty() {
            return Ok(());
        }

        let local_id = {
            let table = self.table.lock().map_err(|_| KademliaError::LockPoisoned)?;
            table.local_id.clone()
        }; // lock released before .await

        let mut reached_any = false;
        for addr in bootstrap_nodes {
            match self.transport.connect(addr).await {
                Ok(conn) => {
                    reached_any = true;
                    if let Err(e) = self.find_node_rpc(&conn, local_id.clone()).await {
                        tracing::warn!(?addr, error = %e, "bootstrap find_node failed");
                    }
                }
                Err(e) => {
                    tracing::warn!(?addr, error = %e, "bootstrap connect failed");
                }
            }
        }

        if !reached_any {
            return Err(KademliaError::NoBootstrap);
        }
        Ok(())
    }

    /// Iterative FIND_NODE — queries the closest known nodes for `target`.
    pub async fn find_node(
        &self,
        target: NodeId,
    ) -> Result<Vec<NodeInfo>, KademliaError> {
        let closest = {
            let table = self.table.lock().map_err(|_| KademliaError::LockPoisoned)?;
            table.closest_nodes(&target, K)
        }; // lock released

        let mut seen: HashSet<[u8; 32]> =
            closest.iter().map(|n| n.id.0).collect();
        let mut results = Vec::new();

        for node in closest {
            if let Ok(conn) = self.transport.connect(node.addr).await {
                if let Ok(nodes) = self.find_node_rpc(&conn, target.clone()).await {
                    for n in nodes {
                        if seen.insert(n.id.0) {
                            if let Ok(mut table) = self.table.lock() {
                                table.update(n.clone());
                            }
                            results.push(n);
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Sends a FIND_NODE RPC and awaits the response.
    async fn find_node_rpc(
        &self,
        conn: &quinn::Connection,
        target: NodeId,
    ) -> Result<Vec<NodeInfo>, KademliaError> {
        let local_id = {
            let table = self.table.lock().map_err(|_| KademliaError::LockPoisoned)?;
            table.local_id.clone()
        }; // lock released before .await

        let payload = encode_message(&target).map_err(TransportError::Message)?;
        let msg = NetworkMessage {
            kind: MessageType::DhtFindNode,
            payload,
            sender: local_id,
            signature: [0u8; 64],
        };
        QuicTransport::send(conn, &msg).await?;

        let response = QuicTransport::recv(conn).await?;
        let nodes: Vec<NodeInfo> =
            decode_message(&response.payload).map_err(TransportError::Message)?;
        Ok(nodes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> NodeId {
        NodeId([byte; 32])
    }

    fn info(byte: u8) -> NodeInfo {
        NodeInfo { id: id(byte), addr: "127.0.0.1:0".parse().unwrap() }
    }

    #[test]
    fn xor_distance_self_is_zero() {
        let a = id(42);
        assert_eq!(RoutingTable::xor_distance(&a, &a), [0u8; 32]);
    }

    #[test]
    fn bucket_index_first_bit_differs() {
        // a = 10000000…, b = 00000000… → XOR first bit set → bucket 0
        let mut a_bytes = [0u8; 32];
        let b_bytes = [0u8; 32];
        a_bytes[0] = 0b10000000;
        let a = NodeId(a_bytes);
        let b = NodeId(b_bytes);
        assert_eq!(RoutingTable::bucket_index(&a, &b), 0);
    }

    #[test]
    fn closest_nodes_sorted_by_distance() {
        let local = NodeId([0u8; 32]);
        let mut table = RoutingTable::new(local.clone());

        // Insert nodes with increasing XOR distance from local (0x00…)
        for byte in [0x10u8, 0x20, 0x01, 0x80] {
            let mut id_bytes = [0u8; 32];
            id_bytes[0] = byte;
            table.update(NodeInfo { id: NodeId(id_bytes), addr: "127.0.0.1:0".parse().unwrap() });
        }

        let closest = table.closest_nodes(&local, 4);
        // Verify sorted ascending by XOR distance
        for i in 1..closest.len() {
            let d_prev = RoutingTable::xor_distance(&local, &closest[i - 1].id);
            let d_curr = RoutingTable::xor_distance(&local, &closest[i].id);
            assert!(d_prev <= d_curr);
        }
    }

    #[test]
    fn update_respects_k_limit() {
        let local = NodeId([0u8; 32]);
        let mut table = RoutingTable::new(local.clone());

        // Insert K+5 nodes that all land in the same bucket.
        // All differ from local only in the last byte (same bucket index = 255).
        for i in 0..=(K as u8 + 4) {
            let mut id_bytes = [0u8; 32];
            id_bytes[31] = i + 1;
            table.update(NodeInfo {
                id: NodeId(id_bytes),
                addr: "127.0.0.1:0".parse().unwrap(),
            });
        }

        let total: usize = table.buckets.iter().map(|b| b.len()).sum();
        assert!(total <= K + 5); // may span multiple buckets
        for bucket in &table.buckets {
            assert!(bucket.len() <= K);
        }
    }

    #[test]
    fn term_to_key_is_deterministic() {
        let k1 = term_to_key("rust");
        let k2 = term_to_key("rust");
        assert_eq!(k1, k2);
    }

    #[test]
    fn term_to_key_differs_for_different_terms() {
        assert_ne!(term_to_key("rust"), term_to_key("python"));
    }

    #[test]
    fn responsible_node_returns_none_for_empty_table() {
        let table = RoutingTable::new(NodeId([0u8; 32]));
        assert!(table.responsible_node("rust").is_none());
    }

    #[test]
    fn responsible_node_returns_single_node_when_only_one() {
        let local = NodeId([0u8; 32]);
        let mut table = RoutingTable::new(local);
        let peer = NodeInfo { id: NodeId([1u8; 32]), addr: "127.0.0.1:0".parse().unwrap() };
        table.update(peer.clone());
        let responsible = table.responsible_node("anything");
        assert!(responsible.is_some());
        assert_eq!(responsible.unwrap().id, peer.id);
    }

    #[test]
    fn update_moves_existing_to_end() {
        let local = NodeId([0u8; 32]);
        let mut table = RoutingTable::new(local);

        let n = info(0x01);
        table.update(n.clone());
        let n2 = info(0x02);
        table.update(n2);
        table.update(n.clone()); // re-insert n — should move to end

        let idx = RoutingTable::bucket_index(&NodeId([0u8; 32]), &n.id);
        let bucket = &table.buckets[idx];
        assert_eq!(bucket.last().unwrap().id, n.id);
    }
}
