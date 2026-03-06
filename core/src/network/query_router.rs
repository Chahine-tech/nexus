use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use futures::future::join_all;

use crate::crypto::identity::NodeKeypair;
use crate::crypto::reputation::ReputationStore;
use crate::network::kademlia::RoutingTable;
use crate::network::messages::{MessageType, NetworkMessage, NodeId, QueryRequest, QueryResponse, decode_message, encode_message};
use crate::network::quic::{QuicTransport, TransportError};
use crate::node::Node;

// ---------------------------------------------------------------------------
// RemoteQueryParams — groups args for the static remote-query helper
// ---------------------------------------------------------------------------

struct RemoteQueryParams {
    addr: SocketAddr,
    peer_id: NodeId,
    terms: Vec<String>,
    limit: usize,
    request_id: u64,
}

// ---------------------------------------------------------------------------
// QueryRouter — term-sharded fanout over QUIC
// ---------------------------------------------------------------------------

/// Routes queries across nodes using consistent hashing (blake3 XOR distance).
///
/// For each term:
///   - If this node is responsible (or routing table is empty): search locally.
///   - Otherwise: send a QueryRequest via QUIC to the responsible node.
///
/// Results are merged by sorting descending by score and truncating to `limit`.
/// Remote errors are logged and treated as empty results (degraded mode).
/// Failed remote queries are recorded in `reputation` to penalize unreliable peers.
///
/// `table` uses `std::sync::Mutex` — never held across `.await`.
pub struct QueryRouter {
    node: Arc<Node>,
    table: Arc<Mutex<RoutingTable>>,
    transport: Arc<QuicTransport>,
    local_id: NodeId,
    reputation: Arc<ReputationStore>,
    keypair: Arc<NodeKeypair>,
}

impl QueryRouter {
    pub fn new(
        node: Arc<Node>,
        table: Arc<Mutex<RoutingTable>>,
        transport: Arc<QuicTransport>,
        local_id: NodeId,
        reputation: Arc<ReputationStore>,
        keypair: Arc<NodeKeypair>,
    ) -> Self {
        Self { node, table, transport, local_id, reputation, keypair }
    }

    /// Routes a query: local BM25 for own shards, QUIC fanout for remote shards.
    ///
    /// Lock is acquired only to determine shard assignment, then released before any .await.
    pub async fn route_query(
        &self,
        terms: Vec<String>,
        limit: usize,
        request_id: u64,
    ) -> Vec<(u32, f32)> {
        if terms.is_empty() {
            return vec![];
        }

        // Step 1: Acquire lock, group terms by responsible node, release immediately.
        let (local_terms, remote_groups) = {
            let table = match self.table.lock() {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(error = %e, "routing table lock poisoned");
                    return self.node.search_terms(&terms, limit);
                }
            };

            let mut local_terms: Vec<String> = Vec::new();
            // (addr, peer_id) → Vec<String>
            let mut remote_groups: HashMap<(SocketAddr, NodeId), Vec<String>> = HashMap::new();

            for term in &terms {
                match table.responsible_node(term) {
                    Some(node_info) if node_info.id != self.local_id => {
                        remote_groups
                            .entry((node_info.addr, node_info.id.clone()))
                            .or_default()
                            .push(term.clone());
                    }
                    _ => {
                        // No responsible node found (empty table) or this node is responsible.
                        local_terms.push(term.clone());
                    }
                }
            }
            (local_terms, remote_groups)
        }; // lock released

        // Step 2: Local search — hybrid (BM25 + vector) when vector index is ready.
        let local_results = if local_terms.is_empty() {
            vec![]
        } else {
            let query = local_terms.join(" ");
            self.node.search_hybrid(&query, limit)
        };

        // Step 3: Remote fanout (concurrent) — skip peers below trust threshold.
        let futures: Vec<_> = remote_groups
            .into_iter()
            .filter(|((_, peer_id), _)| self.reputation.is_trusted(peer_id))
            .map(|((addr, peer_id), terms)| {
                let transport = Arc::clone(&self.transport);
                let local_id = self.local_id.clone();
                let reputation = Arc::clone(&self.reputation);
                let keypair = Arc::clone(&self.keypair);
                async move {
                    Self::query_remote_static(
                        &transport,
                        &local_id,
                        &reputation,
                        &keypair,
                        RemoteQueryParams { addr, peer_id, terms, limit, request_id },
                    )
                    .await
                }
            })
            .collect();

        let remote_results: Vec<Vec<(u32, f32)>> = join_all(futures).await;

        // Step 4: Merge all results — sort by score descending, truncate.
        let mut all: Vec<(u32, f32)> = local_results;
        for batch in remote_results {
            all.extend(batch);
        }
        all.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(limit);
        all
    }

    async fn query_remote_static(
        transport: &QuicTransport,
        local_id: &NodeId,
        reputation: &ReputationStore,
        keypair: &NodeKeypair,
        params: RemoteQueryParams,
    ) -> Vec<(u32, f32)> {
        let addr = params.addr;
        let peer_id = params.peer_id;
        let result: Result<Vec<(u32, f32)>, TransportError> = async {
            let qr = QueryRequest { terms: params.terms, limit: params.limit, request_id: params.request_id };
            let payload = encode_message(&qr).map_err(TransportError::Message)?;
            let signature = keypair.sign(&payload);
            let msg = NetworkMessage {
                kind: MessageType::QueryRequest,
                payload,
                sender: local_id.clone(),
                signature,
            };

            let conn = transport.connect(addr).await?;
            QuicTransport::send(&conn, &msg).await?;
            let response = QuicTransport::recv(&conn).await?;
            let qr: QueryResponse =
                decode_message(&response.payload).map_err(TransportError::Message)?;
            Ok(qr.results)
        }
        .await;

        match result {
            Ok(results) => {
                reputation.record_success(&peer_id);
                results
            }
            Err(e) => {
                reputation.record_failure(&peer_id);
                tracing::warn!(?addr, error = %e, "remote query failed — returning empty results");
                vec![]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::reputation::ReputationStore;
    use crate::network::kademlia::RoutingTable;
    use crate::network::messages::NodeId;
    use crate::node::Node;

    async fn make_router_empty_table() -> QueryRouter {
        let local_id = NodeId([0u8; 32]);
        let node = Arc::new(Node::new());
        let table = Arc::new(Mutex::new(RoutingTable::new(local_id.clone())));
        let reputation = Arc::new(ReputationStore::new());

        rustls::crypto::ring::default_provider().install_default().ok();
        let kp = crate::crypto::identity::NodeKeypair::generate();
        let transport =
            crate::network::quic::QuicTransport::bind("127.0.0.1:0".parse().unwrap(), &kp)
                .await
                .unwrap();
        let keypair = Arc::new(kp);

        QueryRouter::new(node, table, Arc::new(transport), local_id, reputation, keypair)
    }

    #[test]
    fn reputation_arc_shared_with_caller() {
        let reputation = Arc::new(ReputationStore::new());
        let reputation_clone = Arc::clone(&reputation);
        assert!(Arc::ptr_eq(&reputation, &reputation_clone));
    }

    #[tokio::test]
    async fn empty_terms_returns_empty() {
        let router = make_router_empty_table().await;
        let results = router.route_query(vec![], 10, 0).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn empty_routing_table_routes_all_locally() {
        let router = make_router_empty_table().await;

        // Index some documents into the node.
        router.node.index_document(0, "rust async await futures");
        router.node.index_document(1, "python scripting dynamic");

        let results = router.route_query(vec!["rust".to_string()], 10, 1).await;
        // Should find doc 0 locally.
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }

    #[tokio::test]
    async fn results_sorted_by_score_descending() {
        let router = make_router_empty_table().await;
        router.node.index_document(0, "rust rust rust");
        router.node.index_document(1, "rust python");
        router.node.index_document(2, "python only");

        let results = router.route_query(vec!["rust".to_string()], 10, 2).await;
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "results not sorted by score: {:?}",
                results
            );
        }
    }
}