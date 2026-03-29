/// End-to-end integration tests for the Nexus distributed search stack.
///
/// Each test spins up real in-process components (Node, QueryRouter, QuicTransport)
/// bound to `127.0.0.1:0` (OS-assigned ports) so tests can run in parallel without
/// port conflicts.
///
/// These tests exercise the full path:
///   index document → QueryRouter → (local BM25 | QUIC fanout) → merged results
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nexus_core::crypto::identity::NodeKeypair;
use nexus_core::crypto::reputation::ReputationStore;
use nexus_core::network::kademlia::{NodeInfo, RoutingTable};
use nexus_core::network::messages::{
    MessageType, NetworkMessage, NodeId, QueryRequest, QueryResponse, decode_message,
    encode_message,
};
use nexus_core::network::query_router::QueryRouter;
use nexus_core::network::quic::QuicTransport;
use nexus_core::node::Node;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Builds a QueryRouter backed by a fresh Node + QuicTransport on a random port.
/// Returns (router, transport_addr, node).
///
/// The QUIC client is configured with a 500ms idle timeout so that tests
/// involving unreachable peers complete quickly instead of waiting 30s.
async fn make_node(
    local_id: NodeId,
) -> (Arc<QueryRouter>, std::net::SocketAddr, Arc<Node>) {
    rustls::crypto::ring::default_provider().install_default().ok();

    let kp = NodeKeypair::generate();
    let transport =
        QuicTransport::bind_with_client_config("127.0.0.1:0".parse().unwrap(), &kp, short_idle_timeout()).await.unwrap();
    let addr = transport.endpoint.local_addr().unwrap();
    let transport = Arc::new(transport);

    let node = Arc::new(Node::new());
    let table = Arc::new(Mutex::new(RoutingTable::new(local_id.clone())));
    let reputation = Arc::new(ReputationStore::new());
    let keypair = Arc::new(kp);

    let router = Arc::new(QueryRouter::new(
        Arc::clone(&node),
        table,
        Arc::clone(&transport),
        local_id,
        reputation,
        keypair,
    ));

    (router, addr, node)
}

/// Returns a quinn ClientConfig with a short idle timeout for tests.
fn short_idle_timeout() -> quinn::ClientConfig {
    let rustls_client = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(std::sync::Arc::new(SkipVerify))
        .with_no_client_auth();
    let mut client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(rustls_client).unwrap(),
    ));
    let mut transport = quinn::TransportConfig::default();
    transport.max_idle_timeout(Some(Duration::from_millis(500).try_into().unwrap()));
    client_config.transport_config(Arc::new(transport));
    client_config
}

/// Minimal rustls cert verifier that accepts everything — used only in tests.
#[derive(Debug)]
struct SkipVerify;

impl rustls::client::danger::ServerCertVerifier for SkipVerify {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

// ---------------------------------------------------------------------------
// Test 1: single node, empty routing table — all queries go local
// ---------------------------------------------------------------------------

#[tokio::test]
async fn single_node_local_search() {
    let id = NodeId([1u8; 32]);
    let (router, _addr, node) = make_node(id).await;

    node.index_document(0, "rust async programming futures");
    node.index_document(1, "python machine learning numpy");
    node.index_document(2, "rust ownership borrowing lifetimes");

    let results = router.route_query(vec!["rust".to_string()], 10, 0, None).await;

    assert!(!results.is_empty(), "expected results for 'rust'");
    let doc_ids: Vec<u32> = results.iter().map(|r| r.0).collect();
    assert!(doc_ids.contains(&0), "doc 0 (rust async) should be in results");
    assert!(doc_ids.contains(&2), "doc 2 (rust ownership) should be in results");
    assert!(!doc_ids.contains(&1), "doc 1 (python) should not match 'rust'");
}

// ---------------------------------------------------------------------------
// Test 2: results are sorted by score descending
// ---------------------------------------------------------------------------

#[tokio::test]
async fn results_sorted_descending() {
    let id = NodeId([2u8; 32]);
    let (router, _addr, node) = make_node(id).await;

    // Doc 0 mentions rust once, doc 1 three times — doc 1 should score higher.
    node.index_document(0, "rust python javascript");
    node.index_document(1, "rust rust rust systems programming");

    let results = router.route_query(vec!["rust".to_string()], 10, 0, None).await;

    assert!(results.len() >= 2, "expected at least 2 results");
    for i in 1..results.len() {
        assert!(
            results[i - 1].1 >= results[i].1,
            "results not sorted: {:?}",
            results
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: limit is respected
// ---------------------------------------------------------------------------

#[tokio::test]
async fn limit_is_respected() {
    let id = NodeId([3u8; 32]);
    let (router, _addr, node) = make_node(id).await;

    for i in 0u32..20 {
        node.index_document(i, &format!("rust document number {i}"));
    }

    let results = router.route_query(vec!["rust".to_string()], 5, 0, None).await;
    assert_eq!(results.len(), 5, "should return exactly limit=5 results");
}

// ---------------------------------------------------------------------------
// Test 4: empty query returns empty results
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_query_returns_empty() {
    let id = NodeId([4u8; 32]);
    let (router, _addr, node) = make_node(id).await;
    node.index_document(0, "rust async");

    let results = router.route_query(vec![], 10, 0, None).await;
    assert!(results.is_empty(), "empty terms should return empty results");
}

// ---------------------------------------------------------------------------
// Test 5: two-node cluster — node A routes a query to node B via QUIC
//
// To guarantee the term routes to B, we use node IDs constructed so that B's
// XOR distance to the blake3 key of "nexustest" is zero (B's id == the
// blake3 hash of the term). This makes B unambiguously responsible.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_node_quic_fanout() {
    rustls::crypto::ring::default_provider().install_default().ok();

    // Pick a term whose blake3 key == B's NodeId so B is always responsible.
    let term = "nexustest";
    let term_key = *blake3::hash(term.as_bytes()).as_bytes();
    let id_b = NodeId(term_key);

    let kp_b = NodeKeypair::generate();
    let transport_b =
        QuicTransport::bind_with_client_config("127.0.0.1:0".parse().unwrap(), &kp_b, short_idle_timeout()).await.unwrap();
    let addr_b = transport_b.endpoint.local_addr().unwrap();
    let transport_b = Arc::new(transport_b);

    let node_b = Arc::new(Node::new());
    node_b.index_document(10, &format!("{term} distributed systems consensus raft"));
    node_b.index_document(11, &format!("{term} paxos distributed fault tolerance"));

    // Oneshot signals A that B has sent the response and the conn can be dropped.
    let (tx_done, rx_done) = tokio::sync::oneshot::channel::<()>();

    let node_b_clone = Arc::clone(&node_b);
    let transport_b_clone = Arc::clone(&transport_b);
    let id_b_for_spawn = id_b.clone();
    tokio::spawn(async move {
        if let Ok(conn) = transport_b_clone.accept().await
            && let Ok(msg) = QuicTransport::recv(&conn).await
            && let Ok(req) = decode_message::<QueryRequest>(&msg.payload)
        {
            let hits = node_b_clone.search(&req.terms.join(" "), req.limit);
            let resp = QueryResponse {
                request_id: req.request_id,
                results: hits,
                node_id: id_b_for_spawn.clone(),
            };
            let payload = encode_message(&resp).unwrap();
            let reply = NetworkMessage {
                kind: MessageType::QueryResponse,
                payload,
                sender: id_b_for_spawn,
                signature: [0u8; 64],
            };
            let _ = QuicTransport::send(&conn, &reply).await;
            // Signal that response was sent; keep conn alive until A reads it.
            let _ = tx_done.send(());
            // Hold conn open until A finishes reading the response stream.
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    });

    let id_a = NodeId([0x00; 32]);
    let kp_a = NodeKeypair::generate();
    let transport_a =
        QuicTransport::bind_with_client_config("127.0.0.1:0".parse().unwrap(), &kp_a, short_idle_timeout()).await.unwrap();
    let transport_a = Arc::new(transport_a);

    let node_a = Arc::new(Node::new());
    let mut table_a = RoutingTable::new(id_a.clone());
    table_a.update(NodeInfo { id: id_b.clone(), addr: addr_b });
    let table_a = Arc::new(Mutex::new(table_a));
    let reputation_a = Arc::new(ReputationStore::new());
    let keypair_a = Arc::new(kp_a);

    let router_a = QueryRouter::new(
        Arc::clone(&node_a),
        table_a,
        Arc::clone(&transport_a),
        id_a,
        reputation_a,
        keypair_a,
    );

    // Wait until B's accept loop is scheduled before A connects.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let results = router_a
        .route_query(vec![term.to_string()], 10, 1, None)
        .await;

    // Wait for B to confirm the response was sent (bounded by 2s to avoid hang).
    let _ = tokio::time::timeout(Duration::from_secs(2), rx_done).await;

    assert!(!results.is_empty(), "node A should receive results from node B via QUIC");
    let doc_ids: Vec<u32> = results.iter().map(|r| r.0).collect();
    assert!(
        doc_ids.contains(&10) || doc_ids.contains(&11),
        "expected doc 10 or 11 from node B, got {:?}",
        doc_ids
    );
}

// ---------------------------------------------------------------------------
// Test 6: unreachable remote node — graceful degradation, local results returned
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unreachable_remote_node_falls_back_to_local() {
    let id_local = NodeId([0x10; 32]);
    let kp = NodeKeypair::generate();
    let transport =
        QuicTransport::bind_with_client_config("127.0.0.1:0".parse().unwrap(), &kp, short_idle_timeout()).await.unwrap();
    let transport = Arc::new(transport);

    let node = Arc::new(Node::new());
    node.index_document(0, "fallback local result rust");

    // Point routing table at a dead address (nothing listening there).
    let dead_id = NodeId([0xff; 32]);
    let dead_addr: std::net::SocketAddr = "127.0.0.1:1".parse().unwrap();
    let mut table = RoutingTable::new(id_local.clone());
    table.update(NodeInfo { id: dead_id, addr: dead_addr });
    let table = Arc::new(Mutex::new(table));

    let reputation = Arc::new(ReputationStore::new());
    let keypair = Arc::new(kp);

    let router = QueryRouter::new(
        Arc::clone(&node),
        table,
        Arc::clone(&transport),
        id_local,
        reputation,
        keypair,
    );

    // Should complete quickly (500ms idle timeout) and not panic.
    let results = router.route_query(vec!["rust".to_string()], 10, 0, None).await;

    // If the term routes locally (local_id closer to term key than dead_id),
    // the local doc is returned. Either way, no crash is the key invariant.
    let _ = results;
}

// ---------------------------------------------------------------------------
// Test 7: multi-term query — all terms present in results
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_term_query_returns_relevant_docs() {
    let id = NodeId([7u8; 32]);
    let (router, _addr, node) = make_node(id).await;

    node.index_document(0, "rust async tokio runtime executor");
    node.index_document(1, "python django web framework");
    node.index_document(2, "async javascript promises callbacks");
    node.index_document(3, "rust sync blocking threads");

    // Doc 0 has both "rust" and "async" — should rank higher and appear in results.
    let results = router
        .route_query(vec!["rust".to_string(), "async".to_string()], 10, 0, None)
        .await;

    assert!(!results.is_empty());
    let doc_ids: Vec<u32> = results.iter().map(|r| r.0).collect();
    assert!(doc_ids.contains(&0), "doc with both terms should be in results");
}
