mod crawler;
mod indexer;
mod ast;
mod scoring;
mod network;
mod pagerank;
mod crypto;
mod sketch;
mod node;

use std::sync::Arc;

use node::Node;
use network::messages::{MessageType, NetworkMessage, QueryRequest, encode_message, decode_message};
use network::quic::QuicTransport;
use crypto::identity::NodeKeypair;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Install ring TLS provider — must be called once before any rustls operation.
    rustls::crypto::ring::default_provider().install_default().ok();

    tracing::info!("Nexus node starting");

    // --- Local search smoke test ---
    let node = Arc::new(Node::new());
    node.index_document(0, "Rust is a systems programming language focused on safety and performance");
    node.index_document(1, "Python is a dynamic language great for scripting and data science");
    node.index_document(2, "Rust enables fearless concurrency without data races");

    let results = node.search("rust concurrency", 10);
    tracing::info!(?results, "search results");

    // --- QUIC two-node smoke test ---
    let kp_a = NodeKeypair::generate();
    let kp_b = NodeKeypair::generate();

    let node_a = QuicTransport::bind("127.0.0.1:0".parse()?, &kp_a).await?;
    let node_b = QuicTransport::bind("127.0.0.1:0".parse()?, &kp_b).await?;
    let addr_b = node_b.endpoint.local_addr()?;

    let qr = QueryRequest { terms: vec!["rust".to_string()], limit: 10, request_id: 1 };
    let payload = encode_message(&qr)?;
    let msg = NetworkMessage {
        kind: MessageType::QueryRequest,
        payload,
        sender: kp_a.node_id(),
        signature: [0u8; 64],
    };

    let (conn_res, accept_res) =
        tokio::join!(node_a.connect(addr_b), node_b.accept());

    let conn_a = conn_res?;
    let conn_b = accept_res?;

    let (send_res, recv_res) =
        tokio::join!(QuicTransport::send(&conn_a, &msg), QuicTransport::recv(&conn_b));

    send_res?;
    let received = recv_res?;

    let decoded_qr: QueryRequest = decode_message(&received.payload)?;
    tracing::info!(
        sender = ?received.sender,
        request_id = decoded_qr.request_id,
        terms = ?decoded_qr.terms,
        "QUIC smoke test: message received"
    );

    Ok(())
}
