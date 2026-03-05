mod crawler;
mod indexer;
mod ast;
mod scoring;
mod network;
mod pagerank;
mod crypto;
mod sketch;
mod node;
mod http;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use node::Node;
use network::kademlia::RoutingTable;
use network::gossip::GossipEngine;
use network::quic::QuicTransport;
use network::query_router::QueryRouter;
use crypto::identity::NodeKeypair;
use http::{AppState, build_router};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Must be called once before any rustls/QUIC operation.
    rustls::crypto::ring::default_provider().install_default().ok();

    let quic_port: u16 = std::env::var("NEXUS_QUIC_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9000);
    let http_port: u16 = std::env::var("NEXUS_HTTP_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3001);

    tracing::info!(quic_port, http_port, "Nexus node starting");

    let keypair = NodeKeypair::generate();
    let local_id = keypair.node_id();

    let quic_addr: SocketAddr = format!("0.0.0.0:{quic_port}").parse()?;
    let transport = Arc::new(QuicTransport::bind(quic_addr, &keypair).await?);
    tracing::info!(addr = %transport.endpoint.local_addr()?, "QUIC endpoint bound");

    let routing_table = Arc::new(Mutex::new(RoutingTable::new(local_id.clone())));

    let search_node = Arc::new(Node::new());
    search_node.index_document(0, "Rust is a systems programming language focused on safety and performance");
    search_node.index_document(1, "Python is a dynamic language great for scripting and data science");
    search_node.index_document(2, "Rust enables fearless concurrency without data races");

    let query_router = Arc::new(QueryRouter::new(
        Arc::clone(&search_node),
        Arc::clone(&routing_table),
        Arc::clone(&transport),
        local_id.clone(),
    ));

    let gossip = Arc::new(GossipEngine::new(local_id, Arc::clone(&transport)));
    gossip.update_local(search_node.index.doc_count());

    // Gossip loop — broadcasts HLL sketches every 30 seconds.
    // Peer list is empty until Kademlia bootstrap runs (week 4 integration).
    let gossip_loop = {
        let gossip = Arc::clone(&gossip);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                if let Err(e) = gossip.broadcast_idf(&[]).await {
                    tracing::warn!(error = %e, "gossip broadcast failed");
                }
            }
        })
    };

    // HTTP server — gateway calls /search and /health on this endpoint.
    let http_addr: SocketAddr = format!("0.0.0.0:{http_port}").parse()?;
    let app = build_router(AppState { router: query_router });
    let listener = tokio::net::TcpListener::bind(http_addr).await?;
    tracing::info!(addr = %listener.local_addr()?, "HTTP server listening");

    let http_server = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    tokio::select! {
        _ = gossip_loop => tracing::warn!("gossip loop exited"),
        _ = http_server => tracing::warn!("HTTP server exited"),
    }

    Ok(())
}
