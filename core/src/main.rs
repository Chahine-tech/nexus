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
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use url::Url;

use crypto::identity::NodeKeypair;
use crypto::reputation::ReputationStore;
use http::{AppState, build_router};
use indexer::inverted::InvertedIndex;
use indexer::storage;
use network::gossip::GossipEngine;
use network::kademlia::RoutingTable;
use network::query_router::QueryRouter;
use network::quic::QuicTransport;
use node::Node;

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
    let data_dir = std::env::var("NEXUS_DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let seed_urls_env = std::env::var("NEXUS_SEED_URLS").unwrap_or_default();
    let gateway_url = std::env::var("NEXUS_GATEWAY_URL").ok();

    tracing::info!(quic_port, http_port, data_dir, "Nexus node starting");

    let keypair = NodeKeypair::generate();
    let local_id = keypair.node_id();

    let quic_addr: SocketAddr = format!("0.0.0.0:{quic_port}").parse()?;
    let transport = Arc::new(QuicTransport::bind(quic_addr, &keypair).await?);
    tracing::info!(addr = %transport.endpoint.local_addr()?, "QUIC endpoint bound");

    let reputation = Arc::new(ReputationStore::new());
    let routing_table = Arc::new(Mutex::new(RoutingTable::new(local_id.clone())));

    // Load existing index from disk, or start fresh.
    let data_path = PathBuf::from(&data_dir).join("index.msgpack");
    let index = match storage::load(&data_path) {
        Ok(Some(idx)) => {
            tracing::info!(doc_count = idx.doc_count(), "loaded index from disk");
            Arc::new(idx)
        }
        Ok(None) => {
            tracing::info!("no existing index found, starting fresh");
            Arc::new(InvertedIndex::new())
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to load index, starting fresh");
            Arc::new(InvertedIndex::new())
        }
    };
    let search_node = Arc::new(Node::from_index(index));

    let query_router = Arc::new(QueryRouter::new(
        Arc::clone(&search_node),
        Arc::clone(&routing_table),
        Arc::clone(&transport),
        local_id.clone(),
        Arc::clone(&reputation),
    ));

    let gossip = Arc::new(GossipEngine::new(local_id.clone(), Arc::clone(&transport)));
    gossip.update_local(search_node.index.doc_count());

    // Spawn crawler task if seed URLs are provided.
    let seed_urls: Vec<Url> = seed_urls_env
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| Url::parse(s.trim()).ok())
        .collect();

    if !seed_urls.is_empty() {
        tracing::info!(count = seed_urls.len(), "starting crawler");
        let crawl_node = Arc::clone(&search_node);
        let crawl_gossip = Arc::clone(&gossip);
        let crawl_path = data_path.clone();

        tokio::spawn(async move {
            match crawler::engine::Crawler::new(
                Arc::clone(&crawl_node),
                crawler::engine::CrawlerConfig::default(),
            ) {
                Ok(c) => {
                    if let Err(e) = c.run(seed_urls).await {
                        tracing::error!(error = %e, "crawler failed");
                        return;
                    }
                    tracing::info!(
                        doc_count = crawl_node.doc_count(),
                        "crawl complete, rebuilding vector index"
                    );

                    if let Err(e) = crawl_node.rebuild_vector_index() {
                        tracing::warn!(error = %e, "vector index rebuild failed");
                    }

                    let iters = crawl_node.run_pagerank(100);
                    tracing::info!(iters, "pagerank iteration complete");

                    let pr_snapshot = crawl_node.pagerank_snapshot();
                    crawl_gossip.update_pagerank(pr_snapshot);

                    if let Err(e) = storage::save(&crawl_node.index, &crawl_path) {
                        tracing::error!(error = %e, "failed to persist index");
                    } else {
                        tracing::info!("index persisted to disk");
                    }
                }
                Err(e) => tracing::error!(error = %e, "crawler init failed"),
            }
        });
    }

    // Self-register with gateway if configured.
    if let Some(gw_url) = gateway_url {
        let node_id_hex = hex::encode(local_id.0);
        let advertised_url = std::env::var("NEXUS_ADVERTISED_URL")
            .unwrap_or_else(|_| format!("http://127.0.0.1:{http_port}"));
        let gw_url_clone = gw_url.clone();

        tokio::spawn(async move {
            let body = serde_json::json!({
                "nodeId": node_id_hex,
                "url": advertised_url,
            });
            match reqwest::Client::new()
                .post(format!("{gw_url_clone}/nodes/register"))
                .json(&body)
                .send()
                .await
            {
                Ok(resp) => tracing::info!(status = %resp.status(), "registered with gateway"),
                Err(e) => tracing::warn!(error = %e, "gateway self-registration failed"),
            }
        });
    }

    // Gossip loop — broadcasts HLL sketches and PageRank every 30 seconds.
    let gossip_loop = {
        let gossip = Arc::clone(&gossip);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                if let Err(e) = gossip.broadcast_idf(&[]).await {
                    tracing::warn!(error = %e, "IDF gossip broadcast failed");
                }
                if let Err(e) = gossip.broadcast_pagerank(&[]).await {
                    tracing::warn!(error = %e, "PageRank gossip broadcast failed");
                }
            }
        })
    };

    // HTTP server — gateway calls /search, /health, /stats, /crawl.
    let http_addr: SocketAddr = format!("0.0.0.0:{http_port}").parse()?;
    let app = build_router(AppState { router: query_router, node: search_node });
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
