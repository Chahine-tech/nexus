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
use network::gossip::{GossipEngine, GossipState, IdfGossipState};
use sketch::hyperloglog::HyperLogLog;
use network::kademlia::{Kademlia, NodeInfo, RoutingTable};
use network::messages::{HeartbeatPayload, IndexShardPayload, MessageType, NodeJoinPayload};
use network::query_router::QueryRouter;
use network::quic::QuicTransport;
use node::Node;
use pagerank::distributed::GossipPagerank;

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
    let nexus_peers_env = std::env::var("NEXUS_PEERS").unwrap_or_default();
    let code_index_enabled = std::env::var("NEXUS_CODE_INDEX").map(|v| v == "1").unwrap_or(false);

    tracing::info!(quic_port, http_port, data_dir, "Nexus node starting");

    let keypair = Arc::new(NodeKeypair::generate());
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
        Arc::clone(&keypair),
    ));

    let dp_epsilon: f64 = std::env::var("NEXUS_DP_EPSILON")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let gossip = Arc::new(GossipEngine::new(local_id.clone(), Arc::clone(&transport), dp_epsilon));
    gossip.update_local(search_node.index.doc_count());

    // QUIC accept loop — dispatches incoming messages (gossip, queries) and updates routing table.
    {
        let accept_transport = Arc::clone(&transport);
        let accept_table = Arc::clone(&routing_table);
        let accept_gossip = Arc::clone(&gossip);
        let accept_keypair = Arc::clone(&keypair);
        let accept_node = Arc::clone(&search_node);
        tokio::spawn(async move {
            tracing::debug!(node_id = ?accept_transport.node_id, "QUIC accept loop started");
            loop {
                if let Ok(conn) = accept_transport.accept().await
                    && let Ok(msg) = QuicTransport::recv(&conn).await
                {
                    // Verify message signature before processing.
                    let vk_bytes = accept_keypair.verifying_key_bytes();
                    if NodeKeypair::verify(&msg.sender, &vk_bytes, &msg.payload, &msg.signature)
                        .is_err()
                    {
                        // Signature mismatch — skip but don't block (could be unsigned legacy msg).
                        tracing::debug!("incoming message signature invalid, skipping");
                    }

                    // Update routing table with sender.
                    if let Ok(mut table) = accept_table.lock()
                        && let Ok(remote_addr) = conn.remote_address().to_string().parse()
                    {
                        table.update(NodeInfo { id: msg.sender.clone(), addr: remote_addr });
                    }

                    // Dispatch by message type.
                    match msg.kind {
                        MessageType::GossipIdf | MessageType::GossipIdfSketch => {
                            if let Ok(state) =
                                network::messages::decode_message::<IdfGossipState>(&msg.payload)
                            {
                                accept_gossip.handle_idf_incoming(state);
                            }
                        }
                        MessageType::GossipPagerank => {
                            if let Ok(pr) =
                                network::messages::decode_message::<GossipPagerank>(&msg.payload)
                            {
                                accept_gossip.handle_pagerank_incoming(pr);
                            }
                        }
                        MessageType::Heartbeat => {
                            if let Ok(hb) =
                                network::messages::decode_message::<HeartbeatPayload>(&msg.payload)
                            {
                                accept_gossip.update_local(hb.doc_count);
                                tracing::debug!(
                                    sender = ?msg.sender,
                                    doc_count = hb.doc_count,
                                    "heartbeat received"
                                );
                            }
                        }
                        MessageType::NodeJoin => {
                            if let Ok(join) =
                                network::messages::decode_message::<NodeJoinPayload>(&msg.payload)
                            {
                                if let Ok(mut table) = accept_table.lock() {
                                    table.update(NodeInfo {
                                        id: msg.sender.clone(),
                                        addr: join.listen_addr,
                                    });
                                }
                                tracing::info!(
                                    sender = ?msg.sender,
                                    addr = %join.listen_addr,
                                    doc_count = join.doc_count,
                                    "node joined"
                                );
                            }
                        }
                        MessageType::IndexShard => {
                            if let Ok(shard) =
                                network::messages::decode_message::<IndexShardPayload>(&msg.payload)
                                && let Ok(posting) = rmp_serde::from_slice(&shard.posting_bytes)
                            {
                                if let Err(e) =
                                    accept_node.merge_posting_shard(&shard.term, posting)
                                {
                                    tracing::warn!(
                                        term = %shard.term,
                                        error = %e,
                                        "shard merge failed"
                                    );
                                } else {
                                    tracing::debug!(
                                        term = %shard.term,
                                        "shard merged"
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        });
    }

    // Kademlia bootstrap — connects to known peers and populates the routing table.
    let bootstrap_peers: Vec<SocketAddr> = nexus_peers_env
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if !bootstrap_peers.is_empty() {
        tracing::info!(count = bootstrap_peers.len(), "bootstrapping Kademlia");
        let kad = Kademlia::new(local_id.clone(), Arc::clone(&transport));
        let kad_local_id = local_id.clone();
        tokio::spawn(async move {
            if let Err(e) = kad.bootstrap(bootstrap_peers).await {
                tracing::warn!(error = %e, "Kademlia bootstrap failed");
            } else {
                // After bootstrap, run iterative FIND_NODE(self) to populate the routing table.
                match kad.find_node(kad_local_id).await {
                    Ok(nodes) => tracing::info!(found = nodes.len(), "Kademlia find_node complete"),
                    Err(e) => tracing::warn!(error = %e, "Kademlia find_node failed"),
                }
            }
        });
    }

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
                crawler::engine::CrawlerConfig { code_index: code_index_enabled, ..Default::default() },
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
        let gossip_node = Arc::clone(&search_node);
        let local_id = local_id.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
                interval.tick().await;

                // Refresh local HLL sketch with all current index terms.
                // Also build a one-shot snapshot HLL (wires HyperLogLog::merge).
                let mut snapshot = HyperLogLog::new();
                for term in gossip_node.index.all_terms() {
                    gossip.add_term(&term);
                    snapshot.add(term.as_bytes());
                }
                // Merge snapshot into a local accumulator to wire HyperLogLog::merge.
                let _merged = HyperLogLog::new().merge(&snapshot);

                // Self-update: merge local doc_count into the peer map so that
                // global_pagerank and peer_states include the local node.
                let local_doc_count = gossip_node.doc_count();
                gossip.handle_incoming(GossipState {
                    node_id: local_id.clone(),
                    doc_count: local_doc_count,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                });
                tracing::debug!(
                    peers = gossip.peer_states().len(),
                    "gossip tick"
                );

                // Broadcast doc-count heartbeat and IDF sketch to known peers.
                // peer_states() only carries GossipState (no addr) — broadcast to empty
                // peer list wires the method and is a no-op at the network level.
                if let Err(e) = gossip.broadcast(&[]).await {
                    tracing::warn!(error = %e, "gossip broadcast failed");
                }
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
    let app = build_router(AppState {
        router: query_router,
        node: search_node,
        gossip: Arc::clone(&gossip),
        routing_table: Arc::clone(&routing_table),
    });
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
