use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::crawler::engine::{Crawler, CrawlerConfig};
use crate::network::query_router::QueryRouter;
use crate::node::Node;

// ---------------------------------------------------------------------------
// HTTP server — exposes /search, /health, /stats, /crawl for the gateway.
//
// Gateway↔Node uses HTTP (Bun has no native QUIC support).
// Node↔Node uses QUIC via QuicTransport.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<QueryRouter>,
    pub node: Arc<Node>,
}

#[derive(Deserialize)]
pub struct SearchParams {
    pub q: String,
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub doc_id: u32,
    pub score: f32,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/search", get(search_handler))
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .route("/crawl", post(crawl_handler))
        .with_state(state)
}

async fn search_handler(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> Json<Vec<SearchResult>> {
    let limit = params.limit.unwrap_or(10).min(100);
    // Simple whitespace tokenization — consistent with gateway splitting.
    let terms: Vec<String> = params
        .q
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();

    let results = state.router.route_query(terms, limit, 0).await;
    Json(results.into_iter().map(|(doc_id, score)| SearchResult { doc_id, score }).collect())
}

async fn health_handler() -> Json<HashMap<&'static str, &'static str>> {
    let mut map = HashMap::new();
    map.insert("status", "ok");
    Json(map)
}

#[derive(Serialize)]
struct StatsResponse {
    doc_count: u64,
    vocab_size: usize,
    pagerank_ready: bool,
}

async fn stats_handler(State(state): State<AppState>) -> Json<StatsResponse> {
    Json(StatsResponse {
        doc_count: state.node.doc_count(),
        vocab_size: state.node.vocab_size(),
        pagerank_ready: state.node.pagerank_ready(),
    })
}

#[derive(Deserialize)]
struct CrawlBody {
    seeds: Vec<String>,
}

async fn crawl_handler(
    State(state): State<AppState>,
    Json(body): Json<CrawlBody>,
) -> StatusCode {
    let urls: Vec<Url> = body
        .seeds
        .iter()
        .filter_map(|s| Url::parse(s).ok())
        .collect();

    if urls.is_empty() {
        return StatusCode::ACCEPTED;
    }

    let node = Arc::clone(&state.node);
    tokio::spawn(async move {
        match Crawler::new(node, CrawlerConfig::default()) {
            Ok(crawler) => {
                if let Err(e) = crawler.run(urls).await {
                    tracing::error!(error = %e, "crawl task failed");
                }
            }
            Err(e) => tracing::error!(error = %e, "crawler init failed"),
        }
    });

    StatusCode::ACCEPTED
}