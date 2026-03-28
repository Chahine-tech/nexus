use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::crawler::engine::{Crawler, CrawlerConfig};
use crate::network::gossip::GossipEngine;
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
    pub gossip: Arc<GossipEngine>,
    /// Directory where persistent data (index, vector index) is stored.
    pub data_dir: std::path::PathBuf,
}

#[derive(Deserialize)]
pub struct SearchParams {
    pub q: String,
    pub limit: Option<usize>,
    /// HNSW beam width for vector ANN search. Higher = better recall, more latency.
    /// Defaults to `(limit * 4).max(50)` when absent.
    pub ef_search: Option<usize>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub doc_id: u32,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/search", get(search_handler))
        .route("/search/local", get(local_search_handler))
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .route("/crawl", post(crawl_handler))
        .route("/index", post(index_handler))
        .route("/rebuild-vector", post(rebuild_vector_handler))
        .route("/export-shard", get(export_shard_handler))
        .route("/merge-shard", post(merge_shard_handler))
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

    let results = state.router.route_query(terms, limit, 0, params.ef_search).await;
    Json(
        results
            .into_iter()
            .map(|(doc_id, score)| {
                let url = state.node.url_for_doc(doc_id);
                SearchResult { doc_id, score, url }
            })
            .collect(),
    )
}

/// BM25-only local search — bypasses the distributed router.
///
/// Used internally for shard-level searches and as a fallback when the
/// routing table is empty. Calls `Node::search` directly.
async fn local_search_handler(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> Json<Vec<SearchResult>> {
    let limit = params.limit.unwrap_or(10).min(100);
    let results = state.node.search(&params.q, limit);
    Json(
        results
            .into_iter()
            .map(|(doc_id, score)| {
                let url = state.node.url_for_doc(doc_id);
                SearchResult { doc_id, score, url }
            })
            .collect(),
    )
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
    /// Embedding dimension of the HNSW vector index (0 if not built yet).
    vector_dim: usize,
    /// Top-5 documents by local PageRank score.
    top_pagerank: Vec<(u32, f32)>,
    /// Estimated global distinct-term cardinality from merged HyperLogLog sketches.
    estimated_global_terms: f64,
    /// Number of peer nodes known to the gossip engine.
    peer_count: usize,
}

async fn stats_handler(State(state): State<AppState>) -> Json<StatsResponse> {
    let top_pagerank = state.node.pagerank_ranked().into_iter().take(5).collect();
    let peer_count = state.gossip.peer_states().len();
    let estimated_global_terms = state.gossip.estimated_cardinality();

    // Sync local doc_count into gossip, then propagate global N into BM25 scorer.
    state.gossip.update_local(state.node.doc_count());
    state.node.update_global_doc_count(state.gossip.global_doc_count());

    Json(StatsResponse {
        doc_count: state.node.doc_count(),
        vocab_size: state.node.vocab_size(),
        pagerank_ready: state.node.pagerank_ready(),
        vector_dim: state.node.vector_dim(),
        top_pagerank,
        estimated_global_terms,
        peer_count,
    })
}

// ---------------------------------------------------------------------------
// Shard export/import — used by the gateway for rebalancing on node death.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ExportShardParams {
    term: String,
}

/// Returns the raw msgpack bytes of the posting list for `term`.
/// Gateway calls this on the dead node before routing to the successor.
async fn export_shard_handler(
    State(state): State<AppState>,
    Query(params): Query<ExportShardParams>,
) -> Result<axum::response::Response, StatusCode> {
    use axum::body::Body;
    use axum::http::header;

    match state.node.export_posting_shard(&params.term) {
        Some(bytes) => Ok(axum::response::Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(Body::from(bytes))
            .unwrap()),
        None => Err(StatusCode::NOT_FOUND),
    }
}

#[derive(Deserialize)]
struct MergeShardBody {
    term: String,
    /// base64-encoded msgpack PostingList bytes.
    posting_b64: String,
}

/// Merges a remote posting list shard into this node's inverted index.
/// Gateway calls this on the successor node during rebalancing.
async fn merge_shard_handler(
    State(state): State<AppState>,
    Json(body): Json<MergeShardBody>,
) -> StatusCode {
    use base64::Engine as _;
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&body.posting_b64) else {
        return StatusCode::BAD_REQUEST;
    };
    let Ok(posting) = rmp_serde::from_slice(&bytes) else {
        return StatusCode::BAD_REQUEST;
    };
    let _ = state.node.merge_posting_shard(&body.term, posting);
    StatusCode::NO_CONTENT
}

#[derive(Deserialize)]
struct IndexBody {
    url: String,
    /// Short identifier field (e.g. crate name, page title). Indexed with boost=3.0.
    /// When absent, falls back to flat `text`-only indexing.
    name: Option<String>,
    /// Body text (description, content, keywords). Indexed with boost=1.0.
    /// Also accepts legacy `text` field for backward compatibility.
    #[serde(alias = "text")]
    body: String,
}

/// Indexes a document directly from pre-extracted text, bypassing the crawler.
///
/// Accepts two forms:
///   - Fielded: `{"url": "...", "name": "serde", "body": "serialization framework"}`
///   - Legacy:  `{"url": "...", "text": "serde serialization framework"}`
///
/// Fielded indexing enables per-field BM25 boost (name×3, body×1), which
/// significantly improves named-entity retrieval (MRR@10).
async fn index_handler(
    State(state): State<AppState>,
    Json(body): Json<IndexBody>,
) -> StatusCode {
    if url::Url::parse(&body.url).is_err() {
        return StatusCode::BAD_REQUEST;
    }
    let text_for_hll = match &body.name {
        Some(name) => {
            state.node.index_url_fields(&body.url, name, &body.body);
            format!("{name} {}", body.body)
        }
        None => {
            state.node.index_url(&body.url, &body.body);
            body.body.clone()
        }
    };
    // Feed new terms into the local HLL sketch immediately so estimated_global_terms
    // reflects the current index without waiting for the 30s gossip tick.
    for term in state.node.tokenize(&text_for_hll) {
        state.gossip.add_term(&term);
    }
    // Sync local doc_count into gossip so global_doc_count() is accurate immediately.
    state.gossip.update_local(state.node.doc_count());
    state.node.update_global_doc_count(state.gossip.global_doc_count());
    StatusCode::OK
}

/// Triggers a full vector index rebuild from all currently indexed documents.
///
/// Blocks until the rebuild is complete (fastembed batch embed + HNSW insert).
/// Returns 200 OK with the number of documents embedded.
async fn rebuild_vector_handler(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(e) = state.node.rebuild_vector_index().await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        );
    }
    if let Err(e) = state.node.save_vector_index(&state.data_dir).await {
        tracing::warn!(error = %e, "vector index persist failed after rebuild");
    } else {
        tracing::info!("vector index persisted to disk after rebuild");
    }
    let doc_count = state.node.doc_count();
    (StatusCode::OK, Json(serde_json::json!({ "ok": true, "docs_embedded": doc_count })))
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
                tracing::info!(visited = crawler.visited_count(), "crawl complete");
            }
            Err(e) => tracing::error!(error = %e, "crawler init failed"),
        }
    });

    StatusCode::ACCEPTED
}