use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::network::query_router::QueryRouter;

// ---------------------------------------------------------------------------
// HTTP server — exposes /search and /health for the gateway to call.
//
// Gateway↔Node uses HTTP (Bun has no native QUIC support).
// Node↔Node uses QUIC via QuicTransport.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<QueryRouter>,
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