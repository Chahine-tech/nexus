use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use dashmap::DashMap;
use tracing::instrument;

use crate::ast::features;
use crate::ast::normalizer::tokens_from_features;
use crate::ast::parser::{AstError, AstParser};
use crate::indexer::inverted::{InvertedIndex, InvertedIndexError};
use crate::indexer::posting::{PostingError, PostingList};
use crate::indexer::tokenizer::Tokenizer;
use crate::network::kademlia::RoutingTable;
use crate::pagerank::local::LocalPageRank;
use crate::scoring::bm25::Bm25Scorer;
use crate::scoring::hybrid::HybridScorer;
use crate::scoring::vector::{VectorError, VectorIndex};

/// Core search node — owns the index, scorers, and local PageRank graph.
pub struct Node {
    /// Primary (flat) index — used for shard export/import, AST indexing, and legacy paths.
    pub index: Arc<InvertedIndex>,
    /// Dedicated `name` field index. Short documents → low avgdl → high TF_norm for exact matches.
    name_index: Arc<InvertedIndex>,
    /// Dedicated `body` field index (description + keywords). Shares doc_count with `index`.
    body_index: Arc<InvertedIndex>,
    tokenizer: Tokenizer,
    scorer: Bm25Scorer,
    /// HNSW vector index. Wrapped in Arc+RwLock: Arc allows cheap cloning for HybridScorer.
    vector: RwLock<Option<Arc<VectorIndex>>>,
    /// Weight of BM25 vs vector in hybrid search. Range [0.0, 1.0].
    hybrid_alpha: f32,
    pagerank: RwLock<LocalPageRank>,
    /// URL → doc_id mapping for deduplication on insert.
    url_index: DashMap<String, u32>,
    /// doc_id → URL reverse mapping for O(1) lookup at search time.
    doc_index: DashMap<u32, String>,
    /// doc_id → body text for fastembed re-embedding during vector index rebuild.
    doc_text: DashMap<u32, String>,
    /// Shared handle into `scorer.global_n` — updated from gossip.
    global_n: Arc<AtomicU64>,
}

impl Node {
    pub fn new() -> Self {
        let index = Arc::new(InvertedIndex::new());
        Self::from_index(index)
    }

    /// Creates a node from an already-built inverted index (e.g. loaded from disk).
    ///
    /// Uses a single flat index for both `name` and `body` — no field boosting.
    /// Call `index_url_fields` after construction to populate the fielded indexes.
    pub fn from_index(index: Arc<InvertedIndex>) -> Self {
        let name_index = Arc::new(InvertedIndex::new());
        let body_index = Arc::new(InvertedIndex::new());
        let tokenizer = Tokenizer::new();
        let scorer = Bm25Scorer::with_fields(Arc::clone(&name_index), Arc::clone(&body_index));
        let global_n = scorer.global_n_handle();
        Self {
            index,
            name_index,
            body_index,
            tokenizer,
            scorer,
            vector: RwLock::new(None),
            hybrid_alpha: 0.5,
            pagerank: RwLock::new(LocalPageRank::new()),
            url_index: DashMap::new(),
            doc_index: DashMap::new(),
            doc_text: DashMap::new(),
            global_n,
        }
    }

    /// Total number of indexed documents.
    pub fn doc_count(&self) -> u64 {
        self.index.doc_count()
    }

    /// Updates the global document count used for BM25 IDF computation.
    ///
    /// Call this periodically from the gossip loop with `GossipEngine::global_doc_count()`.
    /// When `n` is 0, BM25 falls back to the local doc count.
    pub fn update_global_doc_count(&self, n: u64) {
        self.global_n.store(n, Ordering::Relaxed);
    }

    /// Number of unique terms in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.index.vocabulary_size()
    }

    /// Returns `true` if PageRank has been computed at least once.
    pub fn pagerank_ready(&self) -> bool {
        !self.pagerank.read().expect("pagerank RwLock poisoned").ranked().is_empty()
    }

    /// Indexes a document identified by `url` into separate `name` and `body` fields.
    ///
    /// The doc_id is derived from the URL via FNV-1a 32-bit hash. If the URL was
    /// already indexed, the existing doc_id is returned without re-indexing.
    ///
    /// `name` tokens are indexed into the boosted name field (boost=3.0 in BM25).
    /// `body` tokens are indexed into the body field (boost=1.0).
    /// Both are also merged into the flat `index` for shard export/import compatibility.
    ///
    /// Returns the doc_id used.
    #[instrument(skip(self, name, body), fields(url))]
    pub fn index_url_fields(&self, url: &str, name: &str, body: &str) -> u32 {
        if let Some(existing) = self.url_index.get(url) {
            return *existing;
        }
        let doc_id = fnv1a_32(url.as_bytes());
        self.url_index.insert(url.to_string(), doc_id);
        self.doc_index.insert(doc_id, url.to_string());

        let name_tokens = self.tokenizer.tokenize(name);
        let body_tokens = self.tokenizer.tokenize(body);

        self.name_index.index_document(doc_id, &name_tokens);
        self.body_index.index_document(doc_id, &body_tokens);

        // Flat index = name + body for shard rebalancing compatibility.
        let mut all_tokens = name_tokens.clone();
        all_tokens.extend_from_slice(&body_tokens);
        self.index.index_document(doc_id, &all_tokens);

        // Store body text for fastembed re-embedding during vector index rebuild.
        self.doc_text.insert(doc_id, format!("{name} {body}"));

        tracing::debug!(doc_id, url, name_tokens = name_tokens.len(), body_tokens = body_tokens.len(), "indexed document (fielded)");
        doc_id
    }

    /// Indexes a document identified by `url` (flat, no field separation).
    ///
    /// Falls back to single-field BM25 — use `index_url_fields` for better ranking.
    #[instrument(skip(self, text), fields(url))]
    pub fn index_url(&self, url: &str, text: &str) -> u32 {
        if let Some(existing) = self.url_index.get(url) {
            return *existing;
        }
        let doc_id = fnv1a_32(url.as_bytes());
        self.url_index.insert(url.to_string(), doc_id);
        self.doc_index.insert(doc_id, url.to_string());
        let tokens = self.tokenizer.tokenize(text);
        self.index.index_document(doc_id, &tokens);
        // Also index into body field so scorer has candidates.
        self.body_index.index_document(doc_id, &tokens);
        self.doc_text.insert(doc_id, text.to_string());
        tracing::debug!(doc_id, url, tokens = tokens.len(), "indexed document");
        doc_id
    }

    /// Returns the URL associated with `doc_id`, or `None` if not indexed via `index_url`.
    pub fn url_for_doc(&self, doc_id: u32) -> Option<String> {
        self.doc_index.get(&doc_id).map(|v| v.clone())
    }

    /// Tokenizes `text` using the node's tokenizer. Used by callers that need
    /// the same token stream as indexing (e.g., to feed terms into the HLL sketch).
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenizer.tokenize(text)
    }

    /// Tokenizes `text` and indexes it under `doc_id`.
    ///
    /// Also indexes into `body_index` so the multi-field scorer has candidates.
    #[instrument(skip(self, text), fields(doc_id))]
    pub fn index_document(&self, doc_id: u32, text: &str) {
        let tokens = self.tokenizer.tokenize(text);
        self.index.index_document(doc_id, &tokens);
        self.body_index.index_document(doc_id, &tokens);
        self.doc_text.insert(doc_id, text.to_string());
        tracing::debug!(doc_id, tokens = tokens.len(), "indexed document");
    }

    /// Returns the top `limit` results for `query`, sorted by BM25 score.
    #[instrument(skip(self), fields(terms_count))]
    pub fn search(&self, query: &str, limit: usize) -> Vec<(u32, f32)> {
        let terms = self.tokenizer.tokenize(query);
        self.scorer.search(&terms, limit)
    }

    /// Searches using pre-tokenized terms — avoids double-tokenization in QueryRouter.
    pub fn search_terms(&self, terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        self.scorer.search(terms, limit)
    }

    /// Returns the number of documents indexed for `term`, or 0 if absent.
    ///
    /// Used by shard-merge logic to decide whether to merge a remote posting list
    /// into the local index. Returns an error only if internal serialization fails.
    #[allow(dead_code)]
    pub fn posting_df(&self, term: &str) -> Result<u64, InvertedIndexError> {
        Ok(self.index.lookup(term).map(|pl| pl.len()).unwrap_or(0))
    }

    /// Merges a remote posting list shard for `term` into the local index.
    ///
    /// Used when a node receives a partial shard from a peer during rebalancing.
    /// No-op if `remote` is empty.
    pub fn merge_posting_shard(
        &self,
        term: &str,
        remote: PostingList,
    ) -> Result<(), PostingError> {
        if remote.is_empty() {
            return Ok(());
        }
        self.index.merge_posting(term, remote);
        Ok(())
    }

    /// Returns true if this node is the XOR-closest peer to `term`'s blake3 key.
    ///
    /// Returns true when the routing table is empty (single-node network owns all shards).
    #[allow(dead_code)]
    pub fn responsible_for(&self, term: &str, table: &RoutingTable) -> bool {
        match table.responsible_node(term) {
            Some(closest) => closest.id == table.local_id,
            None => true,
        }
    }

    // -----------------------------------------------------------------------
    // Vector index
    // -----------------------------------------------------------------------

    /// Returns the embedding dimension of the vector index (384 when built, 0 otherwise).
    pub fn vector_dim(&self) -> usize {
        let guard = self.vector.read().expect("vector RwLock poisoned");
        if guard.is_some() { crate::scoring::vector::EMBEDDING_DIM } else { 0 }
    }

    /// (Re-)builds the HNSW vector index using fastembed dense embeddings.
    ///
    /// Collects all indexed body texts, embeds them in batches via BAAI/bge-small-en-v1.5,
    /// and inserts the resulting 384-dim vectors into a fresh HNSW graph.
    ///
    /// Blocking operations (model init + ONNX inference) run on the blocking thread pool.
    /// Must be called after bulk indexing before `search_hybrid` returns vector results.
    #[instrument(skip(self))]
    pub async fn rebuild_vector_index(&self) -> Result<(), VectorError> {
        // Phase 1: initialize model (blocking — may download weights on first call).
        let vi = tokio::task::spawn_blocking(VectorIndex::new)
            .await
            .expect("spawn_blocking panicked")?;

        // Collect (doc_id, text) pairs from the in-memory store.
        let doc_texts: Vec<(u32, String)> = self
            .body_index
            .all_doc_ids()
            .into_iter()
            .filter_map(|id| self.doc_text.get(&id).map(|t| (id, t.clone())))
            .collect();

        // Phase 2: batch embed + insert (blocking — CPU-bound ONNX inference).
        let vi = Arc::new(vi);
        let vi_clone = Arc::clone(&vi);
        tokio::task::spawn_blocking(move || {
            let pairs: Vec<(u32, &str)> =
                doc_texts.iter().map(|(id, t)| (*id, t.as_str())).collect();
            vi_clone.batch_insert(&pairs)
        })
        .await
        .expect("spawn_blocking panicked")?;

        *self.vector.write().expect("vector RwLock poisoned") = Some(vi);
        tracing::info!("vector index rebuilt");
        Ok(())
    }

    /// Hybrid BM25 + vector ANN search.
    ///
    /// Falls back to pure BM25 if the vector index has not been built yet.
    /// Vector search (fastembed ONNX inference) runs on the blocking thread pool.
    ///
    /// `ef_search` controls the HNSW beam width. `None` uses the default `(limit * 4).max(50)`.
    #[instrument(skip(self), fields(limit))]
    pub async fn search_hybrid(&self, query: &str, limit: usize, ef_search: Option<usize>) -> Vec<(u32, f32)> {
        let terms = self.tokenizer.tokenize(query);

        // Scope the RwLockReadGuard so it is dropped before any .await point.
        // RwLockReadGuard is !Send, so it must not be held across an await.
        let vi: Option<Arc<VectorIndex>> = {
            let guard = self.vector.read().expect("vector RwLock poisoned");
            guard.as_ref().map(Arc::clone)
        };

        let Some(vi) = vi else {
            return self.scorer.search(&terms, limit);
        };

        let query_owned = query.to_string();
        let name_idx = Arc::clone(&self.name_index);
        let body_idx = Arc::clone(&self.body_index);
        let hybrid_alpha = self.hybrid_alpha;

        // Snapshot PageRank scores before spawn_blocking (RwLockReadGuard is !Send).
        let pr_scores: std::collections::HashMap<u32, f32> = {
            let guard = self.pagerank.read().expect("pagerank RwLock poisoned");
            guard.scores_snapshot()
        };
        // Only pass PageRank if at least one doc has a non-zero score.
        let has_pagerank = pr_scores.values().any(|&v| v > 0.0);

        tokio::task::spawn_blocking(move || {
            let pr = if has_pagerank { Some(&pr_scores) } else { None };
            if hybrid_alpha == 0.5 {
                // No explicit alpha — let QPP predict the optimal blend.
                HybridScorer::with_fields(name_idx, body_idx, vi)
                    .search_adaptive_with_pagerank_ef(&query_owned, &terms, limit, pr, ef_search)
            } else {
                // Explicit alpha set via NEXUS_HYBRID_ALPHA — honour it.
                HybridScorer::new(
                    Bm25Scorer::with_fields(name_idx, body_idx),
                    vi,
                    hybrid_alpha,
                )
                .search_with_pagerank_ef(&terms, limit, pr, ef_search)
            }
        })
        .await
        .expect("spawn_blocking panicked")
    }

    /// Saves the current HNSW vector index to `<dir>/vector` (two files: `.hnsw.graph` + `.hnsw.data`).
    ///
    /// No-op (returns `Ok(())`) if the vector index has not been built yet.
    /// Blocking operations run on the blocking thread pool.
    #[instrument(skip(self))]
    pub async fn save_vector_index(&self, dir: &std::path::Path) -> Result<(), VectorError> {
        let vi: Option<Arc<VectorIndex>> = {
            let guard = self.vector.read().expect("vector RwLock poisoned");
            guard.as_ref().map(Arc::clone)
        };
        let Some(vi) = vi else {
            tracing::debug!("vector index not built, skipping save");
            return Ok(());
        };
        let dir = dir.to_path_buf();
        tokio::task::spawn_blocking(move || vi.save(&dir, "vector"))
            .await
            .expect("spawn_blocking panicked")
    }

    /// Tries to load a previously saved HNSW vector index from `<dir>/vector`.
    ///
    /// Returns `Ok(true)` if the index was loaded, `Ok(false)` if no saved index exists.
    /// Blocking operations (model init + file I/O) run on the blocking thread pool.
    #[instrument(skip(self))]
    pub async fn try_load_vector_index(&self, dir: &std::path::Path) -> Result<bool, VectorError> {
        let dir = dir.to_path_buf();
        let result = tokio::task::spawn_blocking(move || VectorIndex::load(&dir, "vector"))
            .await
            .expect("spawn_blocking panicked");
        match result {
            Ok(vi) => {
                *self.vector.write().expect("vector RwLock poisoned") = Some(Arc::new(vi));
                tracing::info!("vector index loaded from disk");
                Ok(true)
            }
            Err(VectorError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    // -----------------------------------------------------------------------
    // PageRank
    // -----------------------------------------------------------------------

    /// Adds a directed hyperlink `src → dst` to the local PageRank graph.
    pub fn add_link(&self, src: u32, dst: u32) {
        self.pagerank.write().expect("pagerank RwLock poisoned").add_link(src, dst);
    }

    /// Runs PageRank power iteration and stores the result. Returns iteration count.
    pub fn run_pagerank(&self, max_iter: usize) -> usize {
        self.pagerank.write().expect("pagerank RwLock poisoned").iterate(max_iter)
    }

    /// Returns the PageRank score for `doc_id`. Returns `0.0` if unknown.
    #[allow(dead_code)]
    pub fn pagerank_score(&self, doc_id: u32) -> f32 {
        self.pagerank.read().expect("pagerank RwLock poisoned").score(doc_id)
    }

    /// Returns all `(doc_id, score)` pairs sorted by descending PageRank.
    pub fn pagerank_ranked(&self) -> Vec<(u32, f32)> {
        self.pagerank.read().expect("pagerank RwLock poisoned").ranked()
    }

    /// Returns a snapshot of local PageRank scores for gossip propagation.
    pub fn pagerank_snapshot(&self) -> std::collections::HashMap<u32, f32> {
        self.pagerank.read().expect("pagerank RwLock poisoned").scores_snapshot()
    }

    // -----------------------------------------------------------------------
    // Shard export / import (rebalancing)
    // -----------------------------------------------------------------------

    /// Serializes the posting list for `term` as msgpack bytes.
    ///
    /// Returns `None` if the term is not in this node's index.
    /// Called by the HTTP `/export-shard` endpoint during gateway rebalancing.
    pub fn export_posting_shard(&self, term: &str) -> Option<Vec<u8>> {
        let posting = self.index.lookup(term)?;
        rmp_serde::to_vec(&posting).ok()
    }

    // -----------------------------------------------------------------------
    // Code-file indexing (AST pipeline)
    // -----------------------------------------------------------------------

    /// Indexes a source file using the AST pipeline.
    ///
    /// Detects language from `path` extension, parses with tree-sitter in a
    /// blocking thread (because `AstParser` is `!Send`), then indexes the
    /// extracted function names, type names, imports, and tokenized literals.
    ///
    /// Returns `AstError::UnsupportedLanguage` for non-Rust/TS/Python files.
    #[instrument(skip(self, source), fields(doc_id, path = %path.display()))]
    pub async fn index_code_file(
        &self,
        doc_id: u32,
        path: &Path,
        source: &[u8],
    ) -> Result<(), AstError> {
        let language = AstParser::detect_language(path)?;
        let source_vec = source.to_vec();

        let code_features = tokio::task::spawn_blocking(move || {
            let parser = AstParser::new()?;
            let ast = parser.parse(language, &source_vec)?;
            features::extract(&ast)
        })
        .await
        .expect("spawn_blocking panicked")?;

        let tokens = tokens_from_features(&code_features);
        self.index.index_document(doc_id, &tokens);
        self.body_index.index_document(doc_id, &tokens);
        self.doc_text.insert(doc_id, tokens.join(" "));
        tracing::debug!(doc_id, tokens = tokens.len(), "indexed code file");
        Ok(())
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// FNV-1a 32-bit hash — used as a stable, URL-derived doc_id.
pub(crate) fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash: u32 = 2166136261;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_32_properties() {
        // Stable: same input always yields same output.
        assert_eq!(fnv1a_32(b"hello"), fnv1a_32(b"hello"));
        // Different inputs yield different outputs.
        assert_ne!(fnv1a_32(b"hello"), fnv1a_32(b"world"));
        // Empty input yields the FNV offset basis.
        assert_eq!(fnv1a_32(b""), 2166136261);
    }

    #[test]
    fn tokenize_matches_index_pipeline() {
        let node = Node::new();
        let tokens = node.tokenize("async runtime tokio");
        assert!(tokens.contains(&"async".to_owned()));
        assert!(tokens.contains(&"tokio".to_owned()));
    }

    #[test]
    fn index_url_deduplication() {
        let node = Node::new();
        let id1 = node.index_url("https://example.com/foo", "hello world");
        let id2 = node.index_url("https://example.com/foo", "different text");
        assert_eq!(id1, id2, "same URL must return same doc_id");
    }

    #[test]
    fn from_index_preserves_data() {
        let index = Arc::new(InvertedIndex::new());
        let tokenizer = Tokenizer::new();
        for i in 0..5u32 {
            let text = format!("document {i} about rust systems programming");
            index.index_document(i, &tokenizer.tokenize(&text));
        }
        let node = Node::from_index(Arc::clone(&index));
        assert_eq!(node.doc_count(), 5);
        assert!(node.vocab_size() > 0);
    }

    #[tokio::test]
    async fn index_code_file_rust() {
        let node = Node::new();
        let src = b"fn greet(name: &str) -> String { format!(\"Hello {name}\") }";
        node.index_code_file(0, Path::new("greet.rs"), src)
            .await
            .expect("index code file");
        let results = node.search("greet", 5);
        assert!(!results.is_empty(), "should find doc 0 by function name");
    }
}
