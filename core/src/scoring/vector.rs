use std::path::Path;
use std::sync::{Arc, Mutex};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::{AnnT, DistCosine, HnswIo};

/// Errors produced by `VectorIndex`.
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("embedding model initialization failed: {0}")]
    ModelInit(String),
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),
    #[error("embedder mutex poisoned")]
    MutexPoisoned,
    #[error("HNSW dump failed: {0}")]
    DumpFailed(String),
    #[error("HNSW reload failed: {0}")]
    ReloadFailed(String),
    #[error("vector index file not found at {0}")]
    NotFound(String),
}

/// HNSW-backed approximate nearest-neighbour index using dense embeddings.
///
/// Uses BAAI/bge-small-en-v1.5 (384-dim) via fastembed, running locally via ONNX.
/// No API calls. Model weights (~130 MB) are downloaded once and cached by fastembed.
///
/// Embedding is synchronous and CPU-bound. In async contexts, wrap calls to
/// `new()`, `batch_insert()`, and `search()` with `tokio::task::spawn_blocking`.
pub struct VectorIndex {
    /// HNSW graph over 384-dim f32 vectors with cosine distance.
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// fastembed model. embed() requires &mut self — wrapped in Mutex for shared access.
    embedder: Arc<Mutex<TextEmbedding>>,
}

/// Fixed embedding dimension for BAAI/bge-small-en-v1.5.
pub const EMBEDDING_DIM: usize = 384;

impl VectorIndex {
    /// Initializes the fastembed model and returns an empty HNSW graph.
    ///
    /// Blocking: downloads model weights on first call (~130 MB, then cached).
    /// In async context, call via `tokio::task::spawn_blocking(VectorIndex::new)`.
    pub fn new() -> Result<Self, VectorError> {
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGESmallENV15))
            .map_err(|e| VectorError::ModelInit(e.to_string()))?;

        // max_nb_connection=16, max_elements=100_000, max_layer=16, ef_construction=200
        let hnsw = Hnsw::new(16, 100_000, 16, 200, DistCosine);

        Ok(Self { hnsw, embedder: Arc::new(Mutex::new(model)) })
    }

    /// Embeds `text` and inserts `doc_id` into the HNSW graph.
    ///
    /// Blocking. Prefer `batch_insert` for bulk operations.
    #[allow(dead_code)]
    pub fn insert(&self, doc_id: u32, text: &str) -> Result<(), VectorError> {
        let embedding = self.embed_one(text)?;
        self.hnsw.insert_slice((&embedding, doc_id as usize));
        Ok(())
    }

    /// Embeds all `(doc_id, text)` pairs in batches and inserts them into the HNSW graph.
    ///
    /// Acquires the embedder mutex once for the entire batch — significantly faster than
    /// calling `insert()` in a loop (one ONNX inference per batch of 64 vs one per doc).
    pub fn batch_insert(&self, docs: &[(u32, &str)]) -> Result<(), VectorError> {
        if docs.is_empty() {
            return Ok(());
        }
        let texts: Vec<&str> = docs.iter().map(|(_, t)| *t).collect();
        let embeddings = {
            let mut guard =
                self.embedder.lock().map_err(|_| VectorError::MutexPoisoned)?;
            guard
                .embed(texts, Some(64))
                .map_err(|e| VectorError::EmbeddingFailed(e.to_string()))?
        };
        for ((doc_id, _), emb) in docs.iter().zip(embeddings) {
            self.hnsw.insert_slice((&emb, *doc_id as usize));
        }
        Ok(())
    }

    /// Embeds `query` and performs approximate nearest-neighbour search.
    ///
    /// Returns up to `limit` `(doc_id, cosine_similarity)` pairs sorted by descending
    /// similarity. Returns empty if HNSW has no insertions or embedding fails.
    ///
    /// Blocking. In async context, wrap with `tokio::task::spawn_blocking`.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(u32, f32)> {
        if limit == 0 || query.is_empty() {
            return Vec::new();
        }
        let embedding = match self.embed_one(query) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "vector search: embed failed");
                return Vec::new();
            }
        };
        let ef_search = (limit * 4).max(50);
        let neighbours = self.hnsw.search(&embedding, limit, ef_search);
        neighbours.into_iter().map(|n| (n.d_id as u32, 1.0 - n.distance)).collect()
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Serializes the HNSW graph to disk.
    ///
    /// Writes two files: `<dir>/<basename>.hnsw.graph` and `<dir>/<basename>.hnsw.data`.
    /// The directory must exist. Overwrites any pre-existing files with the same basename.
    ///
    /// Blocking. In async context, wrap with `tokio::task::spawn_blocking`.
    pub fn save(&self, dir: &Path, basename: &str) -> Result<(), VectorError> {
        self.hnsw
            .file_dump(dir, basename)
            .map(|_| ())
            .map_err(|e| VectorError::DumpFailed(e.to_string()))
    }

    /// Loads a previously saved HNSW graph from disk and initializes the fastembed model.
    ///
    /// Expects `<dir>/<basename>.hnsw.graph` and `<dir>/<basename>.hnsw.data` to exist.
    /// Returns `VectorError::NotFound` if the graph file is absent (not yet built).
    ///
    /// Blocking. In async context, wrap with `tokio::task::spawn_blocking`.
    pub fn load(dir: &Path, basename: &str) -> Result<Self, VectorError> {
        let graph_path = dir.join(format!("{basename}.hnsw.graph"));
        if !graph_path.exists() {
            return Err(VectorError::NotFound(graph_path.display().to_string()));
        }

        // WHY Box::leak: `HnswIo::load_hnsw` has an `'a: 'b` lifetime bound that ties
        // the returned `Hnsw<'b>` to the `HnswIo` reloader (`'a`). This bound exists
        // because when mmap is enabled the reloader holds the memory-mapped file backing.
        // We do NOT use mmap (`ReloadOptions::default()` sets `datamap: false`), so all
        // point data is copied into the HNSW graph during loading and the reloader can
        // be safely dropped afterwards — but the type system cannot see this.
        //
        // Storing `HnswIo` alongside `Hnsw` in the same struct would require a
        // self-referential type, which is unsound in safe Rust. The only clean alternative
        // is to give `HnswIo` a `'static` lifetime by leaking it. The allocation is
        // ~few KB (two file paths + options), happens at most once per process startup,
        // and is intentional — it is not a bug.
        let reloader = Box::leak(Box::new(HnswIo::new(dir, basename)));
        let hnsw = reloader
            .load_hnsw::<f32, DistCosine>()
            .map_err(|e| VectorError::ReloadFailed(e.to_string()))?;

        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGESmallENV15))
            .map_err(|e| VectorError::ModelInit(e.to_string()))?;

        Ok(Self { hnsw, embedder: Arc::new(Mutex::new(model)) })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Embeds a single text string, returning a 384-dim f32 vector.
    ///
    /// BGE-small outputs L2-normalized vectors — no manual normalization needed.
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, VectorError> {
        let mut guard = self.embedder.lock().map_err(|_| VectorError::MutexPoisoned)?;
        let mut results = guard
            .embed(vec![text], None)
            .map_err(|e| VectorError::EmbeddingFailed(e.to_string()))?;
        results.pop().ok_or_else(|| VectorError::EmbeddingFailed("empty output".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// All tests require downloading the fastembed model (~130 MB) on first run.
// They are marked #[ignore] to avoid network access in standard `cargo test`.
// Run with: cargo test -p nexus-core scoring::vector -- --include-ignored

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires fastembed model download (~130 MB)"]
    fn test_new_succeeds() {
        assert!(VectorIndex::new().is_ok());
    }

    #[test]
    #[ignore = "requires fastembed model download (~130 MB)"]
    fn test_insert_and_search_finds_related_doc() {
        let vi = VectorIndex::new().expect("build VectorIndex");
        vi.insert(0, "async rust runtime tokio executor").expect("insert 0");
        vi.insert(1, "python scripting dynamic language").expect("insert 1");
        vi.insert(2, "rust performance systems programming").expect("insert 2");

        let results = vi.search("async rust concurrency", 3);
        assert!(!results.is_empty(), "expected non-empty results");
        let ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        // Docs 0 and 2 are about Rust; doc 1 is Python.
        assert!(ids.contains(&0) || ids.contains(&2), "expected a Rust doc in top results");
    }

    #[test]
    #[ignore = "requires fastembed model download (~130 MB)"]
    fn test_search_before_insert_returns_empty() {
        let vi = VectorIndex::new().expect("build VectorIndex");
        let results = vi.search("rust", 5);
        assert!(results.is_empty(), "empty HNSW should return no results");
    }

    #[test]
    #[ignore = "requires fastembed model download (~130 MB)"]
    fn test_batch_insert_and_search() {
        let vi = VectorIndex::new().expect("build VectorIndex");
        let docs = [
            (0u32, "database sql query optimizer"),
            (1u32, "machine learning neural network"),
            (2u32, "rust memory safety borrow checker"),
        ];
        let refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, *t)).collect();
        vi.batch_insert(&refs).expect("batch insert");

        let results = vi.search("safe memory management rust", 3);
        assert!(!results.is_empty());
        let ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&2), "rust doc should rank high for memory safety query");
    }

    #[test]
    #[ignore = "requires fastembed model download (~130 MB)"]
    fn test_search_returns_sorted_scores() {
        let vi = VectorIndex::new().expect("build VectorIndex");
        vi.insert(0, "rust async tokio runtime").expect("insert");
        vi.insert(1, "python web django flask").expect("insert");
        vi.insert(2, "rust systems performance").expect("insert");

        let results = vi.search("rust programming", 3);
        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "scores not sorted: {} < {}",
                window[0].1,
                window[1].1
            );
        }
    }
}
