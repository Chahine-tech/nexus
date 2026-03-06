use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::ast::features;
use crate::ast::parser::{AstError, AstParser};
use crate::indexer::inverted::InvertedIndex;
use crate::indexer::tokenizer::Tokenizer;
use crate::network::kademlia::RoutingTable;
use crate::pagerank::local::LocalPageRank;
use crate::scoring::bm25::Bm25Scorer;
use crate::scoring::hybrid::hybrid_combine;
use crate::scoring::vector::{VectorError, VectorIndex};

/// Core search node — owns the index, scorers, and local PageRank graph.
pub struct Node {
    pub index: Arc<InvertedIndex>,
    tokenizer: Tokenizer,
    scorer: Bm25Scorer,
    /// HNSW vector index. Wrapped in RwLock because Hnsw requires exclusive access
    /// during insertion but allows concurrent reads during search.
    vector: RwLock<Option<VectorIndex>>,
    /// Weight of BM25 vs vector in hybrid search. Range [0.0, 1.0].
    hybrid_alpha: f32,
    pagerank: RwLock<LocalPageRank>,
}

impl Node {
    pub fn new() -> Self {
        let index = Arc::new(InvertedIndex::new());
        Self::from_index(index)
    }

    /// Creates a node from an already-built inverted index (e.g. loaded from disk).
    pub fn from_index(index: Arc<InvertedIndex>) -> Self {
        let tokenizer = Tokenizer::new();
        let scorer = Bm25Scorer::with_defaults(Arc::clone(&index));
        Self {
            index,
            tokenizer,
            scorer,
            vector: RwLock::new(None),
            hybrid_alpha: 0.5,
            pagerank: RwLock::new(LocalPageRank::new()),
        }
    }

    /// Total number of indexed documents.
    pub fn doc_count(&self) -> u64 {
        self.index.doc_count()
    }

    /// Number of unique terms in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.index.vocabulary_size()
    }

    /// Returns `true` if PageRank has been computed at least once.
    pub fn pagerank_ready(&self) -> bool {
        !self.pagerank.read().expect("pagerank RwLock poisoned").ranked().is_empty()
    }

    /// Tokenizes `text` and indexes it under `doc_id`.
    pub fn index_document(&self, doc_id: u32, text: &str) {
        let tokens = self.tokenizer.tokenize(text);
        self.index.index_document(doc_id, &tokens);
        tracing::debug!(doc_id, tokens = tokens.len(), "indexed document");
    }

    /// Returns the top `limit` results for `query`, sorted by BM25 score.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(u32, f32)> {
        let terms = self.tokenizer.tokenize(query);
        self.scorer.search(&terms, limit)
    }

    /// Searches using pre-tokenized terms — avoids double-tokenization in QueryRouter.
    pub fn search_terms(&self, terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        self.scorer.search(terms, limit)
    }

    /// Returns true if this node is the XOR-closest peer to `term`'s blake3 key.
    ///
    /// Returns true when the routing table is empty (single-node network owns all shards).
    pub fn responsible_for(&self, term: &str, table: &RoutingTable) -> bool {
        match table.responsible_node(term) {
            Some(closest) => closest.id == table.local_id,
            None => true,
        }
    }

    // -----------------------------------------------------------------------
    // Vector index
    // -----------------------------------------------------------------------

    /// (Re-)builds the HNSW vector index from the current inverted index.
    ///
    /// Must be called after bulk indexing before `search_hybrid` returns vector results.
    pub fn rebuild_vector_index(&self) -> Result<(), VectorError> {
        let vi = VectorIndex::new(Arc::clone(&self.index))?;
        let doc_ids = self.index.all_doc_ids();
        for doc_id in doc_ids {
            // Ignore NoTermsInVocab — the doc may have been indexed after the vocab snapshot.
            let _ = vi.insert(doc_id);
        }
        let mut guard = self.vector.write().expect("vector RwLock poisoned");
        *guard = Some(vi);
        Ok(())
    }

    /// Hybrid BM25 + vector ANN search.
    ///
    /// Falls back to pure BM25 if the vector index has not been built yet.
    pub fn search_hybrid(&self, query: &str, limit: usize) -> Vec<(u32, f32)> {
        let terms = self.tokenizer.tokenize(query);
        let guard = self.vector.read().expect("vector RwLock poisoned");

        let Some(vi) = guard.as_ref() else {
            drop(guard);
            return self.scorer.search(&terms, limit);
        };

        let fetch = (limit * 4).max(20);
        let bm25 = Bm25Scorer::with_defaults(Arc::clone(&self.index));
        let bm25_hits = bm25.search(&terms, fetch);
        let vec_hits = vi.search(&terms, fetch);
        let alpha = self.hybrid_alpha;
        drop(guard);

        hybrid_combine(bm25_hits, vec_hits, alpha, limit)
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
    // Code-file indexing (AST pipeline)
    // -----------------------------------------------------------------------

    /// Indexes a source file using the AST pipeline.
    ///
    /// Detects language from `path` extension, parses with tree-sitter in a
    /// blocking thread (because `AstParser` is `!Send`), then indexes the
    /// extracted function names, type names, imports, and tokenized literals.
    ///
    /// Returns `AstError::UnsupportedLanguage` for non-Rust/TS/Python files.
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

        let mut tokens: Vec<String> = Vec::new();
        tokens.extend(code_features.function_names);
        tokens.extend(code_features.type_names);
        tokens.extend(code_features.imports);
        for lit in code_features.literals {
            tokens.extend(self.tokenizer.tokenize(&lit));
        }

        self.index.index_document(doc_id, &tokens);
        tracing::debug!(doc_id, tokens = tokens.len(), "indexed code file");
        Ok(())
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
