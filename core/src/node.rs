use std::sync::Arc;

use crate::indexer::inverted::InvertedIndex;
use crate::indexer::tokenizer::Tokenizer;
use crate::network::kademlia::RoutingTable;
use crate::scoring::bm25::Bm25Scorer;

/// Core search node — owns the index and exposes index/search operations.
pub struct Node {
    pub index: Arc<InvertedIndex>,
    tokenizer: Tokenizer,
    scorer: Bm25Scorer,
}

impl Node {
    pub fn new() -> Self {
        let index = Arc::new(InvertedIndex::new());
        let tokenizer = Tokenizer::new();
        let scorer = Bm25Scorer::with_defaults(Arc::clone(&index));
        Self { index, tokenizer, scorer }
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
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}
