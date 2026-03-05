mod crawler;
mod indexer;
mod ast;
mod scoring;
mod network;
mod pagerank;
mod crypto;
mod sketch;

use std::sync::Arc;

use crate::indexer::inverted::InvertedIndex;
use crate::indexer::tokenizer::Tokenizer;
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tracing::info!("Nexus node starting");

    let node = Node::new();

    // Smoke test — will be replaced by the crawler feeding real documents.
    node.index_document(0, "Rust is a systems programming language focused on safety and performance");
    node.index_document(1, "Python is a dynamic language great for scripting and data science");
    node.index_document(2, "Rust enables fearless concurrency without data races");

    let results = node.search("rust concurrency", 10);
    tracing::info!(?results, "search results");

    Ok(())
}
