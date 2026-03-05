mod crawler;
mod indexer;
mod ast;
mod scoring;
mod network;
mod pagerank;
mod crypto;
mod sketch;
mod node;

use node::Node;

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
