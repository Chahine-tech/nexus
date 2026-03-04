mod crawler;
mod indexer;
mod ast;
mod scoring;
mod network;
mod pagerank;
mod crypto;
mod sketch;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tracing::info!("Nexus node starting");
    Ok(())
}
