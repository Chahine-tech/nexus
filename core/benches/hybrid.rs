// Hybrid BM25 + fastembed vector benchmark.
//
// Requires the BAAI/bge-small-en-v1.5 model (~130 MB, downloaded once by fastembed).
// Skipped automatically unless NEXUS_BENCH_VECTOR=1 is set.
//
// Usage:
//   NEXUS_BENCH_VECTOR=1 cargo bench --bench hybrid

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use nexus_core::indexer::inverted::InvertedIndex;
use nexus_core::scoring::bm25::Bm25Scorer;
use nexus_core::scoring::hybrid::HybridScorer;
use nexus_core::scoring::vector::VectorIndex;

fn build_corpus(n: usize) -> (Arc<InvertedIndex>, Arc<InvertedIndex>, Vec<(u32, String)>) {
    let texts = [
        "async runtime tokio futures executor",
        "serialization deserialization serde json msgpack",
        "http web server axum actix router",
        "database sql query orm diesel",
        "machine learning neural network tensor",
        "cryptography hash signature ed25519 blake3",
        "parsing tokenizer lexer grammar ast",
        "concurrent parallel rayon thread pool",
        "memory safety borrow checker lifetime rust",
        "compression encoding deflate gzip lz4",
    ];
    let name_idx = Arc::new(InvertedIndex::new());
    let body_idx = Arc::new(InvertedIndex::new());
    let mut doc_texts = Vec::with_capacity(n);

    for i in 0..n as u32 {
        let text = texts[i as usize % texts.len()];
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        name_idx.index_document(i, &tokens[..1]);
        body_idx.index_document(i, &tokens);
        doc_texts.push((i, text.to_string()));
    }
    (name_idx, body_idx, doc_texts)
}

fn bench_hybrid_search(c: &mut Criterion) {
    if std::env::var("NEXUS_BENCH_VECTOR").as_deref() != Ok("1") {
        eprintln!("[hybrid bench] skipped — set NEXUS_BENCH_VECTOR=1 to run");
        return;
    }

    let _ = rayon::current_num_threads();
    let (name_idx, body_idx, doc_texts) = build_corpus(1_000);

    let vi = VectorIndex::new().expect("VectorIndex::new");
    let pairs: Vec<(u32, &str)> = doc_texts.iter().map(|(id, t)| (*id, t.as_str())).collect();
    vi.batch_insert(&pairs).expect("batch_insert");
    let vi = Arc::new(vi);

    let bm25 = Bm25Scorer::with_fields(Arc::clone(&name_idx), Arc::clone(&body_idx));
    let scorer = HybridScorer::new(bm25, Arc::clone(&vi), 0.5);

    c.bench_function("hybrid_search_1000_docs", |b| {
        b.iter(|| {
            let terms = vec!["async".to_string(), "runtime".to_string()];
            std::hint::black_box(scorer.search(&terms, 10));
        });
    });

    c.bench_function("hybrid_search_adaptive_1000_docs", |b| {
        b.iter(|| {
            let terms = vec!["async".to_string(), "runtime".to_string()];
            std::hint::black_box(
                HybridScorer::with_fields(Arc::clone(&name_idx), Arc::clone(&body_idx), Arc::clone(&vi))
                    .search_adaptive("async runtime", &terms, 10),
            );
        });
    });
}

criterion_group!(benches, bench_hybrid_search);
criterion_main!(benches);
