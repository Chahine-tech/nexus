use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use nexus_core::indexer::inverted::InvertedIndex;
use nexus_core::scoring::bm25::{Bm25Params, Bm25Scorer};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generates `n_docs` documents of `tokens_per_doc` tokens each.
/// Vocabulary is 200 synthetic terms so posting lists grow meaningfully.
fn synthetic_corpus(n_docs: usize, tokens_per_doc: usize) -> Vec<Vec<String>> {
    let vocab: Vec<String> = (0..200).map(|i| format!("token_{i}")).collect();
    (0..n_docs)
        .map(|doc| {
            (0..tokens_per_doc)
                .map(|tok| vocab[(doc + tok) % vocab.len()].clone())
                .collect()
        })
        .collect()
}

fn build_index(corpus: &[Vec<String>]) -> Arc<InvertedIndex> {
    let idx = Arc::new(InvertedIndex::new());
    for (id, tokens) in corpus.iter().enumerate() {
        idx.index_document(id as u32, tokens);
    }
    idx
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_index_document(c: &mut Criterion) {
    let corpus = synthetic_corpus(1000, 100);

    c.bench_function("index_document_1000x100", |b| {
        b.iter(|| {
            let idx = InvertedIndex::new();
            for (id, tokens) in corpus.iter().enumerate() {
                idx.index_document(id as u32, tokens);
            }
        });
    });
}

fn bench_lookup(c: &mut Criterion) {
    let idx = Arc::new(InvertedIndex::new());
    // "hot_term" appears in 500 out of 1000 docs (even doc IDs).
    for doc_id in 0u32..1000 {
        let mut tokens = vec![format!("tok_{}", doc_id % 50)];
        if doc_id % 2 == 0 {
            tokens.push("hot_term".to_string());
        }
        idx.index_document(doc_id, &tokens);
    }

    c.bench_function("lookup_500_docs", |b| {
        b.iter(|| {
            let _ = std::hint::black_box(idx.lookup("hot_term"));
        });
    });
}

fn bench_bm25_search(c: &mut Criterion) {
    // Pre-warm rayon thread pool before measurement.
    let _ = rayon::current_num_threads();

    let corpus = synthetic_corpus(1000, 100);
    let idx = build_index(&corpus);
    let scorer = Bm25Scorer::with_defaults(Arc::clone(&idx));
    let query = vec![
        "token_0".to_string(),
        "token_1".to_string(),
        "token_2".to_string(),
    ];

    c.bench_function("bm25_search_1000docs_3terms", |b| {
        b.iter(|| {
            let results = std::hint::black_box(scorer.search(&query, 10));
            assert!(!results.is_empty());
        });
    });
}

criterion_group!(benches, bench_index_document, bench_lookup, bench_bm25_search);
criterion_main!(benches);
