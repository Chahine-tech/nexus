use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nexus_core::indexer::posting::PostingList;
use nexus_core::indexer::inverted::InvertedIndex;
use nexus_core::indexer::tokenizer::Tokenizer;
use nexus_core::scoring::bm25::Bm25Scorer;

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
    let scorer = Bm25Scorer::with_fields(Arc::new(InvertedIndex::new()), Arc::clone(&idx));
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

fn bench_bm25_search_10k(c: &mut Criterion) {
    let _ = rayon::current_num_threads();
    let corpus = synthetic_corpus(10_000, 50);
    let idx = build_index(&corpus);
    let scorer = Bm25Scorer::with_fields(Arc::new(InvertedIndex::new()), Arc::clone(&idx));
    let query = vec!["token_0".to_string(), "token_1".to_string()];

    c.bench_function("bm25_search_10k_docs", |b| {
        b.iter(|| {
            std::hint::black_box(scorer.search(&query, 10));
        });
    });
}

fn bench_bm25_search_100k(c: &mut Criterion) {
    let _ = rayon::current_num_threads();
    // Build corpus outside the measurement loop (slow to construct).
    let corpus = synthetic_corpus(100_000, 50);
    let idx = build_index(&corpus);
    let scorer = Bm25Scorer::with_fields(Arc::new(InvertedIndex::new()), Arc::clone(&idx));
    let query = vec!["token_0".to_string(), "token_1".to_string()];

    c.bench_function("bm25_search_100k_docs", |b| {
        b.iter(|| {
            std::hint::black_box(scorer.search(&query, 10));
        });
    });
}

fn bench_tokenizer_throughput(c: &mut Criterion) {
    let tok = Tokenizer::new();
    let text = "the quick brown fox jumps over the lazy dog ".repeat(50);

    c.bench_function("tokenizer_500_words", |b| {
        b.iter(|| {
            std::hint::black_box(tok.tokenize(&text));
        });
    });
}

// ---------------------------------------------------------------------------
// BP128 serde benchmarks
// ---------------------------------------------------------------------------

/// Builds a dense posting list with consecutive doc IDs and varying TF values.
fn synthetic_posting_list(n: usize) -> PostingList {
    let mut pl = PostingList::new();
    for i in 0..n as u32 {
        pl.insert(i, (i % 10) + 1);
    }
    pl
}

fn bench_bp128_serde(c: &mut Criterion) {
    let mut group = c.benchmark_group("bp128_serde");

    for &n in &[1_000usize, 10_000, 100_000] {
        let pl = synthetic_posting_list(n);
        let bytes = rmp_serde::to_vec(&pl).expect("serialize");

        // Report compressed size once (visible in bench output).
        eprintln!("n={n}  compressed_bytes={}", bytes.len());

        group.bench_with_input(BenchmarkId::new("encode", n), &pl, |b, pl| {
            b.iter(|| rmp_serde::to_vec(std::hint::black_box(pl)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("decode", n), &bytes, |b, bytes| {
            b.iter(|| {
                rmp_serde::from_slice::<PostingList>(std::hint::black_box(bytes)).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_index_document,
    bench_lookup,
    bench_bm25_search,
    bench_bm25_search_10k,
    bench_bm25_search_100k,
    bench_tokenizer_throughput,
    bench_bp128_serde,
);
criterion_main!(benches);
