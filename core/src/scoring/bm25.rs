use std::sync::Arc;

use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::indexer::inverted::InvertedIndex;

/// Tunable BM25 parameters.
///
/// Defaults: k1=1.2, b=0.75 (standard Okapi BM25 values).
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    /// Term-frequency saturation factor.
    pub k1: f32,
    /// Document-length normalization strength.
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// BM25 scorer backed by a shared `InvertedIndex`.
pub struct Bm25Scorer {
    index: Arc<InvertedIndex>,
    params: Bm25Params,
}

impl Bm25Scorer {
    pub fn new(index: Arc<InvertedIndex>, params: Bm25Params) -> Self {
        Self { index, params }
    }

    pub fn with_defaults(index: Arc<InvertedIndex>) -> Self {
        Self::new(index, Bm25Params::default())
    }

    /// BM25 score for `doc_id` against a multi-term query.
    ///
    ///   IDF(t)      = ln((N - df + 0.5) / (df + 0.5) + 1)
    ///   TF_norm(t,d) = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    ///   score(q, d) = Σ_t [ IDF(t) * TF_norm(t, d) ]
    pub fn score(&self, doc_id: u32, terms: &[String]) -> f32 {
        let n = self.index.doc_count() as f32;
        let avgdl = self.index.avg_doc_len();
        let dl = self.index.total_tokens_in_doc(doc_id) as f32;
        let k1 = self.params.k1;
        let b = self.params.b;

        terms
            .iter()
            .filter_map(|term| {
                // Access posting list via ref — no clone in hot path.
                self.index.with_posting(term, |pl| {
                    let df = pl.len() as f32;
                    let tf = pl.tf(doc_id) as f32;
                    if tf == 0.0 {
                        return 0.0;
                    }
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    let tf_norm =
                        (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl));
                    idf * tf_norm
                })
            })
            .sum()
    }

    /// Returns the top `limit` documents for a query, sorted by descending score.
    ///
    /// Candidate set = union of all posting list bitmaps for query terms.
    /// Scoring is parallelized via rayon.
    pub fn search(&self, terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        // Build candidate union via ref — bitmap clone is unavoidable here
        // since we cannot hold a DashMap shard ref across multiple iterations.
        let candidates: RoaringBitmap = terms
            .iter()
            .filter_map(|term| {
                self.index.with_posting(term, |pl| pl.doc_ids().clone())
            })
            .fold(RoaringBitmap::new(), |mut acc, bm| {
                acc |= bm;
                acc
            });

        if candidates.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(u32, f32)> = candidates
            .iter()
            .par_bridge()
            .map(|doc_id| (doc_id, self.score(doc_id, terms)))
            .collect();

        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);
        scored
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::inverted::InvertedIndex;

    fn build_index() -> Arc<InvertedIndex> {
        let idx = Arc::new(InvertedIndex::new());
        let docs: &[&[&str]] = &[
            &["rust", "fast", "safe", "rust"],
            &["python", "dynamic", "fast"],
            &["rust", "performance", "systems"],
            &["java", "safe", "verbose"],
            &["rust", "rust", "rust", "fast"],
        ];
        for (i, tokens) in docs.iter().enumerate() {
            let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            idx.index_document(i as u32, &owned);
        }
        idx
    }

    #[test]
    fn test_search_returns_relevant_docs() {
        let scorer = Bm25Scorer::with_defaults(build_index());
        let results = scorer.search(&["rust".to_string()], 10);
        let ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();

        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1)); // python doc has no "rust"
    }

    #[test]
    fn test_scores_sorted_descending() {
        let scorer = Bm25Scorer::with_defaults(build_index());
        let results = scorer.search(&["rust".to_string(), "fast".to_string()], 10);
        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "scores not sorted: {} < {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn test_empty_query_returns_empty() {
        let scorer = Bm25Scorer::with_defaults(build_index());
        assert!(scorer.search(&[], 10).is_empty());
    }

    #[test]
    fn test_limit_respected() {
        let scorer = Bm25Scorer::with_defaults(build_index());
        let results = scorer.search(&["rust".to_string()], 2);
        assert!(results.len() <= 2);
    }
}
