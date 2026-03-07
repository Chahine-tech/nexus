use std::sync::atomic::{AtomicU64, Ordering};
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

/// One indexed field with its boost weight.
///
/// BM25 is computed independently per field; scores are summed weighted by `boost`.
/// Keeping fields separate preserves per-field `avgdl`, which is critical:
/// a short `name` field (avgdl ≈ 2) has very different length normalization
/// than a `body` field (avgdl ≈ 15).
pub struct Field {
    pub index: Arc<InvertedIndex>,
    /// Multiplicative score weight. Typical values: name=3.0, body=1.0.
    pub boost: f32,
}

/// Multi-field BM25 scorer.
///
/// Score formula:
///   score(q, d) = Σ_field [ boost_field * Σ_term [ IDF(t,field) * TF_norm(t,d,field) ] ]
///
/// IDF is computed per-field using the field's own doc count (= total indexed docs,
/// since every document is indexed in every field). Global N from gossip overrides
/// the local count for accurate distributed IDF.
pub struct Bm25Scorer {
    /// Ordered list of fields. Field 0 is typically `name`, field 1 `body`.
    /// Kept as `pub(crate)` so `HybridScorer` can clone the primary index.
    pub(crate) fields: Vec<Field>,
    params: Bm25Params,
    /// Global document count from gossip. When zero, falls back to local count.
    global_n: Arc<AtomicU64>,
}

impl Bm25Scorer {
    /// Constructs a scorer from an explicit field list and params.
    pub fn new(fields: Vec<Field>, params: Bm25Params) -> Self {
        Self { fields, params, global_n: Arc::new(AtomicU64::new(0)) }
    }

    /// Single-field scorer — backward-compatible constructor used in tests and
    /// places that don't need field separation (e.g. AST code indexing).
    pub fn with_defaults(index: Arc<InvertedIndex>) -> Self {
        Self::new(vec![Field { index, boost: 1.0 }], Bm25Params::default())
    }

    /// Two-field scorer: `name` (boosted) + `body`.
    pub fn with_fields(name: Arc<InvertedIndex>, body: Arc<InvertedIndex>) -> Self {
        Self::new(
            vec![
                Field { index: name, boost: 3.0 },
                Field { index: body, boost: 1.0 },
            ],
            Bm25Params::default(),
        )
    }

    /// Returns a clone of the `global_n` atomic so `Node` can share it.
    pub fn global_n_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.global_n)
    }

    /// Returns the N to use in IDF: global if set, local (first field) otherwise.
    fn effective_n(&self) -> f32 {
        let g = self.global_n.load(Ordering::Relaxed);
        if g > 0 {
            g as f32
        } else {
            self.fields.first().map(|f| f.index.doc_count() as f32).unwrap_or(1.0)
        }
    }

    /// BM25 score for `doc_id` against a multi-term query, summed across all fields.
    ///
    ///   IDF(t,field)      = ln((N - df + 0.5) / (df + 0.5) + 1)
    ///   TF_norm(t,d,field) = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl_field))
    ///   score(q, d)        = Σ_field [ boost * Σ_t [ IDF * TF_norm ] ]
    pub fn score(&self, doc_id: u32, terms: &[String]) -> f32 {
        let n = self.effective_n();
        let k1 = self.params.k1;
        let b = self.params.b;

        self.fields
            .iter()
            .map(|field| {
                let avgdl = field.index.avg_doc_len();
                let dl = field.index.total_tokens_in_doc(doc_id) as f32;

                let field_score: f32 = terms
                    .iter()
                    .filter_map(|term| {
                        field.index.with_posting(term, |pl| {
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
                    .sum();

                field.boost * field_score
            })
            .sum()
    }

    /// Returns the top `limit` documents for a query, sorted by descending score.
    ///
    /// Candidate set = union of posting lists across ALL fields and terms.
    /// Scoring is parallelized via rayon.
    pub fn search(&self, terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        let candidates: RoaringBitmap = self
            .fields
            .iter()
            .flat_map(|field| {
                terms.iter().filter_map(|term| {
                    field.index.with_posting(term, |pl| pl.doc_ids().clone())
                })
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
