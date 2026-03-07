use std::collections::HashMap;
use std::sync::Arc;

use crate::indexer::inverted::InvertedIndex;
use crate::scoring::bm25::Bm25Scorer;
use crate::scoring::query_features::QueryFeatures;
use crate::scoring::vector::VectorIndex;

/// Combines BM25 keyword scoring with approximate nearest-neighbour vector scoring.
///
/// `score = alpha * normalize(bm25) + (1 - alpha) * cosine_similarity`
///
/// BM25 scores are min-max normalized across the merged candidate set before
/// combining, so that both signals are in the same [0, 1] range.
pub struct HybridScorer {
    bm25: Bm25Scorer,
    vector: Arc<VectorIndex>,
    /// Weight for the BM25 signal. Vector weight = `1.0 - alpha`. Range [0.0, 1.0].
    alpha: f32,
}

impl HybridScorer {
    /// Creates a new `HybridScorer`.
    ///
    /// `alpha = 1.0` → pure BM25. `alpha = 0.0` → pure vector. Default: `0.5`.
    pub fn new(bm25: Bm25Scorer, vector: Arc<VectorIndex>, alpha: f32) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self { bm25, vector, alpha }
    }

    /// Two-field constructor: `name` + `body`, with default `alpha = 0.5`.
    pub fn with_fields(
        name: Arc<InvertedIndex>,
        body: Arc<InvertedIndex>,
        vector: Arc<VectorIndex>,
    ) -> Self {
        let bm25 = Bm25Scorer::with_fields(name, body);
        Self::new(bm25, vector, 0.5)
    }

    /// Returns up to `limit` documents ranked by the hybrid score.
    ///
    /// Candidate set = union of BM25 hits and ANN vector hits (each over-fetched by 4×).
    /// BM25 scores are min-max normalized before linear combination with cosine similarity.
    pub fn search(&self, terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        let fetch = (limit * 4).max(20);
        let bm25_hits = self.bm25.search(terms, fetch);
        let vec_hits = self.vector.search(terms, fetch);
        hybrid_combine(bm25_hits, vec_hits, self.alpha, limit)
    }

    /// Like `search()` but infers alpha from query features (QPP-based fusion).
    ///
    /// Extracts 6 features from the query (length, IDF stats, code token presence,
    /// stop word ratio, token entropy) and applies logistic regression to predict
    /// the optimal BM25/vector blend for this specific query type.
    ///
    /// `raw_query` is the original unprocessed string. `terms` is post-tokenization.
    pub fn search_adaptive(&self, raw_query: &str, terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        // Use the last field (body) for IDF features — it has the richest term distribution.
        let feature_index = self.bm25.fields.last().map(|f| &*f.index)
            .unwrap_or_else(|| &*self.bm25.fields[0].index);
        let features = QueryFeatures::extract(raw_query, terms, feature_index);
        let alpha = features.predict_alpha();
        tracing::debug!(alpha, query = raw_query, "adaptive alpha predicted");
        let fetch = (limit * 4).max(20);
        let bm25_hits = self.bm25.search(terms, fetch);
        let vec_hits = self.vector.search(terms, fetch);
        hybrid_combine(bm25_hits, vec_hits, alpha, limit)
    }
}

/// Merges BM25 hits and vector ANN hits into a single ranked list.
///
/// BM25 scores are min-max normalized across the merged candidate set before
/// the linear combination `alpha * bm25_norm + (1 - alpha) * cosine`.
/// Exported so `Node::search_hybrid` can reuse it without duplicating logic.
pub fn hybrid_combine(
    bm25_hits: Vec<(u32, f32)>,
    vec_hits: Vec<(u32, f32)>,
    alpha: f32,
    limit: usize,
) -> Vec<(u32, f32)> {
    if limit == 0 || (bm25_hits.is_empty() && vec_hits.is_empty()) {
        return Vec::new();
    }

    let mut scores: HashMap<u32, (f32, f32)> = HashMap::new();
    for (doc_id, score) in &bm25_hits {
        scores.entry(*doc_id).or_insert((0.0, 0.0)).0 = *score;
    }
    for (doc_id, sim) in &vec_hits {
        scores.entry(*doc_id).or_insert((0.0, 0.0)).1 = *sim;
    }

    let bm25_values: Vec<f32> = scores.values().map(|(b, _)| *b).collect();
    let bm25_min = bm25_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let bm25_max = bm25_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let bm25_range = bm25_max - bm25_min;

    let mut ranked: Vec<(u32, f32)> = scores
        .into_iter()
        .map(|(doc_id, (bm25, cosine))| {
            let bm25_norm =
                if bm25_range > 0.0 { (bm25 - bm25_min) / bm25_range } else { 0.0 };
            (doc_id, alpha * bm25_norm + (1.0 - alpha) * cosine)
        })
        .collect();

    ranked.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(limit);
    ranked
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::vector::VectorError;

    fn build_scorer(alpha: f32) -> Result<HybridScorer, VectorError> {
        let idx = Arc::new(InvertedIndex::new());
        let docs: &[&[&str]] = &[
            &["rust", "fast", "safe", "rust"],
            &["python", "dynamic", "easy"],
            &["rust", "performance", "systems"],
            &["java", "safe", "verbose"],
        ];
        for (i, tokens) in docs.iter().enumerate() {
            let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            idx.index_document(i as u32, &owned);
        }

        let vi = VectorIndex::new(Arc::clone(&idx))?;
        // Insert all docs into HNSW.
        for i in 0..4u32 {
            let _ = vi.insert(i); // doc 1 and 3 may have terms in vocab — ignore NoTermsInVocab
        }

        let bm25 = Bm25Scorer::with_fields(Arc::new(InvertedIndex::new()), Arc::clone(&idx));
        Ok(HybridScorer::new(bm25, Arc::new(vi), alpha))
    }

    #[test]
    fn test_hybrid_returns_nonempty_results() {
        let scorer = build_scorer(0.5).expect("build scorer");
        let results = scorer.search(&["rust".to_string()], 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_limit_respected() {
        let scorer = build_scorer(0.5).expect("build scorer");
        let results = scorer.search(&["rust".to_string(), "fast".to_string()], 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_empty_terms_returns_empty() {
        let scorer = build_scorer(0.5).expect("build scorer");
        assert!(scorer.search(&[], 10).is_empty());
    }

    #[test]
    fn test_scores_sorted_descending() {
        let scorer = build_scorer(0.5).expect("build scorer");
        let results = scorer.search(&["rust".to_string()], 10);
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
    fn test_alpha_zero_uses_only_vector() {
        // With alpha=0.0, BM25 has no influence. Results still non-empty if vector hits exist.
        let scorer = build_scorer(0.0).expect("build scorer");
        let results = scorer.search(&["rust".to_string()], 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_adaptive_returns_results() {
        let scorer = build_scorer(0.5).expect("build scorer");
        let terms = vec!["rust".to_string()];
        let results = scorer.search_adaptive("rust", &terms, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_adaptive_alpha_differs_by_query_type() {
        let idx = Arc::new(InvertedIndex::new());
        let docs: &[&[&str]] = &[
            &["tokio", "spawn", "async", "rust"],
            &["how", "to", "handle", "errors", "gracefully"],
            &["hashmap", "btreemap", "collections"],
        ];
        for (i, tokens) in docs.iter().enumerate() {
            let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            idx.index_document(i as u32, &owned);
        }
        let vi = VectorIndex::new(Arc::clone(&idx)).expect("build vi");
        for i in 0..3u32 { let _ = vi.insert(i); }
        let bm25 = Bm25Scorer::with_fields(Arc::new(InvertedIndex::new()), Arc::clone(&idx));
        let scorer = HybridScorer::new(bm25, Arc::new(vi), 0.5);

        let code_features = crate::scoring::query_features::QueryFeatures::extract(
            "tokio_spawn", &["tokio".to_string(), "spawn".to_string()], &idx,
        );
        let natural_features = crate::scoring::query_features::QueryFeatures::extract(
            "how to handle errors gracefully",
            &["how".to_string(), "handle".to_string(), "errors".to_string(), "gracefully".to_string()],
            &idx,
        );

        let alpha_code = code_features.predict_alpha();
        let alpha_natural = natural_features.predict_alpha();
        assert!(
            alpha_code > alpha_natural,
            "code query alpha ({alpha_code:.3}) should exceed natural language alpha ({alpha_natural:.3})"
        );

        // Both search_adaptive calls should complete without error.
        let _ = scorer.search_adaptive("tokio_spawn", &["tokio".to_string(), "spawn".to_string()], 3);
        let _ = scorer.search_adaptive("how to handle errors", &["handle".to_string(), "errors".to_string()], 3);
    }

    #[test]
    fn test_alpha_one_matches_bm25_order() {
        // With alpha=1.0, order should be dominated by BM25 (modulo equal-score ties).
        let idx = Arc::new(InvertedIndex::new());
        let docs: &[&[&str]] = &[
            &["rust", "rust", "rust", "fast"],
            &["python", "dynamic"],
            &["rust", "performance"],
        ];
        for (i, tokens) in docs.iter().enumerate() {
            let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            idx.index_document(i as u32, &owned);
        }
        let vi = VectorIndex::new(Arc::clone(&idx)).expect("build vi");
        for i in 0..3u32 {
            let _ = vi.insert(i);
        }
        let bm25_for_check = Bm25Scorer::with_fields(Arc::new(InvertedIndex::new()), Arc::clone(&idx));
        let bm25_for_hybrid = Bm25Scorer::with_fields(Arc::new(InvertedIndex::new()), Arc::clone(&idx));
        let hybrid = HybridScorer::new(bm25_for_hybrid, Arc::new(vi), 1.0);

        let hybrid_results = hybrid.search(&["rust".to_string()], 10);
        let bm25_results = bm25_for_check.search(&["rust".to_string()], 10);

        let hybrid_ids: Vec<u32> = hybrid_results.iter().map(|(id, _)| *id).collect();
        let bm25_ids: Vec<u32> = bm25_results.iter().map(|(id, _)| *id).collect();

        // The top result should agree (doc 0 has the most "rust" occurrences).
        assert_eq!(hybrid_ids.first(), bm25_ids.first());
    }
}
