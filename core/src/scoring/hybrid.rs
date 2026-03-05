use std::collections::HashMap;
use std::sync::Arc;

use crate::indexer::inverted::InvertedIndex;
use crate::scoring::bm25::Bm25Scorer;
use crate::scoring::vector::VectorIndex;

/// Combines BM25 keyword scoring with approximate nearest-neighbour vector scoring.
///
/// `score = alpha * normalize(bm25) + (1 - alpha) * cosine_similarity`
///
/// BM25 scores are min-max normalized across the merged candidate set before
/// combining, so that both signals are in the same [0, 1] range.
pub struct HybridScorer {
    bm25: Bm25Scorer,
    vector: VectorIndex,
    /// Weight for the BM25 signal. Vector weight = `1.0 - alpha`. Range [0.0, 1.0].
    alpha: f32,
}

impl HybridScorer {
    /// Creates a new `HybridScorer`.
    ///
    /// `alpha = 1.0` → pure BM25. `alpha = 0.0` → pure vector. Default: `0.5`.
    pub fn new(bm25: Bm25Scorer, vector: VectorIndex, alpha: f32) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self { bm25, vector, alpha }
    }

    /// Convenience constructor with default `alpha = 0.5`.
    pub fn with_defaults(index: Arc<InvertedIndex>, vector: VectorIndex) -> Self {
        let bm25 = Bm25Scorer::with_defaults(index);
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

        let bm25 = Bm25Scorer::with_defaults(Arc::clone(&idx));
        Ok(HybridScorer::new(bm25, vi, alpha))
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
        let bm25_for_check = Bm25Scorer::with_defaults(Arc::clone(&idx));
        let bm25_for_hybrid = Bm25Scorer::with_defaults(Arc::clone(&idx));
        let hybrid = HybridScorer::new(bm25_for_hybrid, vi, 1.0);

        let hybrid_results = hybrid.search(&["rust".to_string()], 10);
        let bm25_results = bm25_for_check.search(&["rust".to_string()], 10);

        let hybrid_ids: Vec<u32> = hybrid_results.iter().map(|(id, _)| *id).collect();
        let bm25_ids: Vec<u32> = bm25_results.iter().map(|(id, _)| *id).collect();

        // The top result should agree (doc 0 has the most "rust" occurrences).
        assert_eq!(hybrid_ids.first(), bm25_ids.first());
    }
}
