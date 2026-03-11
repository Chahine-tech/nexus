use std::collections::HashMap;
use std::sync::Arc;

use crate::indexer::inverted::InvertedIndex;
use crate::scoring::bm25::Bm25Scorer;
use crate::scoring::query_features::QueryFeatures;
use crate::scoring::vector::VectorIndex;

/// Combines BM25 keyword scoring with approximate nearest-neighbour vector scoring.
///
/// `score = (1 - gamma) * [alpha * normalize(bm25) + (1 - alpha) * cosine] + gamma * normalize(pagerank)`
///
/// BM25 scores are min-max normalized across the merged candidate set before
/// combining, so that both signals are in the same [0, 1] range.
/// PageRank is log-normalized and blended as a small fixed-weight popularity signal.
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
    #[allow(dead_code)]
    pub fn search(&self, terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        self.search_with_pagerank(terms, limit, None)
    }

    /// Like `search()` but also blends PageRank as a popularity signal.
    ///
    /// `ef_search` controls the HNSW beam width. `None` uses the default `(limit * 4).max(50)`.
    pub fn search_with_pagerank(
        &self,
        terms: &[String],
        limit: usize,
        pagerank: Option<&HashMap<u32, f32>>,
    ) -> Vec<(u32, f32)> {
        self.search_with_pagerank_ef(terms, limit, pagerank, None)
    }

    /// Like `search_with_pagerank()` but with an explicit `ef_search` for the HNSW ANN step.
    pub fn search_with_pagerank_ef(
        &self,
        terms: &[String],
        limit: usize,
        pagerank: Option<&HashMap<u32, f32>>,
        ef_search: Option<usize>,
    ) -> Vec<(u32, f32)> {
        let fetch = (limit * 4).max(20);
        let bm25_hits = self.bm25.search(terms, fetch);
        let vec_hits = self.vector.search(&terms.join(" "), fetch, ef_search);
        hybrid_combine(bm25_hits, vec_hits, self.alpha, pagerank, limit)
    }

    /// Like `search()` but infers alpha from query features (QPP-based fusion).
    ///
    /// Extracts 6 features from the query (length, IDF stats, code token presence,
    /// stop word ratio, token entropy) and applies logistic regression to predict
    /// the optimal BM25/vector blend for this specific query type.
    ///
    /// `raw_query` is the original unprocessed string. `terms` is post-tokenization.
    #[allow(dead_code)]
    pub fn search_adaptive(&self, raw_query: &str, terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        self.search_adaptive_with_pagerank(raw_query, terms, limit, None)
    }

    /// Like `search_adaptive()` but also blends PageRank as a popularity signal.
    pub fn search_adaptive_with_pagerank(
        &self,
        raw_query: &str,
        terms: &[String],
        limit: usize,
        pagerank: Option<&HashMap<u32, f32>>,
    ) -> Vec<(u32, f32)> {
        self.search_adaptive_with_pagerank_ef(raw_query, terms, limit, pagerank, None)
    }

    /// Like `search_adaptive_with_pagerank()` but with an explicit `ef_search`.
    pub fn search_adaptive_with_pagerank_ef(
        &self,
        raw_query: &str,
        terms: &[String],
        limit: usize,
        pagerank: Option<&HashMap<u32, f32>>,
        ef_search: Option<usize>,
    ) -> Vec<(u32, f32)> {
        // Use the last field (body) for IDF features — it has the richest term distribution.
        let feature_index = self.bm25.fields.last().map(|f| &*f.index)
            .unwrap_or_else(|| &*self.bm25.fields[0].index);
        let features = QueryFeatures::extract(raw_query, terms, feature_index);
        let alpha = features.predict_alpha();
        tracing::debug!(alpha, query = raw_query, "adaptive alpha predicted");
        let fetch = (limit * 4).max(20);
        let bm25_hits = self.bm25.search(terms, fetch);
        let vec_hits = self.vector.search(raw_query, fetch, ef_search);
        hybrid_combine(bm25_hits, vec_hits, alpha, pagerank, limit)
    }
}

/// Merges BM25 hits and vector ANN hits into a single ranked list.
///
/// BM25 scores are min-max normalized across the merged candidate set before
/// the linear combination. When PageRank scores are provided they are blended
/// as a small popularity signal:
///
/// `score = (1 - gamma) * [alpha * bm25_norm + (1 - alpha) * cosine] + gamma * pr_norm`
///
/// where `gamma = 0.1` and PageRank is log-normalized across the candidate set.
/// If no PageRank is provided (or no candidate has a non-zero score), `gamma = 0`.
///
/// Exported so `Node::search_hybrid` can reuse it without duplicating logic.
pub fn hybrid_combine(
    bm25_hits: Vec<(u32, f32)>,
    vec_hits: Vec<(u32, f32)>,
    alpha: f32,
    pagerank: Option<&HashMap<u32, f32>>,
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

    // Build log-normalized PageRank scores for the candidate set.
    // log1p dampens the heavy-tailed PageRank distribution before min-max normalization.
    const GAMMA: f32 = 0.1;
    let pr_norm_map: Option<HashMap<u32, f32>> = pagerank.and_then(|pr| {
        let log_values: Vec<(u32, f32)> = scores
            .keys()
            .map(|id| (*id, pr.get(id).copied().unwrap_or(0.0).ln_1p()))
            .collect();
        let pr_min = log_values.iter().map(|(_, v)| *v).fold(f32::INFINITY, f32::min);
        let pr_max = log_values.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let pr_range = pr_max - pr_min;
        if pr_range <= 0.0 {
            return None; // all zeros — skip blending
        }
        Some(
            log_values
                .into_iter()
                .map(|(id, v)| (id, (v - pr_min) / pr_range))
                .collect(),
        )
    });

    let mut ranked: Vec<(u32, f32)> = scores
        .into_iter()
        .map(|(doc_id, (bm25, cosine))| {
            let bm25_norm =
                if bm25_range > 0.0 { (bm25 - bm25_min) / bm25_range } else { 0.0 };
            let relevance = alpha * bm25_norm + (1.0 - alpha) * cosine;
            let score = if let Some(ref pr_map) = pr_norm_map {
                let pr = pr_map.get(&doc_id).copied().unwrap_or(0.0);
                (1.0 - GAMMA) * relevance + GAMMA * pr
            } else {
                relevance
            };
            (doc_id, score)
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

    // Tests that exercise VectorIndex require the fastembed model (~130 MB download).
    // They are marked #[ignore]. Run with:
    //   cargo test -p nexus-core scoring::hybrid -- --include-ignored

    // -----------------------------------------------------------------------
    // hybrid_combine — pure function, no fastembed dependency, always runs
    // -----------------------------------------------------------------------

    #[test]
    fn test_hybrid_combine_empty_inputs() {
        let result = hybrid_combine(vec![], vec![], 0.5, None, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_hybrid_combine_bm25_only() {
        let bm25_hits = vec![(0u32, 2.0), (1u32, 1.0)];
        let result = hybrid_combine(bm25_hits, vec![], 1.0, None, 10);
        // alpha=1.0: pure BM25. BM25 min-max normalized — doc 0 scores highest.
        assert_eq!(result[0].0, 0);
        assert!(result[0].1 > result[1].1);
    }

    #[test]
    fn test_hybrid_combine_vector_only() {
        let vec_hits = vec![(0u32, 0.9), (1u32, 0.5)];
        let result = hybrid_combine(vec![], vec_hits, 0.0, None, 10);
        // alpha=0.0: pure vector. Doc 0 has highest cosine sim.
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn test_hybrid_combine_limit_respected() {
        let bm25_hits = vec![(0u32, 3.0), (1u32, 2.0), (2u32, 1.0)];
        let result = hybrid_combine(bm25_hits, vec![], 1.0, None, 2);
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_hybrid_combine_sorted_descending() {
        let bm25_hits = vec![(0u32, 1.0), (1u32, 3.0), (2u32, 2.0)];
        let result = hybrid_combine(bm25_hits, vec![], 1.0, None, 10);
        for window in result.windows(2) {
            assert!(window[0].1 >= window[1].1);
        }
    }

    #[test]
    fn test_hybrid_combine_pagerank_boosts_popular_doc() {
        // Doc 0 and Doc 1 have equal BM25 — PageRank should break the tie in favour of doc 1.
        let bm25_hits = vec![(0u32, 1.0), (1u32, 1.0)];
        let mut pr: HashMap<u32, f32> = HashMap::new();
        pr.insert(0, 0.1);
        pr.insert(1, 0.9);
        let result = hybrid_combine(bm25_hits, vec![], 1.0, Some(&pr), 10);
        assert_eq!(result[0].0, 1, "doc 1 should rank first due to higher PageRank");
    }

    // -----------------------------------------------------------------------
    // QPP alpha prediction — no fastembed dependency, always runs
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_alpha_differs_by_query_type() {
        let idx = Arc::new(InvertedIndex::new());
        let docs: &[&[&str]] = &[
            &["tokio", "spawn", "async", "rust"],
            &["how", "handle", "errors", "gracefully"],
            &["hashmap", "btreemap", "collections"],
        ];
        for (i, tokens) in docs.iter().enumerate() {
            let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            idx.index_document(i as u32, &owned);
        }

        let code_features = crate::scoring::query_features::QueryFeatures::extract(
            "tokio_spawn",
            &["tokio".to_string(), "spawn".to_string()],
            &idx,
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
    }

    // -----------------------------------------------------------------------
    // Full HybridScorer tests — require fastembed model download
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires fastembed model download (~130 MB)"]
    fn test_hybrid_returns_nonempty_results() {
        let idx = Arc::new(InvertedIndex::new());
        let docs: &[&[&str]] = &[
            &["rust", "fast", "safe", "rust"],
            &["python", "dynamic", "easy"],
            &["rust", "performance", "systems"],
        ];
        for (i, tokens) in docs.iter().enumerate() {
            let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            idx.index_document(i as u32, &owned);
        }
        let vi = VectorIndex::new().expect("build vi");
        vi.insert(0, "rust fast safe").expect("insert 0");
        vi.insert(1, "python dynamic easy").expect("insert 1");
        vi.insert(2, "rust performance systems").expect("insert 2");
        let bm25 = Bm25Scorer::with_fields(Arc::new(InvertedIndex::new()), Arc::clone(&idx));
        let scorer = HybridScorer::new(bm25, Arc::new(vi), 0.5);

        let results = scorer.search(&["rust".to_string()], 5);
        assert!(!results.is_empty());
    }

    #[test]
    #[ignore = "requires fastembed model download (~130 MB)"]
    fn test_alpha_one_matches_bm25_order() {
        // alpha=1.0 + empty HNSW → no vector hits → hybrid == pure BM25
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
        let vi = Arc::new(VectorIndex::new().expect("build vi"));
        // No inserts — empty HNSW produces no vector hits.
        let name_idx = Arc::new(InvertedIndex::new());
        let bm25_check = Bm25Scorer::with_fields(Arc::clone(&name_idx), Arc::clone(&idx));
        let bm25_hybrid = Bm25Scorer::with_fields(Arc::clone(&name_idx), Arc::clone(&idx));
        let hybrid = HybridScorer::new(bm25_hybrid, vi, 1.0);

        let hybrid_ids: Vec<u32> =
            hybrid.search(&["rust".to_string()], 10).iter().map(|(id, _)| *id).collect();
        let bm25_ids: Vec<u32> =
            bm25_check.search(&["rust".to_string()], 10).iter().map(|(id, _)| *id).collect();

        assert_eq!(hybrid_ids.first(), bm25_ids.first());
    }
}
