use std::collections::HashMap;
use std::sync::Arc;

use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::DistCosine;

use crate::indexer::inverted::InvertedIndex;

/// Errors produced by `VectorIndex`.
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("vocabulary is empty — build the inverted index before creating a VectorIndex")]
    EmptyVocabulary,
    #[error("document {0} has no terms present in the vocabulary snapshot")]
    NoTermsInVocab(u32),
}

/// HNSW-backed approximate nearest-neighbour index using tf-idf bag-of-words embeddings.
///
/// The vocabulary is snapshotted from `InvertedIndex` at construction time.
/// New terms introduced after construction are ignored until `VectorIndex::new` is
/// called again with the updated index.
pub struct VectorIndex {
    /// HNSW graph over f32 vectors with cosine distance.
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// Maps term → dimension index in the embedding vector.
    vocab: HashMap<String, usize>,
    /// Number of dimensions (= vocabulary size at construction time).
    dim: usize,
    /// Shared reference to the inverted index for IDF weights during embedding.
    index: Arc<InvertedIndex>,
}

impl VectorIndex {
    /// Snapshots the vocabulary from `index` and returns an empty HNSW ready for insertions.
    ///
    /// Returns `VectorError::EmptyVocabulary` if no terms have been indexed yet.
    pub fn new(index: Arc<InvertedIndex>) -> Result<Self, VectorError> {
        let mut vocab: Vec<String> = index.all_terms();

        if vocab.is_empty() {
            return Err(VectorError::EmptyVocabulary);
        }

        // Deterministic dimension order.
        vocab.sort_unstable();
        let dim = vocab.len();
        let vocab_map: HashMap<String, usize> = vocab
            .into_iter()
            .enumerate()
            .map(|(i, t)| (t, i))
            .collect();

        // max_nb_connection=16, max_elements=100_000, max_layer=16, ef_construction=200
        let hnsw = Hnsw::new(16, 100_000, 16, 200, DistCosine);

        Ok(Self { hnsw, vocab: vocab_map, dim, index })
    }

    /// Embeds `doc_id` using tf-idf weights and inserts it into the HNSW index.
    ///
    /// Returns `VectorError::NoTermsInVocab` if none of the document's terms appear
    /// in the vocabulary snapshot.
    pub fn insert(&self, doc_id: u32) -> Result<(), VectorError> {
        let embedding = self.embed_doc(doc_id)?;
        self.hnsw.insert_slice((&embedding, doc_id as usize));
        Ok(())
    }

    /// Approximate nearest-neighbour search for `query_terms`.
    ///
    /// Returns up to `limit` `(doc_id, cosine_similarity)` pairs sorted by
    /// descending similarity. Similarity is `1.0 - cosine_distance` (DistCosine range).
    pub fn search(&self, query_terms: &[String], limit: usize) -> Vec<(u32, f32)> {
        if limit == 0 || query_terms.is_empty() {
            return Vec::new();
        }
        let embedding = self.embed_query(query_terms);
        if embedding.iter().all(|&v| v == 0.0) {
            return Vec::new();
        }

        // ef_search = max(limit * 4, 50) for quality.
        let ef_search = (limit * 4).max(50);
        let neighbours = self.hnsw.search(&embedding, limit, ef_search);

        neighbours
            .into_iter()
            .map(|n| (n.d_id as u32, 1.0 - n.distance))
            .collect()
    }

    /// Returns the vocabulary size captured at construction time.
    pub fn vocab_size(&self) -> usize {
        self.dim
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Builds a tf-idf embedding for `doc_id` from its posting list entries.
    fn embed_doc(&self, doc_id: u32) -> Result<Vec<f32>, VectorError> {
        let n = self.index.doc_count() as f32;
        let mut vec = vec![0.0f32; self.dim];
        let mut has_any = false;

        for (term, &dim_idx) in &self.vocab {
            let weight = self.index.with_posting(term, |pl| {
                let tf = pl.tf(doc_id) as f32;
                if tf == 0.0 {
                    return None;
                }
                let df = pl.len() as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                Some(tf * idf)
            });
            if let Some(Some(w)) = weight {
                vec[dim_idx] = w;
                has_any = true;
            }
        }

        if !has_any {
            return Err(VectorError::NoTermsInVocab(doc_id));
        }

        l2_normalize(&mut vec);
        Ok(vec)
    }

    /// Builds a tf-idf embedding for a query token list.
    ///
    /// Query tf is the count of occurrences in `terms`; IDF from the global index.
    fn embed_query(&self, terms: &[String]) -> Vec<f32> {
        let n = self.index.doc_count() as f32;
        let mut vec = vec![0.0f32; self.dim];

        let mut qtf: HashMap<&str, f32> = HashMap::new();
        for t in terms {
            *qtf.entry(t.as_str()).or_insert(0.0) += 1.0;
        }

        for (term, &qtf_val) in &qtf {
            if let Some(&dim_idx) = self.vocab.get(*term)
                && let Some(idf) = self.index.with_posting(term, |pl| {
                    let df = pl.len() as f32;
                    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
                })
            {
                vec[dim_idx] = qtf_val * idf;
            }
        }

        l2_normalize(&mut vec);
        vec
    }
}

/// Normalizes `v` to unit L2 norm in-place. No-op if the norm is zero.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index() -> Arc<InvertedIndex> {
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
        idx
    }

    #[test]
    fn test_new_returns_error_on_empty_index() {
        let idx = Arc::new(InvertedIndex::new());
        assert!(matches!(VectorIndex::new(idx), Err(VectorError::EmptyVocabulary)));
    }

    #[test]
    fn test_embed_query_produces_unit_vector() {
        let idx = build_index();
        let vi = VectorIndex::new(Arc::clone(&idx)).expect("build VectorIndex");
        let terms = vec!["rust".to_string(), "fast".to_string()];
        let emb = vi.embed_query(&terms);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit vector, got norm={norm}");
    }

    #[test]
    fn test_insert_and_search_finds_rust_docs() {
        let idx = build_index();
        let vi = VectorIndex::new(Arc::clone(&idx)).expect("build VectorIndex");

        vi.insert(0).expect("insert doc 0");
        vi.insert(1).expect("insert doc 1");
        vi.insert(2).expect("insert doc 2");

        let results = vi.search(&["rust".to_string(), "fast".to_string()], 3);
        assert!(!results.is_empty(), "expected non-empty results");
        let ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        // Docs 0 and 2 contain "rust"; doc 1 does not.
        assert!(ids.contains(&0) || ids.contains(&2), "expected rust doc in top results");
    }

    #[test]
    fn test_search_before_any_insert_returns_empty() {
        let idx = build_index();
        let vi = VectorIndex::new(Arc::clone(&idx)).expect("build VectorIndex");
        let results = vi.search(&["rust".to_string()], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_unknown_term_returns_empty() {
        let idx = build_index();
        let vi = VectorIndex::new(Arc::clone(&idx)).expect("build VectorIndex");
        vi.insert(0).expect("insert");
        // "haskell" is not in the vocab snapshot.
        let results = vi.search(&["haskell".to_string()], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_vocab_size_matches_index() {
        let idx = build_index();
        let vi = VectorIndex::new(Arc::clone(&idx)).expect("build VectorIndex");
        assert_eq!(vi.vocab_size(), idx.vocabulary_size());
    }
}
