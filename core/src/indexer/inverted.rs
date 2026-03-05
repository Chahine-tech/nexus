use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::indexer::posting::PostingList;

#[derive(Debug, thiserror::Error)]
pub enum InvertedIndexError {
    #[error("serialization failed: {0}")]
    Serialization(#[from] rmp_serde::encode::Error),

    #[error("deserialization failed: {0}")]
    Deserialization(#[from] rmp_serde::decode::Error),
}

/// A concurrent inverted index mapping terms to their posting lists.
///
/// `index_document` takes `&self` (not `&mut self`) — safe to call from
/// multiple rayon threads simultaneously.
pub struct InvertedIndex {
    /// term → PostingList
    postings: DashMap<String, PostingList>,
    /// doc_id → number of tokens in that document
    doc_lengths: DashMap<u32, u32>,
    /// Total unique documents indexed.
    doc_count: AtomicU64,
    /// Running sum of all document lengths (for avgdl).
    total_token_sum: AtomicU64,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            postings: DashMap::new(),
            doc_lengths: DashMap::new(),
            doc_count: AtomicU64::new(0),
            total_token_sum: AtomicU64::new(0),
        }
    }

    /// Indexes `doc_id` with its token list.
    ///
    /// Builds a local TF map first to avoid acquiring the same DashMap shard
    /// repeatedly for the same term within one document.
    pub fn index_document(&self, doc_id: u32, tokens: &[String]) {
        let mut tf_map: HashMap<&str, u32> = HashMap::with_capacity(tokens.len());
        for token in tokens {
            *tf_map.entry(token.as_str()).or_insert(0) += 1;
        }

        for (term, tf) in &tf_map {
            self.postings
                .entry(term.to_string())
                .or_default()
                .insert(doc_id, *tf);
        }

        let doc_len = tokens.len() as u32;
        self.doc_lengths.insert(doc_id, doc_len);
        self.doc_count.fetch_add(1, Ordering::Relaxed);
        self.total_token_sum.fetch_add(doc_len as u64, Ordering::Relaxed);
    }

    /// Calls `f` with a reference to the posting list for `term`, without cloning.
    ///
    /// Prefer this over `lookup` in hot paths (e.g. BM25 scoring).
    /// The DashMap shard lock is held only for the duration of `f`.
    pub fn with_posting<F, R>(&self, term: &str, f: F) -> Option<R>
    where
        F: FnOnce(&PostingList) -> R,
    {
        self.postings.get(term).map(|entry| f(&*entry))
    }

    /// Returns a clone of the posting list for `term`.
    ///
    /// Use `with_posting` in hot paths to avoid cloning.
    /// This method is kept for serialization and tests.
    pub fn lookup(&self, term: &str) -> Option<PostingList> {
        self.postings.get(term).map(|entry| entry.clone())
    }

    pub fn doc_count(&self) -> u64 {
        self.doc_count.load(Ordering::Relaxed)
    }

    /// Returns the number of tokens in `doc_id`, or 0 if unknown.
    pub fn total_tokens_in_doc(&self, doc_id: u32) -> u32 {
        self.doc_lengths.get(&doc_id).map(|v| *v).unwrap_or(0)
    }

    /// Returns the average document length across all indexed documents.
    /// Returns 1.0 if no documents have been indexed (avoids division by zero).
    pub fn avg_doc_len(&self) -> f32 {
        let count = self.doc_count.load(Ordering::Relaxed);
        if count == 0 {
            return 1.0;
        }
        self.total_token_sum.load(Ordering::Relaxed) as f32 / count as f32
    }

    pub fn vocabulary_size(&self) -> usize {
        self.postings.len()
    }

    /// Returns a snapshot of all indexed terms.
    pub fn all_terms(&self) -> Vec<String> {
        self.postings.iter().map(|e| e.key().clone()).collect()
    }

    /// Returns all indexed doc_ids.
    pub fn all_doc_ids(&self) -> Vec<u32> {
        self.doc_lengths.iter().map(|e| *e.key()).collect()
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Serialization — DashMap ↔ HashMap snapshot
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct InvertedIndexSnapshot {
    postings: HashMap<String, PostingList>,
    doc_lengths: HashMap<u32, u32>,
    doc_count: u64,
    total_token_sum: u64,
}

impl Serialize for InvertedIndex {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let snapshot = InvertedIndexSnapshot {
            postings: self
                .postings
                .iter()
                .map(|e| (e.key().clone(), e.value().clone()))
                .collect(),
            doc_lengths: self
                .doc_lengths
                .iter()
                .map(|e| (*e.key(), *e.value()))
                .collect(),
            doc_count: self.doc_count.load(Ordering::Relaxed),
            total_token_sum: self.total_token_sum.load(Ordering::Relaxed),
        };
        snapshot.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for InvertedIndex {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let snap = InvertedIndexSnapshot::deserialize(deserializer)?;
        Ok(InvertedIndex {
            postings: snap.postings.into_iter().collect(),
            doc_lengths: snap.doc_lengths.into_iter().collect(),
            doc_count: AtomicU64::new(snap.doc_count),
            total_token_sum: AtomicU64::new(snap.total_token_sum),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_index_and_lookup() {
        let idx = InvertedIndex::new();
        idx.index_document(1, &tokens(&["rust", "fast", "rust"]));
        idx.index_document(2, &tokens(&["rust", "safe"]));

        let pl = idx.lookup("rust").expect("rust should be indexed");
        assert_eq!(pl.len(), 2);
        assert_eq!(pl.tf(1), 2);
        assert_eq!(pl.tf(2), 1);
        assert!(idx.lookup("missing").is_none());
    }

    #[test]
    fn test_doc_count_and_avgdl() {
        let idx = InvertedIndex::new();
        idx.index_document(0, &tokens(&["a", "b", "c"]));
        idx.index_document(1, &tokens(&["a", "a"]));

        assert_eq!(idx.doc_count(), 2);
        assert_eq!(idx.total_tokens_in_doc(0), 3);
        assert_eq!(idx.total_tokens_in_doc(1), 2);
        assert!((idx.avg_doc_len() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_serde_roundtrip() {
        let idx = InvertedIndex::new();
        idx.index_document(10, &tokens(&["hello", "world"]));
        idx.index_document(20, &tokens(&["hello", "rust"]));

        let bytes = rmp_serde::to_vec(&idx).expect("serialize");
        let restored: InvertedIndex = rmp_serde::from_slice(&bytes).expect("deserialize");

        assert_eq!(restored.doc_count(), 2);
        let pl = restored.lookup("hello").expect("hello");
        assert_eq!(pl.len(), 2);
    }
}
