use roaring::RoaringBitmap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PostingError {
    #[error("serialization failed: {0}")]
    Serialization(#[from] rmp_serde::encode::Error),

    #[error("deserialization failed: {0}")]
    Deserialization(#[from] rmp_serde::decode::Error),
}

/// A posting list stores the set of document IDs containing a given term,
/// plus per-document term frequency (TF).
///
/// Invariants:
/// - Every doc_id in `doc_ids` has a corresponding entry in `tf_entries`.
/// - `tf_entries` is always sorted ascending by doc_id.
#[derive(Debug, Clone)]
pub struct PostingList {
    /// Presence bitmap — fast union/intersection via roaring.
    doc_ids: RoaringBitmap,
    /// Per-document TF, sorted by doc_id for O(log n) binary search.
    /// 8 bytes per entry, cache-friendly.
    tf_entries: Vec<(u32, u32)>,
}

impl PostingList {
    pub fn new() -> Self {
        Self {
            doc_ids: RoaringBitmap::new(),
            tf_entries: Vec::new(),
        }
    }

    /// Inserts or overwrites the TF for `doc_id`.
    pub fn insert(&mut self, doc_id: u32, tf: u32) {
        self.doc_ids.insert(doc_id);
        match self.tf_entries.binary_search_by_key(&doc_id, |&(id, _)| id) {
            Ok(pos) => self.tf_entries[pos].1 = tf,
            Err(pos) => self.tf_entries.insert(pos, (doc_id, tf)),
        }
    }

    /// Merges `other` into `self`, summing TF for docs present in both.
    /// Used for shard merging.
    pub fn merge(&mut self, other: &PostingList) {
        for &(doc_id, tf) in &other.tf_entries {
            match self.tf_entries.binary_search_by_key(&doc_id, |&(id, _)| id) {
                Ok(pos) => self.tf_entries[pos].1 += tf,
                Err(pos) => self.tf_entries.insert(pos, (doc_id, tf)),
            }
        }
        self.doc_ids |= &other.doc_ids;
    }

    pub fn doc_ids(&self) -> &RoaringBitmap {
        &self.doc_ids
    }

    /// Returns the TF for `doc_id`, or 0 if absent.
    pub fn tf(&self, doc_id: u32) -> u32 {
        match self.tf_entries.binary_search_by_key(&doc_id, |&(id, _)| id) {
            Ok(pos) => self.tf_entries[pos].1,
            Err(_) => 0,
        }
    }

    pub fn len(&self) -> u64 {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }
}

impl Default for PostingList {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Serialization — delta-encoded TF entries + native RoaringBitmap bytes
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct PostingListWire {
    bitmap_bytes: Vec<u8>,
    /// Delta-encoded: entry[0] = raw doc_id, entry[i] = doc_id[i] - doc_id[i-1].
    tf_deltas: Vec<(u32, u32)>,
}

impl Serialize for PostingList {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let mut bitmap_bytes = Vec::new();
        self.doc_ids
            .serialize_into(&mut bitmap_bytes)
            .map_err(|e| S::Error::custom(e.to_string()))?;

        let mut tf_deltas = Vec::with_capacity(self.tf_entries.len());
        let mut prev = 0u32;
        for &(doc_id, tf) in &self.tf_entries {
            tf_deltas.push((doc_id - prev, tf));
            prev = doc_id;
        }

        PostingListWire { bitmap_bytes, tf_deltas }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PostingList {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let wire = PostingListWire::deserialize(deserializer)?;

        let doc_ids = RoaringBitmap::deserialize_from(wire.bitmap_bytes.as_slice())
            .map_err(|e| D::Error::custom(e.to_string()))?;

        let mut tf_entries = Vec::with_capacity(wire.tf_deltas.len());
        let mut prev = 0u32;
        for (delta, tf) in wire.tf_deltas {
            let doc_id = prev + delta;
            tf_entries.push((doc_id, tf));
            prev = doc_id;
        }

        Ok(PostingList { doc_ids, tf_entries })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_tf() {
        let mut pl = PostingList::new();
        pl.insert(3, 2);
        pl.insert(7, 5);
        pl.insert(1, 1);

        assert_eq!(pl.len(), 3);
        assert_eq!(pl.tf(1), 1);
        assert_eq!(pl.tf(3), 2);
        assert_eq!(pl.tf(7), 5);
        assert_eq!(pl.tf(99), 0);
    }

    #[test]
    fn test_overwrite_tf() {
        let mut pl = PostingList::new();
        pl.insert(5, 1);
        pl.insert(5, 4);
        assert_eq!(pl.len(), 1);
        assert_eq!(pl.tf(5), 4);
    }

    #[test]
    fn test_merge() {
        let mut a = PostingList::new();
        a.insert(1, 3);
        a.insert(2, 1);

        let mut b = PostingList::new();
        b.insert(2, 2);
        b.insert(3, 4);

        a.merge(&b);

        assert_eq!(a.len(), 3);
        assert_eq!(a.tf(1), 3);
        assert_eq!(a.tf(2), 3); // 1 + 2
        assert_eq!(a.tf(3), 4);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut pl = PostingList::new();
        for i in [10u32, 20, 30, 100, 500] {
            pl.insert(i, i / 10);
        }

        let bytes = rmp_serde::to_vec(&pl).expect("serialize");
        let restored: PostingList = rmp_serde::from_slice(&bytes).expect("deserialize");

        assert_eq!(restored.len(), pl.len());
        for i in [10u32, 20, 30, 100, 500] {
            assert_eq!(restored.tf(i), pl.tf(i));
            assert!(restored.doc_ids().contains(i));
        }
    }
}
