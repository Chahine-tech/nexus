use bitpacking::{BitPacker, BitPacker4x};
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
// BP128 helpers
// ---------------------------------------------------------------------------

/// Compresses a slice of u32 values using BitPacker4x (SIMD BP128).
///
/// Wire layout per block of 128 integers:
///   - 1 byte: num_bits (minimum bits needed for the block's max value)
///   - num_bits * 16 bytes: compressed block data
///
/// Partial tail blocks (len % 128 != 0) are zero-padded to 128 before
/// compression; the caller stores `count` separately to know how many values
/// to read back from the tail block.
fn compress_stream(values: &[u32]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }

    let packer = BitPacker4x::new();
    let tail_len = values.len() % BitPacker4x::BLOCK_LEN;
    let num_blocks = values.len().div_ceil(BitPacker4x::BLOCK_LEN);

    let mut out = Vec::with_capacity(num_blocks * (1 + BitPacker4x::BLOCK_LEN * 4));

    // Full blocks — chunks_exact guarantees exactly BLOCK_LEN elements per chunk.
    let chunks = values.chunks_exact(BitPacker4x::BLOCK_LEN);
    let tail = chunks.remainder();

    for chunk in chunks {
        let block: &[u32; BitPacker4x::BLOCK_LEN] = chunk.try_into().unwrap_or_else(|_| unreachable!());
        let num_bits = packer.num_bits(block);
        let compressed_len = (num_bits as usize) * BitPacker4x::BLOCK_LEN / 8;
        out.push(num_bits);
        if num_bits > 0 {
            let start = out.len();
            out.resize(start + compressed_len, 0u8);
            packer.compress(block, &mut out[start..], num_bits);
        }
    }

    // Tail block — zero-pad to BLOCK_LEN.
    if tail_len > 0 {
        let mut padded = [0u32; BitPacker4x::BLOCK_LEN];
        padded[..tail_len].copy_from_slice(tail);
        let num_bits = packer.num_bits(&padded);
        let compressed_len = (num_bits as usize) * BitPacker4x::BLOCK_LEN / 8;
        out.push(num_bits);
        if num_bits > 0 {
            let start = out.len();
            out.resize(start + compressed_len, 0u8);
            packer.compress(&padded, &mut out[start..], num_bits);
        }
    }

    out
}

/// Decompresses a BP128-encoded byte stream back into `count` u32 values.
fn decompress_stream(bytes: &[u8], count: usize) -> Result<Vec<u32>, String> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let packer = BitPacker4x::new();
    let num_full_blocks = count / BitPacker4x::BLOCK_LEN;
    let tail_len = count % BitPacker4x::BLOCK_LEN;

    let mut result = Vec::with_capacity(count);
    let mut cursor = 0usize;

    // Full blocks.
    for _ in 0..num_full_blocks {
        if cursor >= bytes.len() {
            return Err(format!("unexpected end of compressed stream at cursor={cursor}"));
        }
        let num_bits = bytes[cursor];
        cursor += 1;
        let compressed_len = (num_bits as usize) * BitPacker4x::BLOCK_LEN / 8;
        let mut block = [0u32; BitPacker4x::BLOCK_LEN];
        if num_bits > 0 {
            if cursor + compressed_len > bytes.len() {
                return Err(format!(
                    "compressed block overflows buffer: cursor={cursor}, len={compressed_len}"
                ));
            }
            packer.decompress(&bytes[cursor..cursor + compressed_len], &mut block, num_bits);
        }
        result.extend_from_slice(&block);
        cursor += compressed_len;
    }

    // Tail block.
    if tail_len > 0 {
        if cursor >= bytes.len() {
            return Err("unexpected end of stream before tail block".to_string());
        }
        let num_bits = bytes[cursor];
        cursor += 1;
        let compressed_len = (num_bits as usize) * BitPacker4x::BLOCK_LEN / 8;
        let mut block = [0u32; BitPacker4x::BLOCK_LEN];
        if num_bits > 0 {
            if cursor + compressed_len > bytes.len() {
                return Err("tail block overflows buffer".to_string());
            }
            packer.decompress(&bytes[cursor..cursor + compressed_len], &mut block, num_bits);
        }
        result.extend_from_slice(&block[..tail_len]);
        cursor += compressed_len;
    }

    if cursor != bytes.len() {
        return Err(format!(
            "compressed stream has {} trailing bytes (cursor={cursor}, total={})",
            bytes.len() - cursor,
            bytes.len()
        ));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Serialization — BP128-compressed streams + native RoaringBitmap bytes
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct PostingListWire {
    /// RoaringBitmap serialized in its native format.
    bitmap_bytes: Vec<u8>,
    /// Total number of (doc_id, tf) pairs encoded in the streams below.
    count: u32,
    /// BP128-compressed stream of doc_id deltas:
    ///   delta[0] = doc_id[0], delta[i] = doc_id[i] - doc_id[i-1].
    doc_id_bytes: Vec<u8>,
    /// BP128-compressed stream of TF values (parallel to doc_id_bytes).
    tf_bytes: Vec<u8>,
}

impl Serialize for PostingList {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;

        let mut bitmap_bytes = Vec::new();
        self.doc_ids
            .serialize_into(&mut bitmap_bytes)
            .map_err(|e| S::Error::custom(e.to_string()))?;

        let count = self.tf_entries.len() as u32;

        let mut doc_id_deltas = Vec::with_capacity(self.tf_entries.len());
        let mut tfs = Vec::with_capacity(self.tf_entries.len());
        let mut prev = 0u32;
        for &(doc_id, tf) in &self.tf_entries {
            doc_id_deltas.push(doc_id - prev);
            prev = doc_id;
            tfs.push(tf);
        }

        let doc_id_bytes = compress_stream(&doc_id_deltas);
        let tf_bytes = compress_stream(&tfs);

        PostingListWire { bitmap_bytes, count, doc_id_bytes, tf_bytes }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PostingList {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let wire = PostingListWire::deserialize(deserializer)?;

        let doc_ids = RoaringBitmap::deserialize_from(wire.bitmap_bytes.as_slice())
            .map_err(|e| D::Error::custom(e.to_string()))?;

        let n = wire.count as usize;
        let doc_id_deltas =
            decompress_stream(&wire.doc_id_bytes, n).map_err(D::Error::custom)?;
        let tfs = decompress_stream(&wire.tf_bytes, n).map_err(D::Error::custom)?;

        let mut tf_entries = Vec::with_capacity(n);
        let mut prev = 0u32;
        for (delta, tf) in doc_id_deltas.into_iter().zip(tfs) {
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

    fn roundtrip(pl: &PostingList) -> PostingList {
        let bytes = rmp_serde::to_vec(pl).expect("serialize");
        rmp_serde::from_slice(&bytes).expect("deserialize")
    }

    #[test]
    fn test_serde_roundtrip() {
        // 5 entries — exercises tail-only path (< 128).
        let mut pl = PostingList::new();
        for i in [10u32, 20, 30, 100, 500] {
            pl.insert(i, i / 10);
        }

        let restored = roundtrip(&pl);

        assert_eq!(restored.len(), pl.len());
        for i in [10u32, 20, 30, 100, 500] {
            assert_eq!(restored.tf(i), pl.tf(i));
            assert!(restored.doc_ids().contains(i));
        }
    }

    #[test]
    fn test_serde_roundtrip_full_block() {
        // 128 entries — exactly one full block, no tail.
        let mut pl = PostingList::new();
        for i in 0u32..128 {
            pl.insert(i * 10, (i % 10) + 1);
        }

        let restored = roundtrip(&pl);

        assert_eq!(restored.len(), 128);
        for i in 0u32..128 {
            assert_eq!(restored.tf(i * 10), (i % 10) + 1);
            assert!(restored.doc_ids().contains(i * 10));
        }
    }

    #[test]
    fn test_serde_roundtrip_full_plus_tail() {
        // 130 entries — one full block + tail of 2.
        let mut pl = PostingList::new();
        for i in 0u32..130 {
            pl.insert(i * 7, (i % 5) + 1);
        }

        let restored = roundtrip(&pl);

        assert_eq!(restored.len(), 130);
        for i in 0u32..130 {
            assert_eq!(restored.tf(i * 7), (i % 5) + 1);
            assert!(restored.doc_ids().contains(i * 7));
        }
    }

    #[test]
    fn test_serde_empty() {
        let pl = PostingList::new();
        let restored = roundtrip(&pl);
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }
}
