use std::io::ErrorKind;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::indexer::inverted::InvertedIndex;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("decode error: {0}")]
    Decode(String),
}

/// Snapshot of Node state that complements the InvertedIndex snapshot.
///
/// Stores the per-field indexes, URL mappings, document texts, and PageRank
/// graph — everything needed to fully restore a Node after restart without
/// re-crawling.
#[derive(Serialize, Deserialize)]
pub struct NodeSnapshot {
    /// Serialized `name_index` (boosted field).
    pub name_index: InvertedIndex,
    /// Serialized `body_index`.
    pub body_index: InvertedIndex,
    /// URL → doc_id mapping.
    pub url_index: std::collections::HashMap<String, u32>,
    /// doc_id → URL reverse mapping.
    pub doc_index: std::collections::HashMap<u32, String>,
    /// doc_id → body text (used for vector index rebuild after restart).
    pub doc_text: std::collections::HashMap<u32, String>,
    /// PageRank link graph: src → list of outbound destinations.
    pub pagerank_graph: std::collections::HashMap<u32, Vec<u32>>,
    /// PageRank scores from the last `iterate()` call (empty if not yet run).
    pub pagerank_scores: std::collections::HashMap<u32, f32>,
}

/// Atomically writes the inverted index to `path` using msgpack encoding.
///
/// Writes to a `.tmp` sibling first, then renames (atomic on POSIX).
/// On error the original file (if any) is left untouched.
pub fn save(index: &InvertedIndex, path: &Path) -> Result<(), StorageError> {
    let bytes = rmp_serde::to_vec(index).map_err(|e| StorageError::Decode(e.to_string()))?;
    let tmp = path.with_extension("tmp");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Atomically writes a `NodeSnapshot` to `path` using msgpack encoding.
pub fn save_node_snapshot(snapshot: &NodeSnapshot, path: &Path) -> Result<(), StorageError> {
    let bytes = rmp_serde::to_vec(snapshot).map_err(|e| StorageError::Decode(e.to_string()))?;
    let tmp = path.with_extension("tmp");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Loads a `NodeSnapshot` from `path`.
///
/// Returns `Ok(None)` if the file does not exist.
pub fn load_node_snapshot(path: &Path) -> Result<Option<NodeSnapshot>, StorageError> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(StorageError::Io(e)),
    };
    let snapshot = rmp_serde::from_slice::<NodeSnapshot>(&bytes)
        .map_err(|e| StorageError::Decode(e.to_string()))?;
    Ok(Some(snapshot))
}

/// Loads the inverted index from `path`.
///
/// Returns `Ok(None)` if the file does not exist — callers should start fresh.
/// Returns `Err` on IO errors or decode failures.
pub fn load(path: &Path) -> Result<Option<InvertedIndex>, StorageError> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(StorageError::Io(e)),
    };
    let index =
        rmp_serde::from_slice::<InvertedIndex>(&bytes).map_err(|e| StorageError::Decode(e.to_string()))?;
    Ok(Some(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::tokenizer::Tokenizer;

    fn index_with_docs(n: usize) -> InvertedIndex {
        let idx = InvertedIndex::new();
        let tok = Tokenizer::new();
        for i in 0..n {
            let text = format!("document number {i} rust programming language");
            idx.index_document(i as u32, &tok.tokenize(&text));
        }
        idx
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn save_and_load_roundtrip() {
        let path = tmp_path("nexus_test_storage_roundtrip.msgpack");

        let idx = index_with_docs(10);
        save(&idx, &path).expect("save");

        let loaded = load(&path).expect("load").expect("should be Some");
        assert_eq!(loaded.doc_count(), 10);
        assert!(loaded.lookup("rust").is_some());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let path = tmp_path("nexus_test_missing_file_xyzzy_12345.msgpack");
        // Ensure it doesn't exist.
        let _ = std::fs::remove_file(&path);
        let result = load(&path).expect("should not error");
        assert!(result.is_none());
    }

    #[test]
    fn node_snapshot_roundtrip() {
        use crate::indexer::inverted::InvertedIndex;
        use crate::indexer::tokenizer::Tokenizer;

        let path = tmp_path("nexus_test_node_snapshot.msgpack");
        let tok = Tokenizer::new();

        let name_index = InvertedIndex::new();
        name_index.index_document(1, &tok.tokenize("rust programming language"));
        let body_index = InvertedIndex::new();
        body_index.index_document(1, &tok.tokenize("fast concurrent systems"));

        let snap = NodeSnapshot {
            name_index,
            body_index,
            url_index: [("https://example.com".to_string(), 1u32)].into(),
            doc_index: [(1u32, "https://example.com".to_string())].into(),
            doc_text: [(1u32, "rust fast concurrent systems".to_string())].into(),
            pagerank_graph: [(1u32, vec![])].into(),
            pagerank_scores: [(1u32, 0.5f32)].into(),
        };

        save_node_snapshot(&snap, &path).expect("save");
        let loaded = load_node_snapshot(&path).expect("load").expect("Some");

        assert_eq!(loaded.url_index.get("https://example.com"), Some(&1u32));
        assert_eq!(loaded.doc_index.get(&1u32).map(|s| s.as_str()), Some("https://example.com"));
        assert_eq!(loaded.doc_text.get(&1u32).map(|s| s.as_str()), Some("rust fast concurrent systems"));
        assert_eq!(loaded.pagerank_scores.get(&1u32), Some(&0.5f32));
        assert!(loaded.name_index.lookup("rust").is_some());
        assert!(loaded.body_index.lookup("system").is_some()); // "systems" stems to "system"

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn atomic_write_leaves_no_tmp() {
        let path = tmp_path("nexus_test_atomic_write.msgpack");
        let tmp = path.with_extension("tmp");

        let idx = index_with_docs(5);
        save(&idx, &path).expect("save");

        assert!(path.exists(), "index file should exist");
        assert!(!tmp.exists(), "tmp file should be gone after rename");

        let _ = std::fs::remove_file(&path);
    }
}
