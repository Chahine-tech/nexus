use std::io::ErrorKind;
use std::path::Path;

use thiserror::Error;

use crate::indexer::inverted::InvertedIndex;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("decode error: {0}")]
    Decode(String),
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
