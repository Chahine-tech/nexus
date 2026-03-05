use std::path::Path;
use std::sync::Mutex;

use thiserror::Error;
use tree_sitter::Parser;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    TypeScript,
    Python,
}

#[derive(Debug, Error)]
pub enum AstError {
    #[error("unsupported file extension: {0}")]
    UnsupportedLanguage(String),

    #[error("failed to set language: {0}")]
    LanguageError(String),

    #[error("parse failed — tree-sitter returned no tree")]
    ParseFailed,
}

/// Parsed source file — stores source text so it can be re-parsed in spawn_blocking.
///
/// `tree_sitter::Tree` is `!Send`, so we never store it across await points.
pub struct ParsedAst {
    pub language: Language,
    /// Original source bytes.
    pub source: Vec<u8>,
}

/// Wraps a tree-sitter `Parser` behind a `Mutex` so it can be used from
/// `tokio::task::spawn_blocking` (blocking thread pool, not async runtime).
///
/// `Parser` is `!Send + !Sync`, hence the `Mutex` wrapper.
pub struct AstParser {
    parser: Mutex<Parser>,
}

impl AstParser {
    pub fn new() -> Result<Self, AstError> {
        Ok(Self { parser: Mutex::new(Parser::new()) })
    }

    /// Detects language from file extension.
    pub fn detect_language(path: &Path) -> Result<Language, AstError> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("rs") => Ok(Language::Rust),
            Some("ts") | Some("tsx") => Ok(Language::TypeScript),
            Some("py") => Ok(Language::Python),
            Some(ext) => Err(AstError::UnsupportedLanguage(ext.to_string())),
            None => Err(AstError::UnsupportedLanguage("<no extension>".to_string())),
        }
    }

    /// Parses `source` as `language`. Returns a `ParsedAst` (no Tree stored).
    ///
    /// Called inside `spawn_blocking` — `Mutex::lock()` is sync and fine here.
    pub fn parse(&self, language: Language, source: &[u8]) -> Result<ParsedAst, AstError> {
        let ts_language = match language {
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
        };

        let mut parser = self.parser.lock().expect("AstParser mutex poisoned");
        parser
            .set_language(&ts_language)
            .map_err(|e| AstError::LanguageError(e.to_string()))?;

        // Verify the source parses without error — we discard the tree.
        parser.parse(source, None).ok_or(AstError::ParseFailed)?;

        Ok(ParsedAst { language, source: source.to_vec() })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(AstParser::detect_language(Path::new("main.rs")).unwrap(), Language::Rust);
        assert_eq!(
            AstParser::detect_language(Path::new("index.ts")).unwrap(),
            Language::TypeScript
        );
        assert_eq!(
            AstParser::detect_language(Path::new("script.py")).unwrap(),
            Language::Python
        );
        assert!(AstParser::detect_language(Path::new("file.go")).is_err());
    }

    #[test]
    fn test_parse_rust() {
        let parser = AstParser::new().unwrap();
        let src = b"fn main() { println!(\"hello\"); }";
        let ast = parser.parse(Language::Rust, src).unwrap();
        assert_eq!(ast.language, Language::Rust);
        assert_eq!(ast.source, src);
    }

    #[test]
    fn test_parse_python() {
        let parser = AstParser::new().unwrap();
        let src = b"def hello():\n    print('hello')\n";
        let ast = parser.parse(Language::Python, src).unwrap();
        assert_eq!(ast.language, Language::Python);
    }
}
