use std::collections::HashSet;
use std::sync::LazyLock;

static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
        "with", "this", "that", "it", "as", "at", "by", "from", "on", "be",
    ]
    .into_iter()
    .collect()
});

/// Stateless tokenizer for plain text and code.
///
/// Pipeline: lowercase → split on non-alphanumeric → filter stop words + short tokens.
pub struct Tokenizer;

impl Tokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Tokenizes `text` into a list of normalized tokens.
    ///
    /// Tokens shorter than 2 characters and stop words are discarded.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|tok| tok.len() >= 2 && !STOP_WORDS.contains(tok))
            .map(|tok| tok.to_string())
            .collect()
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let t = Tokenizer::new();
        let tokens = t.tokenize("Hello World, this is Rust!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"rust".to_string()));
    }

    #[test]
    fn test_stop_words_filtered() {
        let t = Tokenizer::new();
        let tokens = t.tokenize("the quick brown fox");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
    }

    #[test]
    fn test_short_tokens_filtered() {
        let t = Tokenizer::new();
        let tokens = t.tokenize("a fn x hello");
        assert!(!tokens.contains(&"a".to_string()));
        assert!(!tokens.contains(&"x".to_string()));
        // "fn" is 2 chars and not a stop word — should pass
        assert!(tokens.contains(&"fn".to_string()));
        assert!(tokens.contains(&"hello".to_string()));
    }

    #[test]
    fn test_empty_input() {
        let t = Tokenizer::new();
        assert!(t.tokenize("").is_empty());
        assert!(t.tokenize("   ").is_empty());
    }

    #[test]
    fn test_code_like_input() {
        let t = Tokenizer::new();
        let tokens = t.tokenize("fn main() -> Result<(), Error> {");
        assert!(tokens.contains(&"main".to_string()));
        assert!(tokens.contains(&"result".to_string()));
        assert!(tokens.contains(&"error".to_string()));
    }
}
