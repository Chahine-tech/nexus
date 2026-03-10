use std::collections::HashSet;
use std::sync::LazyLock;

use crate::indexer::inverted::InvertedIndex;

// ---------------------------------------------------------------------------
// Stop words (same set as tokenizer.rs)
// ---------------------------------------------------------------------------

static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
        "with", "this", "that", "it", "as", "at", "by", "from", "on", "be",
        "how", "what", "why", "when", "where", "do", "does", "can", "i",
    ]
    .into_iter()
    .collect()
});

// ---------------------------------------------------------------------------
// Logistic regression weights
// ---------------------------------------------------------------------------
//
// alpha = sigmoid(w · x + b)
//
// Positive weight → feature pushes alpha toward 1.0 (BM25 dominant = lexical query).
// Negative weight → feature pushes alpha toward 0.0 (vector dominant = semantic query).
//
// Trained on 115 hand-labeled queries via tools/train_alpha.py (no API calls).
// 5-fold CV accuracy: 0.890. All directional sanity checks pass.
// Re-run tools/train_alpha.py to retrain after adding queries to the dataset.

const WEIGHTS: [f32; 6] = [
    -5.4834, // x[0] query_len_norm      — longer queries → more semantic → lower alpha
    -0.0692, // x[1] avg_idf_norm        — near-zero: approximated without real index
     0.0000, // x[2] idf_variance_norm   — near-zero: approximated without real index
     3.9112, // x[3] has_code_token      — code token → lexical → higher alpha
    -7.8054, // x[4] stop_word_ratio     — natural language → lower alpha
    -0.7509, // x[5] token_entropy_norm  — diverse lengths → semantic → lower alpha
];
const BIAS: f32 = 2.4375;

// ---------------------------------------------------------------------------
// QueryFeatures
// ---------------------------------------------------------------------------

/// Six-dimensional feature vector extracted from a query.
///
/// Used by `predict_alpha()` to select the BM25/vector blend for `HybridScorer`.
/// All fields are normalized to approximately [0, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct QueryFeatures {
    /// Token count / 10, clamped to [0, 1].
    pub query_len_norm: f32,
    /// Mean IDF of query tokens, normalized by ln(N+1).
    pub avg_idf_norm: f32,
    /// Variance of IDF values across query tokens, normalized.
    pub idf_variance_norm: f32,
    /// 1.0 if any token contains '_' or starts with an uppercase letter, else 0.0.
    pub has_code_token: f32,
    /// Fraction of raw whitespace tokens that are stop words.
    pub stop_word_ratio: f32,
    /// Shannon entropy of token lengths, normalized by log2(max_len + 1).
    pub token_entropy_norm: f32,
}

impl QueryFeatures {
    /// Extracts features from a query.
    ///
    /// `raw_query` is the original unprocessed string (used for stop word ratio and
    /// code token detection before lowercasing). `tokens` is the post-tokenization list.
    /// `index` is used to compute IDF-based features.
    pub fn extract(raw_query: &str, tokens: &[String], index: &InvertedIndex) -> Self {
        let n = index.doc_count() as f32;
        let max_idf = if n > 0.0 { (n + 1.0).ln() } else { 1.0 };

        // --- IDF features ---
        let idfs: Vec<f32> = tokens
            .iter()
            .map(|term| {
                index
                    .with_posting(term, |pl| {
                        let df = pl.len() as f32;
                        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
                    })
                    .unwrap_or(0.0)
            })
            .collect();

        let avg_idf = if idfs.is_empty() {
            0.0
        } else {
            idfs.iter().sum::<f32>() / idfs.len() as f32
        };

        let idf_variance = if idfs.len() < 2 {
            0.0
        } else {
            let mean = avg_idf;
            idfs.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / idfs.len() as f32
        };

        let avg_idf_norm = (avg_idf / max_idf).clamp(0.0, 1.0);
        // Max variance when half values are 0 and half are max_idf: (max_idf/2)^2
        let max_variance = (max_idf / 2.0).powi(2).max(1e-6);
        let idf_variance_norm = (idf_variance / max_variance).clamp(0.0, 1.0);

        // --- Length feature ---
        let query_len_norm = (tokens.len() as f32 / 10.0).clamp(0.0, 1.0);

        // --- Code token detection (on raw query before lowercasing) ---
        let has_code = raw_query.contains('_')
            || raw_query.split_whitespace().any(|w| {
                // CamelCase: has at least one uppercase letter after the first character
                w.chars().skip(1).any(|c| c.is_uppercase())
            });
        let has_code_token = if has_code { 1.0 } else { 0.0 };

        // --- Stop word ratio (on raw whitespace tokens) ---
        let raw_tokens: Vec<&str> = raw_query.split_whitespace().collect();
        let stop_word_ratio = if raw_tokens.is_empty() {
            0.0
        } else {
            let stop_count = raw_tokens
                .iter()
                .filter(|w| STOP_WORDS.contains(w.to_lowercase().as_str()))
                .count();
            stop_count as f32 / raw_tokens.len() as f32
        };

        // --- Token length entropy ---
        let token_entropy_norm = token_length_entropy(tokens);

        Self {
            query_len_norm,
            avg_idf_norm,
            idf_variance_norm,
            has_code_token,
            stop_word_ratio,
            token_entropy_norm,
        }
    }

    /// Predicts the BM25 weight α ∈ (0, 1) using logistic regression.
    ///
    /// Returns values close to 1.0 for lexical/code queries and close to 0.0
    /// for long natural-language semantic queries.
    pub fn predict_alpha(&self) -> f32 {
        let x = [
            self.query_len_norm,
            self.avg_idf_norm,
            self.idf_variance_norm,
            self.has_code_token,
            self.stop_word_ratio,
            self.token_entropy_norm,
        ];
        let logit: f32 = x.iter().zip(WEIGHTS.iter()).map(|(xi, wi)| xi * wi).sum::<f32>() + BIAS;
        sigmoid(logit)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Shannon entropy of token lengths, normalized by log2(longest_token_len + 1).
fn token_length_entropy(tokens: &[String]) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }
    // Build length frequency distribution.
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for t in tokens {
        *counts.entry(t.len()).or_insert(0) += 1;
    }
    let n = tokens.len() as f32;
    let entropy: f32 = counts
        .values()
        .map(|&c| {
            let p = c as f32 / n;
            -p * p.log2()
        })
        .sum();

    let max_len = tokens.iter().map(|t| t.len()).max().unwrap_or(1);
    let max_entropy = (max_len as f32 + 1.0).log2().max(1.0);
    (entropy / max_entropy).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::indexer::inverted::InvertedIndex;

    fn make_index() -> Arc<InvertedIndex> {
        let idx = Arc::new(InvertedIndex::new());
        // Diverse corpus: code docs + natural language docs.
        let docs: &[&[&str]] = &[
            &["tokio", "spawn", "async", "rust"],
            &["async", "await", "future", "rust", "error"],
            &["how", "to", "handle", "errors", "gracefully"],
            &["python", "dynamic", "scripting", "language"],
            &["hashmap", "btreemap", "collections", "rust"],
        ];
        for (i, tokens) in docs.iter().enumerate() {
            let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
            idx.index_document(i as u32, &owned);
        }
        idx
    }

    #[test]
    fn test_predict_alpha_range() {
        let idx = make_index();
        let queries = [
            ("tokio", vec!["tokio".to_string()]),
            ("how to handle async errors in rust gracefully", vec!["how".to_string(), "handle".to_string(), "async".to_string(), "errors".to_string(), "rust".to_string()]),
            ("myFunction_name", vec!["myfunction".to_string(), "name".to_string()]),
        ];
        for (raw, tokens) in &queries {
            let f = QueryFeatures::extract(raw, tokens, &idx);
            let alpha = f.predict_alpha();
            assert!(alpha > 0.0 && alpha < 1.0, "alpha={alpha} out of range for query '{raw}'");
        }
    }

    #[test]
    fn test_code_query_higher_alpha_than_natural() {
        let idx = make_index();
        let code_tokens = vec!["tokio".to_string()];
        let natural_tokens = vec!["how".to_string(), "to".to_string(), "handle".to_string(), "errors".to_string(), "gracefully".to_string()];

        let alpha_code = QueryFeatures::extract("tokio", &code_tokens, &idx).predict_alpha();
        let alpha_natural = QueryFeatures::extract(
            "how to handle errors gracefully",
            &natural_tokens,
            &idx,
        ).predict_alpha();

        assert!(
            alpha_code > alpha_natural,
            "code query alpha ({alpha_code:.3}) should be > natural language alpha ({alpha_natural:.3})"
        );
    }

    #[test]
    fn test_extract_has_code_token_snake_case() {
        let idx = make_index();
        let f = QueryFeatures::extract("my_function", &["my".to_string(), "function".to_string()], &idx);
        assert_eq!(f.has_code_token, 1.0);
    }

    #[test]
    fn test_extract_has_code_token_camel_case() {
        let idx = make_index();
        let f = QueryFeatures::extract("myFunction", &["myfunction".to_string()], &idx);
        assert_eq!(f.has_code_token, 1.0);
    }

    #[test]
    fn test_extract_no_code_token() {
        let idx = make_index();
        let f = QueryFeatures::extract("how to handle errors", &["handle".to_string(), "errors".to_string()], &idx);
        assert_eq!(f.has_code_token, 0.0);
    }

    #[test]
    fn test_stop_word_ratio_all_stop_words() {
        let idx = make_index();
        let f = QueryFeatures::extract("how to do it", &[], &idx);
        assert!(f.stop_word_ratio > 0.5, "expected high stop word ratio, got {}", f.stop_word_ratio);
    }

    #[test]
    fn test_predict_alpha_empty_index() {
        let idx = Arc::new(InvertedIndex::new());
        let f = QueryFeatures::extract("tokio", &["tokio".to_string()], &idx);
        let alpha = f.predict_alpha();
        // With empty index, IDF features are 0 — should still be in valid range.
        assert!(alpha > 0.0 && alpha < 1.0);
    }

    #[test]
    fn test_query_len_norm_clamped() {
        let idx = make_index();
        // 20 tokens → len_norm should clamp to 1.0
        let tokens: Vec<String> = (0..20).map(|i| format!("token{i}")).collect();
        let raw = tokens.join(" ");
        let f = QueryFeatures::extract(&raw, &tokens, &idx);
        assert_eq!(f.query_len_norm, 1.0);
    }
}
