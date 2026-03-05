use crate::ast::features::CodeFeatures;

/// Splits an identifier into lowercase tokens.
///
/// Handles both snake_case (`my_func`) and camelCase/PascalCase (`MyFunc`, `myFunc`).
pub fn split_identifier(ident: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in ident.chars() {
        if ch == '_' || ch == '-' {
            if !current.is_empty() {
                tokens.push(current.to_lowercase());
                current.clear();
            }
        } else if ch.is_uppercase() && !current.is_empty() {
            tokens.push(current.to_lowercase());
            current.clear();
            current.push(ch);
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        tokens.push(current.to_lowercase());
    }

    tokens
}

/// Normalizes a function/method signature string into indexable tokens.
///
/// Strips punctuation and splits on whitespace + identifier boundaries.
pub fn normalize_signature(sig: &str) -> Vec<String> {
    sig.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .flat_map(|token| split_identifier(token))
        .filter(|t| t.len() >= 2)
        .collect()
}

/// Converts `CodeFeatures` into a flat list of normalized tokens for indexing.
pub fn tokens_from_features(features: &CodeFeatures) -> Vec<String> {
    let mut tokens = Vec::new();

    for name in &features.function_names {
        tokens.extend(split_identifier(name));
    }
    for name in &features.type_names {
        tokens.extend(split_identifier(name));
    }
    for imp in &features.imports {
        tokens.extend(normalize_signature(imp));
    }
    for lit in &features.literals {
        // Literals are split on whitespace only — preserve word boundaries.
        tokens.extend(
            lit.split_whitespace()
                .filter(|t| t.len() >= 2)
                .map(|t| t.to_lowercase()),
        );
    }

    tokens.dedup();
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_snake_case() {
        assert_eq!(split_identifier("my_function"), vec!["my", "function"]);
    }

    #[test]
    fn test_split_camel_case() {
        assert_eq!(split_identifier("MyStruct"), vec!["my", "struct"]);
        assert_eq!(split_identifier("parseDocument"), vec!["parse", "document"]);
    }

    #[test]
    fn test_split_single_word() {
        assert_eq!(split_identifier("main"), vec!["main"]);
    }

    #[test]
    fn test_normalize_signature() {
        let tokens = normalize_signature("fn parse_document(src: &str) -> Result<Doc>");
        assert!(tokens.contains(&"parse".to_string()));
        assert!(tokens.contains(&"document".to_string()));
        assert!(tokens.contains(&"src".to_string()));
        assert!(tokens.contains(&"str".to_string()));
    }

    #[test]
    fn test_tokens_from_features() {
        let features = CodeFeatures {
            function_names: vec!["parseDocument".to_string(), "main".to_string()],
            type_names: vec!["InvertedIndex".to_string()],
            imports: vec![],
            literals: vec!["hello world".to_string()],
        };
        let tokens = tokens_from_features(&features);
        assert!(tokens.contains(&"parse".to_string()));
        assert!(tokens.contains(&"document".to_string()));
        assert!(tokens.contains(&"inverted".to_string()));
        assert!(tokens.contains(&"index".to_string()));
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }
}
