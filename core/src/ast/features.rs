use std::sync::OnceLock;

use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

use crate::ast::parser::{AstError, Language, ParsedAst};

/// Extracted structural features from a source file.
#[derive(Debug, Default, Clone)]
pub struct CodeFeatures {
    /// Function/method names.
    pub function_names: Vec<String>,
    /// Type/class/struct/interface names.
    pub type_names: Vec<String>,
    /// Import paths / module names.
    pub imports: Vec<String>,
    /// String and doc-comment literals.
    pub literals: Vec<String>,
}

// ---------------------------------------------------------------------------
// Cached compiled queries — Query is Send + Sync, OnceLock is safe.
// ---------------------------------------------------------------------------

struct RustQueries {
    functions: Query,
    types: Query,
    imports: Query,
    strings: Query,
}

struct TsQueries {
    functions: Query,
    types: Query,
    imports: Query,
    strings: Query,
}

struct PyQueries {
    functions: Query,
    types: Query,
    imports: Query,
    strings: Query,
}

static RUST_QUERIES: OnceLock<RustQueries> = OnceLock::new();
static TS_QUERIES: OnceLock<TsQueries> = OnceLock::new();
static PY_QUERIES: OnceLock<PyQueries> = OnceLock::new();

fn rust_queries() -> &'static RustQueries {
    RUST_QUERIES.get_or_init(|| {
        let lang = tree_sitter_rust::LANGUAGE.into();
        RustQueries {
            functions: Query::new(&lang, "(function_item name: (identifier) @fn)").expect("static tree-sitter query is valid"),
            types: Query::new(
                &lang,
                "[(struct_item name: (type_identifier) @ty)
                  (enum_item name: (type_identifier) @ty)
                  (trait_item name: (type_identifier) @ty)
                  (impl_item type: (type_identifier) @ty)]",
            )
            .expect("static tree-sitter query is valid"),
            imports: Query::new(&lang, "(use_declaration argument: (_) @imp)").expect("static tree-sitter query is valid"),
            strings: Query::new(&lang, "(string_literal) @s").expect("static tree-sitter query is valid"),
        }
    })
}

fn ts_queries() -> &'static TsQueries {
    TS_QUERIES.get_or_init(|| {
        let lang = tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into();
        TsQueries {
            functions: Query::new(
                &lang,
                "[(function_declaration name: (identifier) @fn)
                  (method_definition name: (property_identifier) @fn)]",
            )
            .expect("static tree-sitter query is valid"),
            types: Query::new(
                &lang,
                "[(class_declaration name: (type_identifier) @ty)
                  (interface_declaration name: (type_identifier) @ty)
                  (type_alias_declaration name: (type_identifier) @ty)]",
            )
            .expect("static tree-sitter query is valid"),
            imports: Query::new(&lang, "(import_statement source: (string) @imp)").expect("static tree-sitter query is valid"),
            strings: Query::new(&lang, "(string) @s").expect("static tree-sitter query is valid"),
        }
    })
}

fn py_queries() -> &'static PyQueries {
    PY_QUERIES.get_or_init(|| {
        let lang = tree_sitter_python::LANGUAGE.into();
        PyQueries {
            functions: Query::new(
                &lang,
                "[(function_definition name: (identifier) @fn)
                  (decorated_definition definition: (function_definition name: (identifier) @fn))]",
            )
            .expect("static tree-sitter query is valid"),
            types: Query::new(&lang, "(class_definition name: (identifier) @ty)").expect("static tree-sitter query is valid"),
            imports: Query::new(
                &lang,
                "[(import_statement name: (dotted_name) @imp)
                  (import_from_statement module_name: (dotted_name) @imp)]",
            )
            .expect("static tree-sitter query is valid"),
            strings: Query::new(&lang, "(string) @s").expect("static tree-sitter query is valid"),
        }
    })
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// Extracts `CodeFeatures` from a `ParsedAst`.
///
/// Re-parses source inside this function — call from `tokio::task::spawn_blocking`.
pub fn extract(ast: &ParsedAst) -> Result<CodeFeatures, AstError> {
    let (ts_language, fq, tq, iq, sq): (
        tree_sitter::Language,
        &Query,
        &Query,
        &Query,
        &Query,
    ) = match ast.language {
        Language::Rust => {
            let q = rust_queries();
            (tree_sitter_rust::LANGUAGE.into(), &q.functions, &q.types, &q.imports, &q.strings)
        }
        Language::TypeScript => {
            let q = ts_queries();
            (
                tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
                &q.functions,
                &q.types,
                &q.imports,
                &q.strings,
            )
        }
        Language::Python => {
            let q = py_queries();
            (tree_sitter_python::LANGUAGE.into(), &q.functions, &q.types, &q.imports, &q.strings)
        }
    };

    let mut parser = Parser::new();
    parser.set_language(&ts_language).map_err(|e| AstError::LanguageError(e.to_string()))?;
    let tree = parser.parse(&ast.source, None).ok_or(AstError::ParseFailed)?;

    let src = &ast.source;
    let mut features = CodeFeatures::default();

    features.function_names = run_query(fq, &tree, src);
    features.type_names = run_query(tq, &tree, src);
    features.imports = run_query(iq, &tree, src);
    features.literals = run_query(sq, &tree, src);

    Ok(features)
}

/// Runs a query against `tree` and collects all captured node texts.
fn run_query(query: &Query, tree: &tree_sitter::Tree, src: &[u8]) -> Vec<String> {
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), src);
    let mut results = Vec::new();

    while let Some(m) = matches.next() {
        for cap in m.captures {
            if let Ok(text) = std::str::from_utf8(&src[cap.node.byte_range()]) {
                let trimmed = text.trim().trim_matches('"').trim_matches('\'');
                if !trimmed.is_empty() {
                    results.push(trimmed.to_string());
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::parser::{AstParser, Language};

    #[test]
    fn test_extract_rust_functions() {
        let parser = AstParser::new().expect("static tree-sitter query is valid");
        let src = b"fn add(a: i32, b: i32) -> i32 { a + b }\nfn main() {}";
        let ast = parser.parse(Language::Rust, src).expect("static tree-sitter query is valid");
        let features = extract(&ast).expect("static tree-sitter query is valid");
        assert!(features.function_names.contains(&"add".to_string()));
        assert!(features.function_names.contains(&"main".to_string()));
    }

    #[test]
    fn test_extract_rust_types() {
        let parser = AstParser::new().expect("static tree-sitter query is valid");
        let src = b"struct Foo { x: i32 }\nenum Bar { A, B }";
        let ast = parser.parse(Language::Rust, src).expect("static tree-sitter query is valid");
        let features = extract(&ast).expect("static tree-sitter query is valid");
        assert!(features.type_names.contains(&"Foo".to_string()));
        assert!(features.type_names.contains(&"Bar".to_string()));
    }

    #[test]
    fn test_extract_python_functions() {
        let parser = AstParser::new().expect("static tree-sitter query is valid");
        let src = b"def greet(name):\n    return 'hello ' + name\n";
        let ast = parser.parse(Language::Python, src).expect("static tree-sitter query is valid");
        let features = extract(&ast).expect("static tree-sitter query is valid");
        assert!(features.function_names.contains(&"greet".to_string()));
    }
}
