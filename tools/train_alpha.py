"""
train_alpha.py — Fit logistic regression weights for HybridScorer adaptive alpha.

Usage:
    python tools/train_alpha.py

Output:
    Trained WEIGHTS and BIAS constants ready to paste into
    core/src/scoring/query_features.rs

No API calls. No internet access required.
Dependencies: numpy, scikit-learn (pip install numpy scikit-learn)

Feature vector (mirrors query_features.rs exactly):
    x[0] = query_len_norm       — token count / 10, clamped [0,1]
    x[1] = avg_idf_norm         — mean IDF / ln(N+1), approx with df=1
    x[2] = idf_variance_norm    — IDF variance / max_variance
    x[3] = has_code_token       — 1.0 if snake_case or CamelCase present
    x[4] = stop_word_ratio      — fraction of raw tokens that are stop words
    x[5] = token_entropy_norm   — Shannon entropy of token lengths / log2(max_len+1)

Alpha label:
    1.0 = pure BM25 (lexical/code query)
    0.0 = pure vector (semantic/natural language query)
"""

import math
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Stop words (identical to query_features.rs)
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
    "with", "this", "that", "it", "as", "at", "by", "from", "on", "be",
    "how", "what", "why", "when", "where", "do", "does", "can", "i",
}

FEATURE_NAMES = [
    "query_len_norm",
    "avg_idf_norm",
    "idf_variance_norm",
    "has_code_token",
    "stop_word_ratio",
    "token_entropy_norm",
]

# ---------------------------------------------------------------------------
# Dataset — (raw_query, post_tokenization_tokens, alpha_label)
#
# alpha_label:
#   ~0.85  = strongly lexical (code query, BM25 dominant)
#   ~0.15  = strongly semantic (natural language, vector dominant)
#   ~0.50  = balanced / hybrid
#
# Tokenization: lowercase, split on non-alphanumeric, filter stop words + len<2
# ---------------------------------------------------------------------------

DATASET = [
    # --- Code queries → alpha ~0.80–0.90 ---
    ("tokio_spawn",                     ["tokio", "spawn"],                                     0.88),
    ("HashMap::new",                    ["hashmap", "new"],                                     0.85),
    ("Arc<Mutex<T>>",                   ["arc", "mutex"],                                       0.83),
    ("Vec::with_capacity",              ["vec", "with", "capacity"],                            0.85),
    ("impl Iterator for MyStruct",      ["impl", "iterator", "mystruct"],                       0.82),
    ("fn main() -> Result",             ["fn", "main", "result"],                               0.80),
    ("#[derive(Debug, Clone)]",         ["derive", "debug", "clone"],                           0.82),
    ("async fn handle_request",         ["async", "fn", "handle", "request"],                   0.80),
    ("Box<dyn Error>",                  ["box", "dyn", "error"],                                0.83),
    ("std::collections::BTreeMap",      ["std", "collections", "btreemap"],                     0.87),
    ("MyError::InvalidInput",           ["myerror", "invalidinput"],                            0.88),
    ("parse::<i32>()",                  ["parse", "i32"],                                       0.85),
    ("RwLock<HashMap<String, Vec>>",    ["rwlock", "hashmap", "string", "vec"],                 0.86),
    ("impl From<io::Error>",            ["impl", "from", "io", "error"],                        0.81),
    ("#[tokio::main]",                  ["tokio", "main"],                                      0.87),
    ("serde::Deserialize",              ["serde", "deserialize"],                               0.86),
    ("PhantomData<T>",                  ["phantomdata"],                                        0.89),
    ("Rc::clone(&ptr)",                 ["rc", "clone", "ptr"],                                 0.84),
    ("tracing::instrument",             ["tracing", "instrument"],                              0.85),
    ("axum::Router::new",               ["axum", "router", "new"],                              0.85),
    ("anyhow::bail!",                   ["anyhow", "bail"],                                     0.86),
    ("tokio::sync::mpsc",               ["tokio", "sync", "mpsc"],                              0.87),
    ("std::sync::atomic::AtomicUsize",  ["std", "sync", "atomic", "atomicusize"],               0.87),
    ("impl_trait_for_type",             ["impl", "trait", "for", "type"],                       0.83),
    ("unwrap_or_else",                  ["unwrap", "or", "else"],                               0.82),
    ("collect::<Vec<_>>()",             ["collect", "vec"],                                     0.86),
    ("cargo test --release",            ["cargo", "test", "release"],                           0.80),
    ("rustfmt::skip",                   ["rustfmt", "skip"],                                    0.88),
    ("lifetime 'a parameter",           ["lifetime", "parameter"],                              0.78),
    ("HashMap entry API",               ["hashmap", "entry", "api"],                            0.78),
    ("iter().filter_map()",             ["iter", "filter", "map"],                              0.82),
    ("spawn_blocking closure",          ["spawn", "blocking", "closure"],                       0.80),
    ("BufReader::new(file)",            ["bufreader", "new", "file"],                           0.84),
    ("String::from_utf8_lossy",         ["string", "from", "utf8", "lossy"],                   0.85),
    ("mutable reference &mut",          ["mutable", "reference", "mut"],                        0.78),

    # --- Natural language queries → alpha ~0.10–0.30 ---
    ("how to handle errors gracefully",                     ["handle", "errors", "gracefully"],                         0.15),
    ("what is the difference between arc and rc",           ["difference", "between", "arc", "rc"],                    0.18),
    ("best way to structure a rust project",                ["best", "way", "structure", "rust", "project"],           0.20),
    ("why is my program running slow",                      ["program", "running", "slow"],                            0.22),
    ("how do async runtimes work internally",               ["async", "runtimes", "work", "internally"],               0.18),
    ("explain ownership and borrowing in rust",             ["explain", "ownership", "borrowing", "rust"],             0.15),
    ("when should i use channels instead of mutex",         ["channels", "instead", "mutex"],                          0.20),
    ("how to avoid deadlocks in concurrent code",           ["avoid", "deadlocks", "concurrent", "code"],              0.17),
    ("what are the trade offs between performance and safety", ["trade", "offs", "between", "performance", "safety"],  0.14),
    ("how does the borrow checker work",                    ["borrow", "checker", "work"],                             0.18),
    ("why does rust have no garbage collector",             ["rust", "garbage", "collector"],                          0.20),
    ("how to write idiomatic rust code",                    ["write", "idiomatic", "rust", "code"],                    0.22),
    ("what is zero cost abstraction",                       ["zero", "cost", "abstraction"],                           0.19),
    ("difference between stack and heap allocation",        ["difference", "between", "stack", "heap", "allocation"],  0.16),
    ("how to read a file line by line",                     ["read", "file", "line", "by", "line"],                    0.22),
    ("explain the actor model for concurrency",             ["explain", "actor", "model", "concurrency"],              0.15),
    ("how to serialize and deserialize json",               ["serialize", "deserialize", "json"],                      0.25),
    ("what is the difference between threads and tasks",    ["difference", "between", "threads", "tasks"],             0.18),
    ("how to profile a rust application",                   ["profile", "rust", "application"],                        0.22),
    ("when to use trait objects vs generics",               ["trait", "objects", "vs", "generics"],                    0.20),
    ("how does tokio schedule tasks",                       ["tokio", "schedule", "tasks"],                            0.22),
    ("explain lifetime elision rules",                      ["explain", "lifetime", "elision", "rules"],               0.18),
    ("what makes rust memory safe without gc",              ["makes", "rust", "memory", "safe", "without", "gc"],      0.15),
    ("how to implement a state machine",                    ["implement", "state", "machine"],                         0.22),
    ("best practices for error handling in libraries",      ["best", "practices", "error", "handling", "libraries"],   0.18),
    ("how does cargo resolve dependency conflicts",         ["cargo", "resolve", "dependency", "conflicts"],           0.20),
    ("why use result instead of exceptions",                ["use", "result", "instead", "exceptions"],                0.18),
    ("how to write unit tests in rust",                     ["write", "unit", "tests", "rust"],                        0.22),
    ("what is monomorphization and when does it matter",    ["monomorphization", "when", "matter"],                    0.17),
    ("how to build a rest api with axum",                   ["build", "rest", "api", "axum"],                          0.25),
    ("explain send and sync traits",                        ["explain", "send", "sync", "traits"],                     0.18),
    ("when should you prefer box over stack allocation",    ["prefer", "box", "over", "stack", "allocation"],          0.17),
    ("how to handle backpressure in async streams",         ["handle", "backpressure", "async", "streams"],            0.20),
    ("what is the newtype pattern used for",                ["newtype", "pattern", "used"],                            0.20),
    ("how to share state between tokio tasks",              ["share", "state", "between", "tokio", "tasks"],           0.20),

    # --- Natural language (continued) — more semantic diversity ---
    ("what is the best way to do zero copy deserialization",    ["best", "zero", "copy", "deserialization"],               0.20),
    ("how to expose a rust library to python",                  ["expose", "rust", "library", "python"],                   0.18),
    ("when is unsafe code justified",                           ["unsafe", "code", "justified"],                           0.20),
    ("how to build a plugin system in rust",                    ["build", "plugin", "system", "rust"],                     0.22),
    ("difference between future and stream",                    ["difference", "between", "future", "stream"],             0.18),
    ("why does my async code block the executor",               ["async", "code", "block", "executor"],                    0.20),
    ("how to reduce binary size in release mode",               ["reduce", "binary", "size", "release", "mode"],           0.18),
    ("what is the difference between clone and copy",           ["difference", "between", "clone", "copy"],                0.16),
    ("how to implement custom iterator adapters",               ["implement", "custom", "iterator", "adapters"],           0.20),
    ("when to use an arena allocator",                          ["use", "arena", "allocator"],                             0.18),
    ("how to model a domain with algebraic types",              ["model", "domain", "algebraic", "types"],                 0.17),
    ("explain the difference between box rc and arc",           ["explain", "difference", "between", "box", "rc", "arc"],  0.15),
    ("how to write a compiler frontend in rust",                ["write", "compiler", "frontend", "rust"],                 0.22),
    ("what are the limits of the type system",                  ["limits", "type", "system"],                              0.18),
    ("how to avoid unnecessary heap allocations",               ["avoid", "unnecessary", "heap", "allocations"],           0.20),

    # --- Mixed / hybrid queries → alpha ~0.40–0.60 ---
    ("tokio async runtime",                 ["tokio", "async", "runtime"],                          0.55),
    ("rust error handling Result",          ["rust", "error", "handling", "result"],                0.50),
    ("hashmap performance benchmark",       ["hashmap", "performance", "benchmark"],                0.52),
    ("vector similarity search cosine",     ["vector", "similarity", "search", "cosine"],          0.45),
    ("bm25 ranking algorithm",              ["bm25", "ranking", "algorithm"],                       0.50),
    ("rayon parallel iterator",             ["rayon", "parallel", "iterator"],                      0.58),
    ("serde json serialization",            ["serde", "json", "serialization"],                     0.55),
    ("axum middleware tower",               ["axum", "middleware", "tower"],                        0.57),
    ("hyper http client",                   ["hyper", "http", "client"],                            0.55),
    ("tracing structured logging",          ["tracing", "structured", "logging"],                   0.50),
    ("roaring bitmap compression",          ["roaring", "bitmap", "compression"],                   0.52),
    ("hnsw approximate nearest neighbor",   ["hnsw", "approximate", "nearest", "neighbor"],         0.48),
    ("diesel query builder postgres",       ["diesel", "query", "builder", "postgres"],             0.55),
    ("wasm bindgen javascript interop",     ["wasm", "bindgen", "javascript", "interop"],           0.57),
    ("criterion benchmark harness",         ["criterion", "benchmark", "harness"],                  0.52),
    ("async stream tokio broadcast",        ["async", "stream", "tokio", "broadcast"],              0.54),
    ("thiserror derive macro",              ["thiserror", "derive", "macro"],                       0.58),
    ("crossbeam channel sync",              ["crossbeam", "channel", "sync"],                       0.55),
    ("quinn quic protocol",                 ["quinn", "quic", "protocol"],                          0.52),
    ("tree sitter parsing",                 ["tree", "sitter", "parsing"],                          0.50),
    ("inverted index posting list",         ["inverted", "index", "posting", "list"],               0.48),
    ("distributed gossip protocol",         ["distributed", "gossip", "protocol"],                  0.47),
    ("hyperloglog cardinality estimation",  ["hyperloglog", "cardinality", "estimation"],           0.48),
    ("laplace differential privacy",        ["laplace", "differential", "privacy"],                 0.47),
    ("pagerank convergence graph",          ["pagerank", "convergence", "graph"],                   0.48),
    ("tfidf cosine similarity",             ["tfidf", "cosine", "similarity"],                      0.48),
    ("simd bitpacking compression",         ["simd", "bitpacking", "compression"],                  0.55),
    ("raft consensus distributed",          ["raft", "consensus", "distributed"],                   0.48),
    ("lru cache eviction policy",           ["lru", "cache", "eviction", "policy"],                 0.50),
    ("bloom filter false positive rate",    ["bloom", "filter", "false", "positive", "rate"],       0.48),
]

# ---------------------------------------------------------------------------
# Feature extraction (mirrors query_features.rs exactly)
# ---------------------------------------------------------------------------

DOC_COUNT = 1000  # representative corpus size for IDF approximation


def _token_length_entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    lengths = [len(t) for t in tokens]
    counts = Counter(lengths)
    n = len(tokens)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_len = max(lengths)
    max_entropy = math.log2(max_len + 1) if max_len > 0 else 1.0
    return min(entropy / max(max_entropy, 1e-9), 1.0)


def extract_features(raw_query: str, tokens: list[str], doc_count: int = DOC_COUNT) -> list[float]:
    n = float(doc_count)
    max_idf = math.log(n + 1) if n > 0 else 1.0

    # x[0] — query_len_norm
    query_len_norm = min(len(tokens) / 10.0, 1.0)

    # x[1], x[2] — IDF approximation: df=1 for every query token (they are likely rare).
    # This captures the structural signal (constant avg_idf across queries) and is
    # sufficient to fit the weights for the features that DO vary (x[3], x[4], x[5]).
    if tokens:
        idf_val = math.log((n - 1 + 0.5) / (1 + 0.5) + 1)
        idfs = [idf_val] * len(tokens)
        avg_idf = idf_val
        # Variance is 0 when df is constant — this feature will have near-zero weight after fit.
        idf_variance = 0.0
    else:
        avg_idf = 0.0
        idf_variance = 0.0

    avg_idf_norm = min(avg_idf / max_idf, 1.0)
    max_var = (max_idf / 2.0) ** 2
    idf_variance_norm = min(idf_variance / max(max_var, 1e-6), 1.0)

    # x[3] — has_code_token (snake_case or CamelCase in raw query)
    has_code = "_" in raw_query or any(
        c.isupper() for word in raw_query.split() for c in word[1:]
    )
    has_code_token = 1.0 if has_code else 0.0

    # x[4] — stop_word_ratio on raw whitespace tokens
    raw_tokens = raw_query.split()
    stop_count = sum(1 for w in raw_tokens if w.lower() in STOP_WORDS)
    stop_word_ratio = stop_count / len(raw_tokens) if raw_tokens else 0.0

    # x[5] — Shannon entropy of token lengths
    token_entropy_norm = _token_length_entropy(tokens)

    return [
        query_len_norm,
        avg_idf_norm,
        idf_variance_norm,
        has_code_token,
        stop_word_ratio,
        token_entropy_norm,
    ]


# ---------------------------------------------------------------------------
# Main — fit and export
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Dataset size: {len(DATASET)} queries ({sum(1 for _,_,a in DATASET if a > 0.5)} lexical / {sum(1 for _,_,a in DATASET if a <= 0.5 and a >= 0.4)} hybrid / {sum(1 for _,_,a in DATASET if a < 0.4)} semantic)\n")

    X = np.array([extract_features(r, t) for r, t, _ in DATASET])
    y_continuous = np.array([label for _, _, label in DATASET])
    y = (y_continuous > 0.5).astype(int)  # 1 = BM25-dominant, 0 = vector-dominant

    print(f"Class distribution: {y.sum()} lexical (BM25), {(1-y).sum()} semantic/hybrid")

    # Feature summary
    print("\nFeature means by class:")
    print(f"  {'Feature':<25} {'Lexical mean':>14} {'Semantic mean':>14}")
    for i, name in enumerate(FEATURE_NAMES):
        lex_mean = X[y == 1, i].mean()
        sem_mean = X[y == 0, i].mean()
        print(f"  {name:<25} {lex_mean:>14.4f} {sem_mean:>14.4f}")

    # Fit
    clf = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs")
    clf.fit(X, y)

    # Cross-validation accuracy
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\n5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Full-dataset report
    y_pred = clf.predict(X)
    print("\nClassification report (train set):")
    print(classification_report(y, y_pred, target_names=["semantic", "lexical"]))

    weights = clf.coef_[0]
    bias = float(clf.intercept_[0])

    # Output Rust constants
    print("=" * 60)
    print("// Paste into core/src/scoring/query_features.rs")
    print("// Trained on {} labeled queries (no API calls)".format(len(DATASET)))
    print("const WEIGHTS: [f32; 6] = [")
    descriptions = [
        "query_len_norm    — longer queries → more semantic → lower alpha",
        "avg_idf_norm      — rare terms → BM25 advantage → higher alpha",
        "idf_variance_norm — high variance → mixed → slight semantic",
        "has_code_token    — code token → lexical → higher alpha",
        "stop_word_ratio   — natural language → lower alpha",
        "token_entropy_norm — diverse lengths → semantic → lower alpha",
    ]
    for w, desc in zip(weights, descriptions):
        print(f"    {w:>8.4f}, // x[?] {desc}")
    print("];")
    print(f"const BIAS: f32 = {bias:.4f};")
    print("=" * 60)

    # Sanity check: verify direction of key features
    print("\nSanity checks:")
    w = weights
    checks = [
        (w[3] > 0,  f"has_code_token weight {w[3]:.4f} > 0 (code → higher alpha)"),
        (w[4] < 0,  f"stop_word_ratio weight {w[4]:.4f} < 0 (stop words → lower alpha)"),
        (w[0] < 0,  f"query_len_norm weight {w[0]:.4f} < 0 (longer → lower alpha)"),
    ]
    all_ok = True
    for ok, msg in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {msg}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll sanity checks passed. Safe to update query_features.rs.")
    else:
        print("\nSome checks failed — review dataset balance before updating constants.")


if __name__ == "__main__":
    main()
