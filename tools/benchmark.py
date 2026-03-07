"""
benchmark.py — Nexus vs Tantivy on the crates.io corpus.

Usage:
    # Fetch corpus from crates.io API, build indexes, run benchmark
    python3 tools/benchmark.py

    # Use cached corpus JSON if already fetched
    python3 tools/benchmark.py --skip-fetch

    # Tantivy only (no Nexus node required)
    python3 tools/benchmark.py --no-nexus

    # Custom Nexus URL
    python3 tools/benchmark.py --nexus-url http://localhost:3000

Dependencies:
    pip install requests tantivy tqdm

Corpus:
    crates.io REST API (public, no auth). Fetches top N crates by downloads.
    Cached as JSON in /tmp/nexus_bench/crates.json.

Methodology:
    - Index top K crates (name + description + keywords).
    - Queries = names of the top Q most-downloaded crates (Q <= K).
    - Judgment: the crate whose name exactly matches the query = expected result.
    - Metrics: MRR@10, Hits@1, median latency, QPS.

    Named-entity retrieval benchmark — reproducible, engine-agnostic.

Two benchmark suites are run:
  1. Named-entity (NE): query = crate name, expected = that exact crate.
     Good for measuring exact-match precision but insensitive to stemming.
  2. Natural-language (NL): 25 hand-labeled queries in plain English.
     Each query has one expected crate. Sensitive to stemming and recall.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: pip install requests")

try:
    import tantivy
except ImportError:
    raise SystemExit("Missing dependency: pip install tantivy")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        return it

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path("/tmp/nexus_bench")
CORPUS_CACHE = CACHE_DIR / "crates.json"
CRATES_API = "https://crates.io/api/v1/crates"
PAGE_SIZE = 100       # max allowed by crates.io API
CORPUS_SIZE = 2000    # total crates to index
QUERY_SIZE = 200      # top crates used as queries
SEARCH_LIMIT = 10

# ---------------------------------------------------------------------------
# Step 1 — Fetch corpus from crates.io API
# ---------------------------------------------------------------------------


def fetch_corpus(skip: bool = False) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if skip and CORPUS_CACHE.exists():
        print(f"[corpus] Using cached data at {CORPUS_CACHE}")
        with open(CORPUS_CACHE) as f:
            return json.load(f)

    print(f"[corpus] Fetching top {CORPUS_SIZE} crates from crates.io API ...")
    docs: list[dict] = []
    headers = {"User-Agent": "nexus-benchmark/0.1 (github.com/nexus)"}
    page = 1

    with tqdm(total=CORPUS_SIZE, desc="  fetching", unit="crate") as bar:
        while len(docs) < CORPUS_SIZE:
            resp = requests.get(
                CRATES_API,
                params={"sort": "downloads", "per_page": PAGE_SIZE, "page": page},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("crates", [])
            if not data:
                break
            for c in data:
                docs.append({
                    "name": c["name"],
                    "description": c.get("description") or "",
                    "keywords": c.get("keywords") or [],
                    "downloads": c.get("downloads", 0),
                })
            bar.update(len(data))
            page += 1
            time.sleep(0.5)  # respect crates.io rate limit

    docs = docs[:CORPUS_SIZE]
    with open(CORPUS_CACHE, "w") as f:
        json.dump(docs, f)
    print(f"[corpus] Fetched {len(docs):,} crates, cached to {CORPUS_CACHE}")
    return docs


# ---------------------------------------------------------------------------
# Step 2 — Build Tantivy index
# ---------------------------------------------------------------------------


def build_tantivy_index(docs: list[dict]) -> tantivy.Index:
    print(f"[tantivy] Building index over {len(docs):,} crates ...")
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("name", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("body", stored=False, tokenizer_name="en_stem")
    schema = schema_builder.build()

    index = tantivy.Index(schema)
    writer = index.writer(heap_size=64_000_000)

    for doc in tqdm(docs, desc="  indexing", unit="crate"):
        body = " ".join(filter(None, [
            doc["name"],
            doc["description"],
            " ".join(doc["keywords"]),
        ]))
        writer.add_document(tantivy.Document(name=doc["name"], body=body))

    writer.commit()
    index.reload()
    print(f"[tantivy] Index ready")
    return index


def search_tantivy(index: tantivy.Index, query: str, limit: int) -> list[str]:
    searcher = index.searcher()
    try:
        qp = index.parse_query(query, ["name", "body"])
    except Exception:
        return []
    results = searcher.search(qp, limit)
    return [searcher.doc(addr)["name"][0] for _score, addr in results.hits]


# ---------------------------------------------------------------------------
# Step 3 — Nexus index + search
# ---------------------------------------------------------------------------


def fnv1a_32(s: str) -> int:
    """Mirror of the fnv1a_32 function in http.rs — must stay in sync."""
    hash_val = 2166136261
    for byte in s.encode():
        hash_val ^= byte
        hash_val = (hash_val * 16777619) & 0xFFFFFFFF
    return hash_val


def build_nexus_index(docs: list[dict], nexus_url: str) -> None:
    # POST /index accepts {"url": "...", "text": "..."} and indexes directly without crawling.
    # Falls back to the node HTTP port if the gateway URL is provided (gateway proxies /search
    # but not /index — /index is a node-level endpoint on port 3001/3002).
    index_url = nexus_url
    print(f"[nexus] Indexing {len(docs):,} crates via {index_url}/index ...")
    ok = errors = 0
    for doc in tqdm(docs, desc="  indexing", unit="crate"):
        body = " ".join(filter(None, [doc["description"], " ".join(doc["keywords"])]))
        try:
            resp = requests.post(
                f"{index_url}/index",
                json={
                    "url": f"https://crates.io/crates/{doc['name']}",
                    "name": doc["name"],
                    "body": body,
                },
                timeout=5,
            )
            if resp.status_code == 200:
                ok += 1
            else:
                errors += 1
        except requests.RequestException:
            errors += 1
    print(f"[nexus] {ok:,} OK, {errors} errors")


def search_nexus(nexus_url: str, query: str, limit: int, docid_to_name: dict[int, str]) -> list[str]:
    try:
        resp = requests.get(
            f"{nexus_url}/search",
            params={"q": query, "limit": limit},
            timeout=5,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        items = data if isinstance(data, list) else data.get("results", [])
        names = []
        for item in items:
            doc_id = int(item.get("doc_id") or item.get("id") or 0)
            name = docid_to_name.get(doc_id)
            if name:
                names.append(name)
        return names
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Natural-language query set — hand-labeled, stemmer-sensitive
# ---------------------------------------------------------------------------

# Each entry: (query, expected_crate_name)
# Expected crate must be present in the corpus (top 2000 by downloads).
NL_QUERIES: list[tuple[str, str]] = [
    # --- original 25 ---
    ("async runtime tokio",                  "tokio"),
    ("serialize deserialize json",            "serde"),
    ("serialize deserialize",                 "serde_json"),
    ("http client requests",                  "reqwest"),
    ("command line argument parsing",         "clap"),
    ("error handling library",               "anyhow"),
    ("random number generation",             "rand"),
    ("logging structured tracing",           "tracing"),
    ("regular expressions pattern matching", "regex"),
    ("parallel iterators rayon",             "rayon"),
    ("futures async combinators",            "futures"),
    ("web framework actix",                  "actix-web"),
    ("hash map concurrent",                  "dashmap"),
    ("datetime time parsing",                "chrono"),
    ("uuid generation unique identifier",    "uuid"),
    ("compression gzip zlib",                "flate2"),
    ("base64 encoding decoding",             "base64"),
    ("cryptography hashing sha256",          "sha2"),
    ("url parsing",                          "url"),
    ("toml configuration parsing",           "toml"),
    ("integer arbitrary precision",          "num"),
    ("terminal color output",                "colored"),
    ("image processing",                     "image"),
    ("csv parsing reading",                  "csv"),
    ("database sqlite",                      "rusqlite"),
    # --- networking / protocols ---
    ("websocket client server",              "tokio-tungstenite"),
    ("grpc protocol buffers",               "tonic"),
    ("tls certificate rustls",               "rustls"),
    ("dns resolution async",                 "hickory-resolver"),
    ("http server framework",                "axum"),
    ("mime type detection",                  "mime"),
    ("network protocol encoding",            "bytes"),
    # --- data formats ---
    ("yaml configuration file",             "serde_yaml"),
    ("xml parsing serialization",           "quick-xml"),
    ("json schema validation",              "jsonschema"),
    ("messagepack binary serialization",    "rmp-serde"),
    ("protobuf encoding decoding",          "prost"),
    ("arrow columnar format",               "arrow"),
    # --- async / concurrency ---
    ("channel message passing async",       "tokio"),
    ("actor model concurrency",             "actix"),
    ("async stream processing",             "futures"),
    ("mutex read write lock async",         "tokio"),
    ("thread pool work stealing",           "rayon"),
    ("semaphore rate limiting",             "tokio"),
    # --- CLI / terminal ---
    ("progress bar terminal",               "indicatif"),
    ("terminal table formatting",           "comfy-table"),
    ("interactive prompt readline",         "rustyline"),
    ("ansi color terminal styling",         "colored"),
    ("argument parser derive macros",       "clap"),
    # --- cryptography / security ---
    ("hmac message authentication",         "hmac"),
    ("aes encryption symmetric",            "aes"),
    ("rsa public key cryptography",         "rsa"),
    ("password hashing bcrypt argon",       "argon2"),
    ("random secure bytes generation",      "rand"),
    ("certificate tls x509",               "rcgen"),
    # --- database / storage ---
    ("orm query builder diesel",            "diesel"),
    ("postgres async driver",               "tokio-postgres"),
    ("key value embedded store",            "rocksdb"),
    ("redis client async",                  "redis"),
    ("connection pool database",            "bb8"),
    ("migration schema database",           "diesel"),
    # --- text / parsing ---
    ("string diff patch",                   "similar"),
    ("unicode normalization",               "unicode-normalization"),
    ("markdown html rendering",             "pulldown-cmark"),
    ("template engine html",               "tera"),
    ("natural language tokenizer",          "rust-stemmers"),
    ("fuzzy string matching",               "fuzzy-matcher"),
    # --- system / OS ---
    ("file system watch events",            "notify"),
    ("cross platform path handling",        "dunce"),
    ("process spawn command execution",     "tokio"),
    ("environment variables config",        "dotenvy"),
    ("memory mapped files",                 "memmap2"),
    ("signal handling unix",               "signal-hook"),
    # --- testing / dev tools ---
    ("property based testing",              "proptest"),
    ("mock http server testing",            "wiremock"),
    ("fake test data generation",           "rand"),
    ("snapshot testing assertions",        "insta"),
    ("benchmarking performance",            "criterion"),
]

# ---------------------------------------------------------------------------
# Step 4 — Metrics
# ---------------------------------------------------------------------------


def reciprocal_rank(results: list[str], expected: str) -> float:
    for i, name in enumerate(results, start=1):
        if name.lower() == expected.lower():
            return 1.0 / i
    return 0.0


def run_nl_benchmark(
    tantivy_index: tantivy.Index,
    nexus_url: Optional[str],
    docid_to_name: Optional[dict[int, str]] = None,
) -> dict:
    """Run the natural-language query set. Each query has one expected crate."""
    print(f"\n[nl-benchmark] Running {len(NL_QUERIES)} natural-language queries ...")

    tv_rr, tv_lat = [], []
    nx_rr, nx_lat = [], []

    for query, expected in tqdm(NL_QUERIES, desc="  querying", unit="query"):
        t0 = time.perf_counter()
        tv_results = search_tantivy(tantivy_index, query, SEARCH_LIMIT)
        tv_lat.append(time.perf_counter() - t0)
        tv_rr.append(reciprocal_rank(tv_results, expected))

        if nexus_url:
            t0 = time.perf_counter()
            nx_results = search_nexus(nexus_url, query, SEARCH_LIMIT, docid_to_name or {})
            nx_lat.append(time.perf_counter() - t0)
            nx_rr.append(reciprocal_rank(nx_results, expected))

    def summarize(rr: list[float], lat: list[float]) -> dict:
        n = len(rr)
        lat_ms = [x * 1000 for x in lat]
        lat_sorted = sorted(lat_ms)
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = (len(data) - 1) * p / 100
            lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
            return data[lo] + (data[hi] - data[lo]) * (idx - lo)
        return {
            "mrr@10": sum(rr) / n if n else 0.0,
            "hits@1": sum(1 for r in rr if r == 1.0) / n if n else 0.0,
            "p50_ms": percentile(lat_sorted, 50),
            "p95_ms": percentile(lat_sorted, 95),
            "p99_ms": percentile(lat_sorted, 99),
            "qps": n / sum(lat) if sum(lat) > 0 else 0.0,
        }

    out = {"tantivy": summarize(tv_rr, tv_lat)}
    if nexus_url:
        out["nexus"] = summarize(nx_rr, nx_lat)
    return out


def run_benchmark(
    queries: list[dict],
    tantivy_index: tantivy.Index,
    nexus_url: Optional[str],
    docid_to_name: Optional[dict[int, str]] = None,
) -> dict:
    print(f"\n[benchmark] Running {len(queries)} queries ...")

    tv_rr, tv_lat = [], []
    nx_rr, nx_lat = [], []

    for q in tqdm(queries, desc="  querying", unit="query"):
        name = q["name"]

        t0 = time.perf_counter()
        tv_results = search_tantivy(tantivy_index, name, SEARCH_LIMIT)
        tv_lat.append(time.perf_counter() - t0)
        tv_rr.append(reciprocal_rank(tv_results, name))

        if nexus_url:
            t0 = time.perf_counter()
            nx_results = search_nexus(nexus_url, name, SEARCH_LIMIT, docid_to_name or {})
            nx_lat.append(time.perf_counter() - t0)
            nx_rr.append(reciprocal_rank(nx_results, name))

    def summarize(rr: list[float], lat: list[float]) -> dict:
        n = len(rr)
        lat_ms = [x * 1000 for x in lat]
        lat_sorted = sorted(lat_ms)
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = (len(data) - 1) * p / 100
            lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
            return data[lo] + (data[hi] - data[lo]) * (idx - lo)
        return {
            "mrr@10": sum(rr) / n if n else 0.0,
            "hits@1": sum(1 for r in rr if r == 1.0) / n if n else 0.0,
            "p50_ms": percentile(lat_sorted, 50),
            "p95_ms": percentile(lat_sorted, 95),
            "p99_ms": percentile(lat_sorted, 99),
            "qps": n / sum(lat) if sum(lat) > 0 else 0.0,
        }

    out = {"tantivy": summarize(tv_rr, tv_lat)}
    if nexus_url:
        out["nexus"] = summarize(nx_rr, nx_lat)
    return out


# ---------------------------------------------------------------------------
# Step 5 — Report
# ---------------------------------------------------------------------------


def print_report(metrics: dict, n_queries: int, n_docs: int, title: str = "crates.io benchmark") -> None:
    has_nexus = "nexus" in metrics
    nx = metrics.get("nexus", {})
    tv = metrics["tantivy"]

    print("\n" + "=" * 58)
    print(f"  Nexus vs Tantivy — {title}")
    print(f"  Corpus: {n_docs:,} crates  |  Queries: {n_queries}")
    print("=" * 58)
    header = f"  {'Metric':<24} {'Tantivy':>10}"
    if has_nexus:
        header += f" {'Nexus':>10} {'Delta':>8}"
    print(header)
    print(f"  {'-' * (52 if has_nexus else 36)}")

    rows = [
        ("mrr@10",  "MRR@10",      "{:.4f}"),
        ("hits@1",  "Hits@1",      "{:.4f}"),
        ("p50_ms",  "P50 lat ms",  "{:.2f}"),
        ("p95_ms",  "P95 lat ms",  "{:.2f}"),
        ("p99_ms",  "P99 lat ms",  "{:.2f}"),
        ("qps",     "QPS",         "{:.1f}"),
    ]
    for key, label, fmt in rows:
        tv_val = tv.get(key, float("nan"))
        tv_str = fmt.format(tv_val)
        line = f"  {label:<24} {tv_str:>10}"
        if has_nexus:
            nx_val = nx.get(key, float("nan"))
            nx_str = fmt.format(nx_val)
            delta = nx_val - tv_val
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{fmt.format(delta)}"
            line += f" {nx_str:>10} {delta_str:>8}"
        print(line)

    print("=" * 58)
    if has_nexus:
        mrr_delta = nx.get("mrr@10", 0) - tv.get("mrr@10", 0)
        winner = "Nexus" if mrr_delta > 0 else "Tantivy"
        print(f"  MRR@10 winner: {winner} (delta {mrr_delta:+.4f})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus vs Tantivy — crates.io benchmark")
    parser.add_argument("--skip-fetch", action="store_true", help="Use cached corpus JSON")
    parser.add_argument("--nexus-url", default="http://localhost:3000")
    parser.add_argument("--no-nexus", action="store_true", help="Tantivy only")
    parser.add_argument("--corpus-size", type=int, default=CORPUS_SIZE)
    parser.add_argument("--query-size", type=int, default=QUERY_SIZE)
    args = parser.parse_args()

    nexus_url = None if args.no_nexus else args.nexus_url

    if nexus_url:
        try:
            r = requests.get(f"{nexus_url}/health", timeout=3)
            if r.status_code != 200:
                print(f"[nexus] /health returned {r.status_code}, disabling Nexus")
                nexus_url = None
        except requests.RequestException:
            print(f"[nexus] Cannot reach {nexus_url}, disabling. Use --no-nexus to suppress.")
            nexus_url = None

    docs = fetch_corpus(skip=args.skip_fetch)
    docs = docs[:args.corpus_size]
    queries = docs[:args.query_size]

    tantivy_index = build_tantivy_index(docs)

    docid_to_name: dict[int, str] = {}
    if nexus_url:
        build_nexus_index(docs, nexus_url)
        # Build reverse mapping: FNV1a(url) → crate name (mirrors http.rs index_handler)
        for doc in docs:
            url = f"https://crates.io/crates/{doc['name']}"
            docid_to_name[fnv1a_32(url)] = doc["name"]

    metrics = run_benchmark(queries, tantivy_index, nexus_url, docid_to_name)
    print_report(metrics, n_queries=len(queries), n_docs=len(docs), title="Named-entity retrieval")

    nl_metrics = run_nl_benchmark(tantivy_index, nexus_url, docid_to_name)
    print_report(nl_metrics, n_queries=len(NL_QUERIES), n_docs=len(docs), title="Natural-language queries")


if __name__ == "__main__":
    main()
