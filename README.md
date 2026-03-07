# Nexus

Distributed search engine — AST code indexing, hybrid BM25 + vector scoring, gossip-based PageRank, ε-differential privacy. Written in Rust + TypeScript.

## Architecture

```
nexus/
├── core/        Rust engine  — indexing, scoring, gossip, PageRank
├── gateway/     TypeScript   — HTTP/WebSocket API (Elysia + Bun)
└── tools/       Python scripts — benchmarking (benchmark.py) + model training (train_alpha.py)
```

Nodes communicate over QUIC. State is propagated via gossip (no leader, no consensus).

## Stack

| Layer | Tech |
|-------|------|
| Transport | QUIC (quinn) + ed25519 node identity |
| Index | Inverted index, BP128-compressed posting lists, HNSW vectors |
| Scoring | BM25F multi-field + dense hybrid (BAAI/bge-small-en-v1.5 via fastembed), QPP-adaptive alpha, distributed IDF via HyperLogLog |
| PageRank | Local power iteration + gossip aggregation |
| Privacy | Laplace ε-DP on gossiped HLL sketches |
| Gateway | Bun + Elysia, msgpack, WebSocket streaming, RAG via Claude |

## Run

```bash
# Core node
cargo run -p nexus-core --release

# Gateway
cd gateway && bun run index.ts
```

## Tools

| Script | Purpose |
|--------|---------|
| `tools/train_alpha.py` | Fit BM25/vector fusion weights from 100 labeled queries (no API). Outputs `WEIGHTS`/`BIAS` constants for `query_features.rs`. |
| `tools/benchmark.py` | Benchmark Nexus vs Tantivy on the crates.io corpus (MRR@10, Hits@1, P50/P95/P99 latency, QPS). Two suites: named-entity retrieval (crate name = query) and 78 hand-labeled natural-language queries. |

```bash
# Train QPP weights
python3 tools/train_alpha.py

# Benchmark — Tantivy only (no Nexus node required)
python3 tools/benchmark.py --no-nexus

# Benchmark — Nexus + Tantivy (node running on localhost:3001)
# After indexing, rebuild the vector index first:
curl -X POST http://localhost:3001/rebuild-vector
python3 tools/benchmark.py --skip-fetch --nexus-url http://localhost:3001

# Smaller run for quick iteration
python3 tools/benchmark.py --no-nexus --corpus-size 500 --query-size 50
```

Dependencies: `pip install requests tantivy tqdm`

## Benchmark

crates.io corpus (top 2,000 crates), 200 named-entity queries + 78 hand-labeled natural-language queries.

### Relevance

Two configurations measured: **BM25F only** (no vector index) and **Hybrid** (BM25F + fastembed dense vectors, BAAI/bge-small-en-v1.5 384-dim).

| Suite | Nexus Hybrid | Nexus BM25F | Tantivy |
|-------|-------------|-------------|---------|
| NE MRR@10 | 0.9587 | **0.9714** | 0.9422 |
| NE Hits@1 | 0.9300 | **0.9500** | 0.9100 |
| NL MRR@10 | **0.3796** | 0.3580 | 0.3416 |
| NL Hits@1 | 0.2051 | 0.2564 | 0.2436 |

BM25F (multi-field with name boost ×3) dominates named-entity retrieval. Dense hybrid improves NL MRR@10 by +3.8 pts over Tantivy — semantic embeddings help when query wording differs from indexed text. NL Hits@1 gap is under investigation (QPP alpha calibration).

### Latency (Nexus HTTP loopback, single node, release build)

| Suite | Config | P50 | P95 | P99 |
|-------|--------|-----|-----|-----|
| NE (200 queries) | BM25F only | 0.85 ms | 1.02 ms | 1.12 ms |
| NL (78 queries)  | BM25F only | 0.86 ms | 1.01 ms | 1.20 ms |
| NE (200 queries) | Hybrid     | 5.54 ms | 6.58 ms | 7.13 ms |
| NL (78 queries)  | Hybrid     | 5.88 ms | 6.65 ms | 7.04 ms |

Tantivy runs in-process (no network). Nexus latency includes HTTP loopback + Rust server.
Hybrid overhead (~5 ms) is fastembed ONNX inference on a single CPU core — expected to improve significantly on multi-core hardware or with batched query serving.

## Docs

- [CONSISTENCY.md](CONSISTENCY.md) — consistency model, CRDT proofs, PageRank convergence
