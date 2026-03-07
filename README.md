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
| Scoring | BM25 + hybrid vector, QPP-adaptive alpha, distributed IDF via HyperLogLog |
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
| `tools/benchmark.py` | Benchmark Nexus vs Tantivy on the crates.io corpus (MRR@10, Hits@1, QPS). Two suites: named-entity retrieval (crate name = query) and 25 hand-labeled natural-language queries. |

```bash
# Train QPP weights
python3 tools/train_alpha.py

# Benchmark — Tantivy only (no Nexus node required)
python3 tools/benchmark.py --no-nexus

# Benchmark — Nexus + Tantivy (Nexus must be running on localhost:3000)
python3 tools/benchmark.py --skip-fetch

# Smaller run for quick iteration
python3 tools/benchmark.py --no-nexus --corpus-size 500 --query-size 50
```

Dependencies: `pip install requests tantivy tqdm`

## Benchmark

crates.io corpus (top 2,000 crates), 200 named-entity queries + 78 hand-labeled natural-language queries.

| Suite | Nexus | Tantivy | Delta |
|-------|-------|---------|-------|
| NE MRR@10 | **0.9714** | 0.9422 | **+0.0292** |
| NE Hits@1 | **0.9500** | 0.9100 | **+0.0400** |
| NL MRR@10 | 0.3496 | 0.3480 | +0.0016 |
| NL Hits@1 | 0.1923 | 0.2564 | -0.0641 |

Nexus wins on named-entity retrieval (+3% MRR@10) via BM25F with field boosts (name=3.0, body=1.0)
and Snowball stemming. NL results are statistically equivalent (78 queries).

## Docs

- [CONSISTENCY.md](CONSISTENCY.md) — consistency model, CRDT proofs, PageRank convergence
