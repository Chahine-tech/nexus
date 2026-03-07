# Nexus

Distributed search engine — AST code indexing, hybrid BM25 + vector scoring, gossip-based PageRank, ε-differential privacy. Written in Rust + TypeScript.

## Architecture

```
nexus/
├── core/        Rust engine  — indexing, scoring, gossip, PageRank
├── gateway/     TypeScript   — HTTP/WebSocket API (Elysia + Bun)
└── tools/       Python scripts — model training (train_alpha.py)
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
| `tools/benchmark.py` | Benchmark Nexus vs Tantivy on the crates.io corpus (MRR@10, Hits@1, QPS). |

```bash
# Train QPP weights
python3 tools/train_alpha.py

# Benchmark (requires Nexus running on localhost:3000, or --no-nexus for Tantivy only)
python3 tools/benchmark.py --skip-download --no-nexus
```

## Docs

- [CONSISTENCY.md](CONSISTENCY.md) — consistency model, CRDT proofs, PageRank convergence
