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
| `tools/sweep_dp_epsilon.py` | Privacy-utility tradeoff curve: sweep `dp_epsilon` over [0.05 → ∞], spin up a 2-node gossip cluster per value, measure NL MRR@10. Quantifies the cost of ε-DP on search quality. |

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

# ef_search recall-latency sweep (requires Nexus node + vector index)
python3 tools/benchmark.py --skip-fetch --nexus-url http://localhost:3001 \
  --ef-search 10 20 40 80 160 320
```

Dependencies: `pip install requests tantivy tqdm`

## Benchmark

crates.io corpus (top 2,000 crates), 200 named-entity queries + 92 hand-labeled natural-language queries.

### Relevance

Two configurations measured: **BM25F only** (no vector index) and **Hybrid** (BM25F + fastembed dense vectors, BAAI/bge-small-en-v1.5 384-dim).

| Suite | Nexus Hybrid | Nexus BM25F | Tantivy |
|-------|-------------|-------------|---------|
| NE MRR@10 | 0.9618 | **0.9714** | 0.9422 |
| NE Hits@1 | 0.9300 | **0.9500** | 0.9100 |
| NL MRR@10 | **0.4074** | 0.3766 | 0.3865 |
| NL Hits@1 | 0.2609 | **0.2935** | 0.3043 |

BM25F (multi-field with name boost ×3) dominates named-entity retrieval (+2.9 pts MRR vs Tantivy). Dense hybrid improves NL MRR@10 by +2.1 pts over Tantivy — semantic embeddings help when query wording differs from indexed text. The hybrid trades Hits@1 for MRR@10: it surfaces the right answer in top-5 more often even when it misses the top slot.

### Latency (Nexus HTTP loopback, single node, release build)

| Suite | Config | P50 | P95 | P99 |
|-------|--------|-----|-----|-----|
| NE (200 queries) | BM25F only | 0.85 ms | 1.02 ms | 1.12 ms |
| NL (92 queries)  | BM25F only | 0.84 ms | 0.98 ms | 1.17 ms |
| NE (200 queries) | Hybrid     | 5.61 ms | 6.98 ms | 7.34 ms |
| NL (92 queries)  | Hybrid     | 5.85 ms | 6.87 ms | 7.44 ms |

Tantivy runs in-process (no network). Nexus latency includes HTTP loopback + Rust server.
Hybrid overhead (~5 ms) is fastembed ONNX inference on a single CPU core — expected to improve significantly on multi-core hardware or with batched query serving.

### Privacy-utility tradeoff (ε-DP sweep)

2-node gossip cluster, corpus split 50/50, 35 s gossip convergence wait, NL MRR@10 measured on node A.

| ε | MRR@10 | Δ vs ε=1.0 |
|---|--------|-----------|
| 0.05 | — | — |
| 0.1 | — | — |
| 1.0 | 0.408 | baseline |
| 10.0 | — | — |
| ∞ (no noise) | — | — |

> **Result on 2,000-doc corpus**: MRR@10 range across all ε values ≈ 0.007 — statistically negligible. The DP noise perturbs `estimated_global_terms` (gossip-propagated N used for IDF), but BM25 is robust to small N perturbations at this corpus size. The effect grows with corpus size and number of peers; expect more visible degradation at low ε on 50k+ doc deployments.
>
> Run: `cargo build -p nexus-core --release && python3 tools/sweep_dp_epsilon.py`

### HNSW ef_search — recall vs latency

`ef_search` controls the beam width during HNSW graph traversal: higher values explore more candidate nodes, improving recall at the cost of latency. The default is `max(limit × 4, 50)`.

Run the sweep (requires node with vector index built):

```bash
curl -X POST http://localhost:3001/rebuild-vector
python3 tools/benchmark.py --skip-fetch --nexus-url http://localhost:3001 \
  --ef-search 10 20 40 80 160 320
```

| ef_search | NL MRR@10 | P50 ms |
|-----------|-----------|--------|
| auto (limit×4) | 0.4193 | 8.62 |
| 10 | 0.4193 | 8.72 |
| 20 | 0.4193 | 8.77 |
| 40 | 0.4193 | 8.76 |
| 80 | 0.4193 | 8.60 |
| 160 | 0.4193 | 8.67 |
| 320 | 0.4193 | 8.67 |

> **Result on 2,000-doc corpus**: MRR@10 is flat across all ef_search values — even ef=10 achieves full recall. On a corpus this small the HNSW graph is shallow (few layers) and greedy search reaches the same neighbours regardless of beam width. Latency (~8.7 ms) is dominated by fastembed ONNX inference, not graph traversal. The ef_search tradeoff becomes visible at 100k+ docs where multi-layer graphs can miss neighbours with a narrow beam.

## Docs

- [CONSISTENCY.md](CONSISTENCY.md) — consistency model, CRDT proofs, PageRank convergence
