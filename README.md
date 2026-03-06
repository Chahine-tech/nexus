# Nexus

Distributed search engine written in Rust + TypeScript.

## Architecture

```
nexus/
├── core/        Rust engine  — indexing, scoring, gossip, PageRank
└── gateway/     TypeScript   — HTTP/WebSocket API (Elysia + Bun)
```

Nodes communicate over QUIC. State is propagated via gossip (no leader, no consensus).

## Stack

| Layer | Tech |
|-------|------|
| Transport | QUIC (quinn) + ed25519 node identity |
| Index | Inverted index, BP128-compressed posting lists, HNSW vectors |
| Scoring | BM25 + hybrid vector, distributed IDF via HyperLogLog |
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

## Docs

- [CONSISTENCY.md](CONSISTENCY.md) — consistency model, CRDT proofs, PageRank convergence
