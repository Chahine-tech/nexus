# Nexus — Consistency Model

## Overview

Nexus is a distributed search engine. Each node maintains a local index and
exchanges state with peers via gossip over QUIC. This document formally defines
the consistency guarantees of each subsystem and proves convergence where
applicable.

---

## Consistency classes used in Nexus

```
Strong consistency   (linearizable)
         │
         │  ← NOT provided by Nexus (no consensus, no leader)
         │
 Sequential consistency
         │
         │  ← NOT provided (no global ordering)
         │
  Causal consistency
         │
         │  ← Provided for HLL sketches (CRDT merge is causal-safe)
         │
  Eventual consistency  ← PRIMARY MODEL for gossip state
         │
   Monotonic reads    ← Provided for HLL (register values never decrease)
```

Nexus provides **eventual consistency** for all gossiped state, with stronger
**monotonic read** guarantees for HyperLogLog sketches.

---

## 1. Doc-count heartbeats — Last-Write-Wins (LWW)

### State

```
GossipState {
    node_id:   NodeId,
    doc_count: u64,
    timestamp: u64,   // Unix epoch seconds
}
```

### Merge rule

```
merge(a, b) = if a.timestamp >= b.timestamp { a } else { b }
```

### Proof of eventual consistency

Let `S_i(t)` be the state stored for node `i` at local time `t`.

**Termination:** The network is assumed eventually connected (standard gossip
assumption). Each node broadcasts periodically, so every state update reaches
every peer in finite time with probability 1.

**Convergence:** The merge function is a join-semilattice on `(timestamp, state)`
ordered by timestamp. For any two states `a`, `b` with `a.timestamp != b.timestamp`,
`merge(a, b)` is deterministic and equals the state with the higher timestamp.
After all messages from a broadcast round are delivered, all nodes hold the
same value for `S_i`.

**Caveat:** Clock skew can cause a stale update with a high timestamp to win
over a fresh update with a low timestamp. Nexus accepts this trade-off; it is
standard for LWW registers in distributed systems (Cassandra, Riak).

---

## 2. HyperLogLog IDF sketches — CRDT (join-semilattice)

### State

```
HyperLogLog {
    registers: [u8; 64],   // each register = max rho seen
}
```

### Merge rule

```
merge(a, b)[i] = max(a[i], b[i])   for all i in 0..64
```

### Lattice structure

Define the partial order `a ≤ b` iff `∀i, a[i] ≤ b[i]`.

The merge function is the **least upper bound** (join) in this lattice:

- **Commutativity:** `merge(a,b) = merge(b,a)` — max is commutative.
- **Associativity:** `merge(merge(a,b),c) = merge(a,merge(b,c))` — max is associative.
- **Idempotency:** `merge(a,a) = a` — max of identical values is the same value.

This makes `HyperLogLog` a **state-based CRDT (CvRDT)**, which guarantees:

> Given an eventually connected network where every node eventually receives
> every other node's state, all replicas converge to the same value.

### Monotonic read guarantee

Because `merge` only ever increases register values, the estimated cardinality
`estimate(merge(local, peer))` is monotonically non-decreasing over time as
peers are merged in. A query for global term count never returns a value lower
than a previous query (on the same node).

```
                    Node A                Node B
                 ┌──────────┐          ┌──────────┐
  add("rust") ──►│ reg[42]=3│          │ reg[42]=5│◄── add("rust") x more
                 └────┬─────┘          └────┬─────┘
                      │   broadcast_idf     │
                      │◄────────────────────┘
                      │
                 merge: reg[42] = max(3,5) = 5
                      │
                 estimate() ≥ previous estimate()   ✓ monotonic
```

### Privacy note

Before broadcast, the sketch is perturbed with Laplace(0, 1/ε) noise per
register (ε-differential privacy). The merge rule remains valid on noisy
sketches: max of two noisy values is still a consistent (if noisy) upper bound.
The local sketch is never modified — only the wire copy carries noise.

---

## 3. PageRank — Hybrid model (LWW per node + additive read)

PageRank uses a different model from HLL because it is not a CRDT: adding
partial scores from multiple nodes is inherently non-idempotent.

### Write: LWW per node

```
GossipPagerank {
    node_id:        NodeId,
    partial_scores: HashMap<u32, f32>,
    timestamp:      u64,
}

merge(existing, incoming):
    if incoming.timestamp > existing.timestamp:
        replace existing with incoming
    else:
        discard incoming
```

Each node owns its own `partial_scores` entry. An incoming message replaces
the stored entry only if it is strictly newer. This prevents double-counting
when a broadcast is replicated.

### Read: additive aggregation

```
global_pagerank(doc_id) =
    sum(node.partial_scores[doc_id] for all nodes)
    ─────────────────────────────────────────────
    sum(all scores across all nodes)
```

This is computed at read time, not at write time — so repeated broadcasts of
the same `partial_scores` are idempotent (LWW ensures we store at most one
snapshot per node).

```
  Node A (local)        Node B (peer)         Node C (peer)
  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
  │ doc1: 0.4   │       │ doc1: 0.2   │       │ doc1: 0.1   │
  │ doc2: 0.6   │       │ doc3: 0.8   │       │ doc2: 0.3   │
  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                               │
                    global_pagerank(doc1)
                    = (0.4 + 0.2 + 0.1) / (0.4+0.6+0.2+0.8+0.1+0.3)
                    = 0.7 / 2.4
                    ≈ 0.29
```

### Convergence condition

Given N nodes, all broadcasting their `partial_scores` at interval T:

- After at most `N * (max_network_delay / T)` rounds, every node has the latest
  snapshot from every other node.
- The global PageRank value then stabilizes until any node re-runs `iterate()`.

This is **eventual consistency** with a bounded staleness of approximately
`max_network_delay` seconds.

---

## 4. Local PageRank power iteration — convergence proof

`LocalPageRank::iterate()` runs the standard power iteration:

```
r_0[v] = 1/N   for all v

r_{t+1}[v] = (1 - d) / N  +  d * Σ_{u→v}  r_t[u] / out_degree(u)
```

where `d = 0.85` (damping factor), `N` = number of nodes.

### Proof of convergence

The iteration is a multiplication by the **Google matrix** `G`:

```
G = d * M  +  (1-d)/N * E
```

where `M` is the column-stochastic transition matrix and `E` is the all-ones
matrix divided by N.

**G is a stochastic matrix** (columns sum to 1, all entries in [0,1]).

By the **Perron-Frobenius theorem**, since `G` is a positive stochastic matrix
(all entries > 0 due to the `(1-d)/N` teleportation term):

1. G has a unique dominant eigenvalue λ₁ = 1.
2. The corresponding eigenvector (the PageRank vector) is unique and positive.
3. Power iteration converges to this eigenvector at rate `|λ₂/λ₁|^t`,
   where `λ₂ < 1` is the second-largest eigenvalue.

The convergence rate is bounded by `d^t` (the damping factor raised to the
iteration count), since the second eigenvalue of G satisfies `|λ₂| ≤ d = 0.85`.

**In practice:** with ε = 1e-6 and d = 0.85, convergence occurs in
`log(ε) / log(d) ≈ 82` iterations in the worst case. Nexus caps at 100.

---

## 5. Global consistency diagram

```
                          ┌─────────────────────────────────┐
                          │           NEXUS NODE             │
                          │                                  │
  index_document()        │  InvertedIndex  (local, mutable) │
  ──────────────────────► │  PostingList    (BP128 on disk)  │
                          │                                  │
                          │  LocalPageRank  (local, power    │
                          │                  iteration)      │
                          └──────────┬──────────────────────┘
                                     │ gossip (QUIC/UDP)
                          ┌──────────▼──────────────────────┐
                          │         GossipEngine             │
                          │                                  │
                          │  GossipState   ── LWW            │
                          │  HyperLogLog   ── CRDT (max)     │
                          │  GossipPagerank── LWW per node   │
                          └──────────┬──────────────────────┘
                                     │ merge on receive
                          ┌──────────▼──────────────────────┐
                          │         Query path               │
                          │                                  │
                          │  BM25 score     (local)          │
                          │  IDF estimate   (HLL, eventual)  │
                          │  PageRank score (aggregated,     │
                          │                  eventual)       │
                          └─────────────────────────────────┘
```

---

## 6. Consistency guarantees summary

| Subsystem            | Model               | Monotonic | Idempotent | Bounded staleness |
|----------------------|---------------------|-----------|------------|-------------------|
| Doc-count heartbeat  | LWW (timestamp)     | No        | Yes        | ~gossip interval  |
| HyperLogLog IDF      | CRDT (join, max)    | Yes       | Yes        | ~gossip interval  |
| PageRank (per node)  | LWW (timestamp)     | No        | Yes        | ~gossip interval  |
| PageRank (global)    | Additive read       | No        | Yes (LWW)  | ~gossip interval  |
| Local index          | Single-writer       | N/A       | N/A        | 0 (local)         |

**Nexus does not provide:**
- Linearizability (no global lock, no Paxos/Raft)
- Sequential consistency (no global ordering)
- Read-your-writes across nodes (a write on node A is not immediately visible on node B)

**Nexus provides:**
- Eventual consistency for all gossiped state
- Monotonic reads for HyperLogLog cardinality estimates
- Idempotent merges for all gossip messages
- Deterministic conflict resolution (no split-brain ambiguity)
