import { createHash } from "node:crypto"
import { registry } from "./NodeRegistry"
import type { NodeId } from "../errors"

export interface QueryPlan {
  shards: Array<{ nodeId: NodeId; url: string; terms: string[] }>
  mergeStrategy: "rrf" | "boolean"
  timeoutMs: number
}

// SHA-256 of a term as a Buffer — used for XOR consistent hashing.
// Note: Rust nodes use blake3 internally. This only affects gateway-side load distribution,
// not correctness — each Rust node handles any QueryRequest it receives.
function termKey(term: string): Buffer {
  return createHash("sha256").update(term).digest()
}

function xorDistance(a: Buffer, b: Buffer): Buffer {
  const len = Math.max(a.length, b.length)
  const result = Buffer.alloc(len)
  for (let i = 0; i < len; i++) {
    result[i] = (a[i] ?? 0) ^ (b[i] ?? 0)
  }
  return result
}

function bufferLt(a: Buffer, b: Buffer): boolean {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if ((a[i] ?? 0) < (b[i] ?? 0)) return true
    if ((a[i] ?? 0) > (b[i] ?? 0)) return false
  }
  return false
}

// Returns the live node whose nodeId has minimum XOR distance to the term's SHA-256 key.
export function termToNode(
  term: string,
  nodes: Array<{ nodeId: NodeId; url: string }>,
): { nodeId: NodeId; url: string } | undefined {
  if (nodes.length === 0) return undefined
  const key = termKey(term)
  return nodes.reduce((best, node) => {
    const dBest = xorDistance(key, Buffer.from(best.nodeId, "hex"))
    const dNode = xorDistance(key, Buffer.from(node.nodeId, "hex"))
    return bufferLt(dNode, dBest) ? node : best
  })
}

export function planQuery(query: string): QueryPlan {
  const terms = query
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 0)

  const nodes = registry.liveNodes()

  if (nodes.length === 0) {
    return { shards: [], mergeStrategy: "rrf", timeoutMs: 150 }
  }

  // Group terms by responsible node using XOR consistent hashing.
  const shardMap = new Map<NodeId, { url: string; terms: string[] }>()

  for (const term of terms) {
    const node = termToNode(term, nodes)
    if (!node) continue
    const existing = shardMap.get(node.nodeId)
    if (existing) {
      existing.terms.push(term)
    } else {
      shardMap.set(node.nodeId, { url: node.url, terms: [term] })
    }
  }

  const shards = [...shardMap.entries()].map(([nodeId, { url, terms }]) => ({
    nodeId,
    url,
    terms,
  }))

  return { shards, mergeStrategy: "rrf", timeoutMs: 150 }
}
