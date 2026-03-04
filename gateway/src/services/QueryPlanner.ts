import { registry } from "./NodeRegistry"
import type { NodeId } from "../errors"

export interface QueryPlan {
  shards: Array<{ nodeId: NodeId; terms: string[] }>
  mergeStrategy: "rrf" | "boolean"
  timeoutMs: number
}

export function planQuery(query: string): QueryPlan {
  const terms = query.trim().toLowerCase().split(/\s+/)
  const nodes = registry.locate(terms)

  // TODO: consistent hashing per term via DHT (week 3)
  // For now, broadcast to all nodes
  const shards = nodes.map((nodeId) => ({ nodeId, terms }))

  return {
    shards,
    mergeStrategy: "rrf",
    timeoutMs: 150,
  }
}
