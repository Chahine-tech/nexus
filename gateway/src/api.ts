import { Elysia, t } from "elysia"
import { registry } from "./services/NodeRegistry"
import { planQuery } from "./services/QueryPlanner"
import { reciprocalRankFusion } from "./services/MergeEngine"
import type { NodeId } from "./errors"
import type { NodeResult } from "./proto"

interface RawShardHit {
  doc_id: number
  score: number
}

type ParseResult<T> = { ok: true; value: T } | { ok: false }

function parseShardResponse(raw: unknown): ParseResult<RawShardHit[]> {
  if (!Array.isArray(raw)) return { ok: false }
  const value: RawShardHit[] = []
  for (const item of raw) {
    if (typeof item !== "object" || item === null) return { ok: false }
    const rec = item as Record<string, unknown>
    const { doc_id, score } = rec
    if (typeof doc_id !== "number" || typeof score !== "number") return { ok: false }
    value.push({ doc_id, score })
  }
  return { ok: true, value }
}

async function fetchShard(
  url: string,
  terms: string[],
  limit: number,
  nodeId: NodeId,
  timeoutMs: number,
): Promise<NodeResult[]> {
  const q = encodeURIComponent(terms.join(" "))
  const res = await fetch(`${url}/search?q=${q}&limit=${limit}`, {
    signal: AbortSignal.timeout(timeoutMs),
  })
  if (!res.ok) {
    registry.markDead(nodeId)
    return []
  }
  const parsed = parseShardResponse(await res.json())
  if (!parsed.ok) {
    console.warn("unexpected response shape from shard", url)
    return []
  }
  return parsed.value.map((r) => ({ id: String(r.doc_id), score: r.score }))
}

export const app = new Elysia()
  .get("/health", () => ({ status: "ok" }))

  .post(
    "/nodes/register",
    ({ body }) => {
      registry.registerFromRust(body.nodeId, body.url)
      return new Response(null, { status: 201 })
    },
    { body: t.Object({ nodeId: t.String(), url: t.String() }) },
  )

  .get(
    "/search",
    async ({ query }) => {
      const q = query.q ?? ""
      const rawLimit = parseInt(query.limit ?? "10", 10)
      const limit = Math.min(Number.isNaN(rawLimit) ? 10 : rawLimit, 100)

      if (!q.trim()) {
        return { results: [] }
      }

      const plan = planQuery(q)

      if (plan.shards.length === 0) {
        return { results: [], error: "no live nodes registered" }
      }

      const settled = await Promise.allSettled(
        plan.shards.map(({ url, terms, nodeId }) =>
          fetchShard(url, terms, limit, nodeId, plan.timeoutMs),
        ),
      )

      const allNodeResults: NodeResult[][] = []
      for (const result of settled) {
        if (result.status === "fulfilled") {
          allNodeResults.push(result.value)
        } else {
          console.warn("shard failed:", result.reason)
        }
      }

      const merged = reciprocalRankFusion(allNodeResults)
      return { results: merged.slice(0, limit) }
    },
    { query: t.Object({ q: t.Optional(t.String()), limit: t.Optional(t.String()) }) },
  )
