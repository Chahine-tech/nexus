import { Effect } from "effect"
import { termToNode } from "./QueryPlanner"
import type { NodeRegistry } from "./NodeRegistry"
import type { NodeId } from "../errors"

// Payload sent to a Rust node's /merge-shard endpoint.
interface MergeShardBody {
  term: string
  // base64-encoded msgpack PostingList bytes fetched from the dead node's /export-shard.
  posting_b64: string
}

// Fetches the serialized posting list for `term` from the dead node, then
// pushes it to the successor node via POST /merge-shard.
function rebalanceTerm(
  term: string,
  deadUrl: string,
  successorUrl: string,
): Effect.Effect<void, Error> {
  return Effect.tryPromise({
    try: async () => {
      // 1. Export the posting list bytes from the (still-reachable) dead node.
      //    In practice the node may be down; we attempt and skip on failure.
      const exportRes = await fetch(
        `${deadUrl}/export-shard?term=${encodeURIComponent(term)}`,
        { signal: AbortSignal.timeout(1_000) },
      )
      if (!exportRes.ok) return // node truly unreachable — skip this term

      const bytes = await exportRes.arrayBuffer()
      const posting_b64 = Buffer.from(bytes).toString("base64")

      // 2. Push to the successor node.
      const body: MergeShardBody = { term, posting_b64 }
      await fetch(`${successorUrl}/merge-shard`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(2_000),
      })
    },
    catch: (err) => (err instanceof Error ? err : new Error(String(err))),
  })
}

// Called when `deadNodeId` goes down.
// Finds all terms it owned, picks the new responsible node via XOR hashing,
// and transfers each posting list concurrently.
export function rebalanceNode(
  deadNodeId: NodeId,
  deadUrl: string,
  reg: NodeRegistry,
): Effect.Effect<void> {
  const terms = reg.termsOf(deadNodeId)
  if (terms.length === 0) return Effect.void

  const liveNodes = reg.liveNodes()
  if (liveNodes.length === 0) return Effect.void

  return Effect.forEach(
    terms,
    (term) => {
      const successor = termToNode(term, liveNodes)
      if (!successor) return Effect.void
      return rebalanceTerm(term, deadUrl, successor.url).pipe(
        Effect.tapError((e) =>
          Effect.sync(() =>
            console.warn(`rebalance failed for term "${term}":`, e.message),
          ),
        ),
        // Errors are non-fatal — best-effort rebalancing.
        Effect.ignore,
      )
    },
    { concurrency: 8 },
  )
}
