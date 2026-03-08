import Anthropic from "@anthropic-ai/sdk";
import { Effect } from "effect";
import { Elysia, t } from "elysia";
import type { NodeId } from "./errors";
import {
	DeserializationError,
	NodeDeadError,
	NodeTimeoutError,
} from "./errors";
import type { NodeResult } from "./proto";
import { reciprocalRankFusion } from "./services/MergeEngine";
import { registry } from "./services/NodeRegistry";
import { expandQuery } from "./services/QueryExpander";
import { planQuery } from "./services/QueryPlanner";
import { RAG_PROMPT, ragPipeline } from "./services/RagPipeline";
import { rebalanceNode } from "./services/RebalanceService";

// Register rebalance hook once: when a node dies, transfer its shards.
registry.onNodeDead((nodeId, url) => {
	console.warn(`node ${nodeId} marked dead — starting rebalance`);
	Effect.runPromise(rebalanceNode(nodeId, url, registry)).catch((e) =>
		console.error("rebalance error:", e),
	);
});

// Heartbeat loop: ping all nodes every 5 s, triggers onNodeDead callbacks.
setInterval(() => {
	registry.checkHeartbeats().catch((e) => console.error("heartbeat error:", e));
}, 5_000);

interface RawShardHit {
	doc_id: number;
	score: number;
	url?: string;
}

interface NodeStats {
	doc_count: number;
	vocab_size: number;
	pagerank_ready: boolean;
}

function parseShardResponse(raw: unknown): RawShardHit[] | null {
	if (!Array.isArray(raw)) return null;
	const value: RawShardHit[] = [];
	for (const item of raw) {
		if (typeof item !== "object" || item === null) return null;
		const rec = item as Record<string, unknown>;
		const { doc_id, score, url } = rec;
		if (typeof doc_id !== "number" || typeof score !== "number") return null;
		value.push({ doc_id, score, ...(typeof url === "string" ? { url } : {}) });
	}
	return value;
}

function parseNodeStats(raw: unknown): NodeStats | null {
	if (typeof raw !== "object" || raw === null) return null;
	const rec = raw as Record<string, unknown>;
	const { doc_count, vocab_size, pagerank_ready } = rec;
	if (
		typeof doc_count !== "number" ||
		typeof vocab_size !== "number" ||
		typeof pagerank_ready !== "boolean"
	)
		return null;
	return { doc_count, vocab_size, pagerank_ready };
}

// Fetches one shard as an Effect, yielding typed errors instead of throwing.
function fetchShardEffect(
	url: string,
	terms: string[],
	limit: number,
	nodeId: NodeId,
	timeoutMs: number,
): Effect.Effect<
	NodeResult[],
	NodeTimeoutError | NodeDeadError | DeserializationError
> {
	return Effect.tryPromise({
		try: async () => {
			const q = encodeURIComponent(terms.join(" "));
			const res = await fetch(`${url}/search?q=${q}&limit=${limit}`, {
				signal: AbortSignal.timeout(timeoutMs),
			});
			if (!res.ok) throw new NodeDeadError({ nodeId });
			const body = await res.arrayBuffer();
			const parsed = parseShardResponse(
				JSON.parse(new TextDecoder().decode(body)),
			);
			if (parsed === null) {
				throw new DeserializationError({
					raw: new Uint8Array(body),
					cause: "unexpected shape",
				});
			}
			return parsed.map((r) => ({
				id: String(r.doc_id),
				score: r.score,
				...(r.url ? { url: r.url } : {}),
			}));
		},
		catch: (err) => {
			if (err instanceof NodeDeadError || err instanceof DeserializationError)
				return err;
			// AbortSignal timeout fires as DOMException with name "TimeoutError"
			if (err instanceof Error && err.name === "TimeoutError") {
				return new NodeTimeoutError({ nodeId, terms });
			}
			return new NodeDeadError({ nodeId });
		},
	});
}

export const app = new Elysia()
	.get("/health", () => ({ status: "ok" }))

	.post(
		"/nodes/register",
		({ body }) => {
			registry.registerFromRust(body.nodeId, body.url);
			return new Response(null, { status: 201 });
		},
		{ body: t.Object({ nodeId: t.String(), url: t.String() }) },
	)

	.get(
		"/search",
		async ({ query }) => {
			const q = query.q ?? "";
			const rawLimit = parseInt(query.limit ?? "10", 10);
			const limit = Math.min(Number.isNaN(rawLimit) ? 10 : rawLimit, 100);

			if (!q.trim()) {
				return { results: [] };
			}

			const expanded = await Effect.runPromise(expandQuery(q));
			const plan = planQuery(expanded);

			if (plan.shards.length === 0) {
				return { results: [], error: "no live nodes registered" };
			}

			// Record which terms each node owns (used for rebalancing on death).
			for (const { nodeId, terms } of plan.shards) {
				registry.recordTermOwnership(nodeId, terms);
			}

			// Fan-out to all shards concurrently; collect successes, log failures.
			const program = Effect.forEach(
				plan.shards,
				({ url, terms, nodeId }) =>
					fetchShardEffect(url, terms, limit, nodeId, plan.timeoutMs).pipe(
						Effect.tapError((e) =>
							Effect.sync(() => {
								if (e._tag === "NodeDeadError") registry.markDead(e.nodeId);
								console.warn(`shard ${nodeId} failed:`, e._tag);
							}),
						),
						Effect.option,
					),
				{ concurrency: "unbounded" },
			);

			const options = await Effect.runPromise(program);
			const allNodeResults = options.flatMap((opt) =>
				opt._tag === "Some" ? [opt.value] : [],
			);

			const merged = reciprocalRankFusion(allNodeResults);
			const top = merged.slice(0, limit);
			const answer =
				query.rag === "true"
					? await Effect.runPromise(ragPipeline(q, top))
					: undefined;

			return {
				results: top,
				...(answer ? { answer } : {}),
				...(expanded !== q ? { expanded_query: expanded } : {}),
			};
		},
		{
			query: t.Object({
				q: t.Optional(t.String()),
				limit: t.Optional(t.String()),
				rag: t.Optional(t.String()),
			}),
		},
	)

	.post(
		"/crawl",
		async ({ body }) => {
			const nodes = registry.liveNodes();
			const results = await Promise.allSettled(
				nodes.map(({ url }) =>
					fetch(`${url}/crawl`, {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({ seeds: body.seeds }),
						signal: AbortSignal.timeout(5_000),
					}),
				),
			);
			const triggered = results.filter((r) => r.status === "fulfilled").length;
			return { triggered, total: nodes.length };
		},
		{ body: t.Object({ seeds: t.Array(t.String()) }) },
	)

	.get("/stats", async () => {
		const nodes = registry.liveNodes();
		const results = await Promise.allSettled(
			nodes.map(({ url }) =>
				fetch(`${url}/stats`, { signal: AbortSignal.timeout(1_000) }).then(
					(r) => r.json(),
				),
			),
		);

		let total_docs = 0;
		let total_vocab = 0;
		let pagerank_ready = false;

		for (const r of results) {
			if (r.status === "fulfilled") {
				const stats = parseNodeStats(r.value);
				if (stats !== null) {
					total_docs += stats.doc_count;
					total_vocab += stats.vocab_size;
					pagerank_ready = pagerank_ready || stats.pagerank_ready;
				}
			}
		}

		return {
			total_docs,
			total_vocab,
			pagerank_ready,
			live_nodes: nodes.length,
		};
	})

	.ws("/search/stream", {
		open(ws) {
			ws.send(JSON.stringify({ type: "connected" }));
		},
		async message(ws, raw) {
			const msg =
				typeof raw === "string"
					? JSON.parse(raw)
					: (raw as Record<string, unknown>);
			const q: string = (msg.q as string) ?? "";
			const useRag: boolean = msg.rag === true;
			const limit = Math.min((msg.limit as number) ?? 10, 100);

			if (!q.trim()) {
				ws.send(JSON.stringify({ type: "done" }));
				return;
			}

			const expanded = await Effect.runPromise(expandQuery(q));
			const plan = planQuery(expanded);

			if (plan.shards.length === 0) {
				ws.send(
					JSON.stringify({
						type: "error",
						message: "no live nodes registered",
					}),
				);
				ws.send(JSON.stringify({ type: "done" }));
				return;
			}

			for (const { nodeId, terms } of plan.shards) {
				registry.recordTermOwnership(nodeId, terms);
			}

			const program = Effect.forEach(
				plan.shards,
				({ url, terms, nodeId }) =>
					fetchShardEffect(url, terms, limit, nodeId, plan.timeoutMs).pipe(
						Effect.tapError((e) =>
							Effect.sync(() => {
								if (e._tag === "NodeDeadError") registry.markDead(e.nodeId);
							}),
						),
						Effect.option,
					),
				{ concurrency: "unbounded" },
			);

			const options = await Effect.runPromise(program);
			const allNodeResults = options.flatMap((opt) =>
				opt._tag === "Some" ? [opt.value] : [],
			);

			const merged = reciprocalRankFusion(allNodeResults).slice(0, limit);

			for (const result of merged) {
				ws.send(JSON.stringify({ type: "result", data: result }));
			}

			if (useRag) {
				const apiKey = process.env.ANTHROPIC_API_KEY;
				const snippets = merged
					.slice(0, 5)
					.flatMap((r) => (r.snippet ? [r.snippet] : []));
				if (apiKey && snippets.length > 0) {
					try {
						const client = new Anthropic({ apiKey });
						const stream = client.messages.stream({
							model: "claude-haiku-4-5-20251001",
							max_tokens: 256,
							messages: [{ role: "user", content: RAG_PROMPT(q, snippets) }],
						});
						for await (const event of stream) {
							if (
								event.type === "content_block_delta" &&
								event.delta.type === "text_delta"
							) {
								ws.send(
									JSON.stringify({
										type: "answer_chunk",
										text: event.delta.text,
									}),
								);
							}
						}
					} catch {
						// streaming errors are non-fatal
					}
				}
			}

			ws.send(JSON.stringify({ type: "done" }));
		},
	});
