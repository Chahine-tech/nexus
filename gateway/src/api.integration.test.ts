/**
 * Integration tests — spin up a mock Rust node (Bun HTTP server) and exercise
 * the full gateway request path: registration, search fan-out, stats, rebalancing.
 */
import {
	afterAll,
	beforeAll,
	beforeEach,
	describe,
	expect,
	test,
} from "bun:test";
import { app } from "./api";
import { registry } from "./services/NodeRegistry";

// Prevent any accidental Anthropic API calls during tests.
delete process.env.ANTHROPIC_API_KEY;

// Typed JSON helper — avoids 'unknown' casts throughout tests.
async function asJson<T>(res: Response): Promise<T> {
	return res.json() as Promise<T>;
}
interface SearchBody {
	results: Array<{ id: string; score: number }>;
	error?: string;
}
interface StatsBody {
	total_docs: number;
	total_vocab: number;
	pagerank_ready: boolean;
	live_nodes: number;
}

// ---------------------------------------------------------------------------
// Mock node server — simulates a Rust nexus node's HTTP API.
// ---------------------------------------------------------------------------

interface MockNode {
	nodeId: string;
	url: string;
	docs: Map<string, number>; // term → doc_id
	server: ReturnType<typeof Bun.serve>;
	port: number;
}

function startMockNode(port: number, nodeId: string): MockNode {
	const docs = new Map<string, number>();

	const server = Bun.serve({
		port,
		fetch(req) {
			const url = new URL(req.url);

			if (url.pathname === "/health") {
				return Response.json({ status: "ok" });
			}

			if (url.pathname === "/search" || url.pathname === "/search/local") {
				const q = url.searchParams.get("q") ?? "";
				const limit = parseInt(url.searchParams.get("limit") ?? "10", 10);
				const terms = q.split(" ").filter(Boolean);
				const results: Array<{ doc_id: number; score: number }> = [];
				for (const term of terms) {
					const docId = docs.get(term);
					if (docId !== undefined) results.push({ doc_id: docId, score: 1.0 });
				}
				return Response.json(results.slice(0, limit));
			}

			if (url.pathname === "/stats") {
				return Response.json({
					doc_count: docs.size,
					vocab_size: docs.size,
					pagerank_ready: false,
				});
			}

			if (url.pathname === "/merge-shard" && req.method === "POST") {
				// Accept any merge — just return 204.
				return new Response(null, { status: 204 });
			}

			if (url.pathname === "/export-shard") {
				const term = url.searchParams.get("term") ?? "";
				if (!docs.has(term)) return new Response(null, { status: 404 });
				// Return minimal valid msgpack bytes for a PostingList (empty payload).
				return new Response(new Uint8Array([0x90]), {
					headers: { "Content-Type": "application/octet-stream" },
				});
			}

			return new Response("not found", { status: 404 });
		},
	});

	return { nodeId, url: `http://localhost:${port}`, docs, server, port };
}

// ---------------------------------------------------------------------------
// Test setup
// ---------------------------------------------------------------------------

let nodeA: MockNode;
let nodeB: MockNode;

beforeAll(() => {
	nodeA = startMockNode(19001, "aa".repeat(32));
	nodeB = startMockNode(19002, "bb".repeat(32));
});

afterAll(() => {
	nodeA.server.stop();
	nodeB.server.stop();
});

beforeEach(() => {
	// Reset registry and term ownership between tests.
	const r = registry as unknown as {
		nodes: Map<unknown, unknown>;
		termOwnership: Map<unknown, unknown>;
	};
	r.nodes.clear();
	r.termOwnership.clear();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("mock node sanity", () => {
	test("mock /health returns ok", async () => {
		const res = await fetch(`${nodeA.url}/health`);
		expect(res.ok).toBe(true);
		expect(await res.json()).toMatchObject({ status: "ok" });
	});
});

describe("POST /nodes/register", () => {
	test("registers a node and it appears in live nodes", async () => {
		const res = await app.handle(
			new Request("http://localhost/nodes/register", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ nodeId: nodeA.nodeId, url: nodeA.url }),
			}),
		);
		expect(res.status).toBe(201);
		expect(registry.liveNodes()).toHaveLength(1);
		expect(registry.liveNodes()[0]?.nodeId).toBe(nodeA.nodeId);
	});
});

describe("GET /search", () => {
	test("returns empty results when no nodes registered", async () => {
		const res = await app.handle(new Request("http://localhost/search?q=rust"));
		expect(res.status).toBe(200);
		const body = await res.json();
		expect(body).toMatchObject({ error: "no live nodes registered" });
	});

	test("returns empty results for empty query", async () => {
		await app.handle(
			new Request("http://localhost/nodes/register", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ nodeId: nodeA.nodeId, url: nodeA.url }),
			}),
		);
		const res = await app.handle(new Request("http://localhost/search?q="));
		expect(res.status).toBe(200);
		const emptyBody = await asJson<SearchBody>(res);
		expect(emptyBody.results).toHaveLength(0);
	});

	test("fan-out finds doc indexed on mock node", async () => {
		// Pre-seed the mock node's in-memory index.
		nodeA.docs.set("rust", 42);

		await app.handle(
			new Request("http://localhost/nodes/register", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ nodeId: nodeA.nodeId, url: nodeA.url }),
			}),
		);

		const res = await app.handle(
			new Request("http://localhost/search?q=rust&limit=5"),
		);
		expect(res.status).toBe(200);
		const body = await asJson<SearchBody>(res);
		// The mock node returns doc_id=42 for "rust" — gateway should relay it.
		expect(body.results.length).toBeGreaterThan(0);
		expect(body.results[0]?.id).toBe("42");

		nodeA.docs.delete("rust");
	});

	test("fan-out across two nodes merges results via RRF", async () => {
		// blake3("python") XOR-closest to nodeA (aaaa...).
		// blake3("rust")   XOR-closest to nodeB (bbbb...).
		// Each shard receives only its term, so both docs must appear in the merged result.
		nodeA.docs.set("python", 1);
		nodeB.docs.set("rust", 2);

		for (const node of [nodeA, nodeB]) {
			await app.handle(
				new Request("http://localhost/nodes/register", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ nodeId: node.nodeId, url: node.url }),
				}),
			);
		}

		const res = await app.handle(
			new Request("http://localhost/search?q=python+rust&limit=10"),
		);
		expect(res.status).toBe(200);
		const body = await asJson<SearchBody>(res);
		// Both docs should appear after RRF merge.
		const ids = body.results.map((r) => r.id);
		expect(ids).toContain("1");
		expect(ids).toContain("2");

		nodeA.docs.delete("python");
		nodeB.docs.delete("rust");
	});

	test("dead node is skipped gracefully — other shards still return results", async () => {
		// blake3("python") routes to nodeA (aaaa...) — nodeA is live, dead node is cc..cc.
		// blake3("rust") routes to nodeB (bbbb...) which is not registered here.
		// We query only "python" so nodeA handles it; the dead node owns no terms in this query.
		// To guarantee nodeA is chosen over the dead node (cc..cc), we query "python".
		nodeA.docs.set("python", 10);

		// Register nodeA (live) and a dead node (port nobody listens on).
		await app.handle(
			new Request("http://localhost/nodes/register", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ nodeId: nodeA.nodeId, url: nodeA.url }),
			}),
		);
		await app.handle(
			new Request("http://localhost/nodes/register", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					nodeId: "cc".repeat(32),
					url: "http://localhost:19099",
				}),
			}),
		);

		const res = await app.handle(
			new Request("http://localhost/search?q=python&limit=5"),
		);
		expect(res.status).toBe(200);
		const body = await asJson<SearchBody>(res);
		// nodeA responds with doc 10; dead node owns no terms in this query.
		const ids = body.results.map((r) => r.id);
		expect(ids).toContain("10");

		nodeA.docs.delete("python");
	});
});

describe("GET /stats", () => {
	test("aggregates doc_count across live nodes", async () => {
		nodeA.docs.set("a", 1);
		nodeA.docs.set("b", 2);

		await app.handle(
			new Request("http://localhost/nodes/register", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ nodeId: nodeA.nodeId, url: nodeA.url }),
			}),
		);

		const res = await app.handle(new Request("http://localhost/stats"));
		expect(res.status).toBe(200);
		const body = await asJson<StatsBody>(res);
		expect(body.live_nodes).toBe(1);
		expect(body.total_docs).toBeGreaterThanOrEqual(2);

		nodeA.docs.clear();
	});
});

describe("NodeRegistry heartbeat + rebalance", () => {
	test("markDead triggers onNodeDead callback once", () => {
		registry.register("dead01", "http://localhost:19099");
		let callCount = 0;
		registry.onNodeDead(() => {
			callCount++;
		});
		registry.markDead("dead01");
		registry.markDead("dead01"); // second call should be no-op
		expect(callCount).toBe(1);
	});

	test("recordTermOwnership and termsOf round-trip", () => {
		registry.register(nodeA.nodeId, nodeA.url);
		registry.recordTermOwnership(nodeA.nodeId, ["rust", "async"]);
		registry.recordTermOwnership(nodeA.nodeId, ["tokio"]);
		const terms = registry.termsOf(nodeA.nodeId);
		expect(terms).toContain("rust");
		expect(terms).toContain("async");
		expect(terms).toContain("tokio");
	});
});
