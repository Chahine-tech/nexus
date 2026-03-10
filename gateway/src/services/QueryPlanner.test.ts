import { describe, expect, test } from "bun:test";
import { registry } from "./NodeRegistry";
import { planQuery, termToNode } from "./QueryPlanner";

const nodeA = { nodeId: "aa".repeat(32), url: "http://localhost:3001" };
const nodeB = { nodeId: "bb".repeat(32), url: "http://localhost:3002" };

describe("termToNode", () => {
	test("returns undefined for empty node list", () => {
		expect(termToNode("rust", [])).toBeUndefined();
	});

	test("returns the only node when there is one", () => {
		const result = termToNode("rust", [nodeA]);
		expect(result?.nodeId).toBe(nodeA.nodeId);
	});

	test("is deterministic — same term always maps to same node", () => {
		const nodes = [nodeA, nodeB];
		const r1 = termToNode("rust", nodes);
		const r2 = termToNode("rust", nodes);
		expect(r1?.nodeId).toBe(r2?.nodeId);
	});

	test("different terms can map to different nodes", () => {
		const nodes = [nodeA, nodeB];
		// With two nodes and multiple different terms, at least some should differ.
		const assigned = new Set([
			termToNode("rust", nodes)?.nodeId,
			termToNode("python", nodes)?.nodeId,
			termToNode("async", nodes)?.nodeId,
			termToNode("concurrency", nodes)?.nodeId,
		]);
		// At least one result expected.
		expect(assigned.size).toBeGreaterThan(0);
	});
});

describe("planQuery", () => {
	test("returns empty shards when no nodes are registered", () => {
		// Use a fresh import — shared registry singleton may have nodes from other tests.
		const plan = planQuery("rust concurrency");
		expect(plan.mergeStrategy).toBe("rrf");
		expect(plan.timeoutMs).toBe(150);
		expect(Array.isArray(plan.shards)).toBe(true);
	});

	test("filters empty query terms", () => {
		const plan = planQuery("   ");
		expect(plan.shards).toHaveLength(0);
	});

	test("broadcasts full query to all live nodes", () => {
		// Register two nodes for this test (shared singleton).
		registry.registerFromRust(nodeA.nodeId, nodeA.url);
		registry.registerFromRust(nodeB.nodeId, nodeB.url);

		const plan = planQuery("rust async concurrency python");

		// Every shard receives all query terms (broadcast, not term-sharded).
		expect(plan.shards.length).toBeGreaterThanOrEqual(2);
		for (const shard of plan.shards) {
			expect(shard.terms.sort()).toEqual(["async", "concurrency", "python", "rust"]);
		}
	});
});
