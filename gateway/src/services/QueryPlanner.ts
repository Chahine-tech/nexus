import { createHash } from "node:crypto";
import type { NodeId } from "../errors";
import { registry } from "./NodeRegistry";

export interface QueryPlan {
	shards: Array<{ nodeId: NodeId; url: string; terms: string[] }>;
	mergeStrategy: "rrf" | "boolean";
	timeoutMs: number;
}

// SHA-256 of a term as a Buffer — used for XOR consistent hashing.
// Note: Rust nodes use blake3 internally. This only affects gateway-side load distribution,
// not correctness — each Rust node handles any QueryRequest it receives.
function termKey(term: string): Buffer {
	return createHash("sha256").update(term).digest();
}

function xorDistance(a: Buffer, b: Buffer): Buffer {
	const len = Math.max(a.length, b.length);
	const result = Buffer.alloc(len);
	for (let i = 0; i < len; i++) {
		result[i] = (a[i] ?? 0) ^ (b[i] ?? 0);
	}
	return result;
}

function bufferLt(a: Buffer, b: Buffer): boolean {
	for (let i = 0; i < Math.max(a.length, b.length); i++) {
		if ((a[i] ?? 0) < (b[i] ?? 0)) return true;
		if ((a[i] ?? 0) > (b[i] ?? 0)) return false;
	}
	return false;
}

// Returns the live node whose nodeId has minimum XOR distance to the term's SHA-256 key.
// Currently unused by planQuery (broadcast mode), but available for future term-sharded routing.
export function termToNode(
	term: string,
	nodes: Array<{ nodeId: NodeId; url: string }>,
): { nodeId: NodeId; url: string } | undefined {
	if (nodes.length === 0) return undefined;
	const key = termKey(term);
	return nodes.reduce((best, node) => {
		const dBest = xorDistance(key, Buffer.from(best.nodeId, "hex"));
		const dNode = xorDistance(key, Buffer.from(node.nodeId, "hex"));
		return bufferLt(dNode, dBest) ? node : best;
	});
}

export function planQuery(query: string): QueryPlan {
	const terms = query
		.trim()
		.toLowerCase()
		.split(/\s+/)
		.filter((t) => t.length > 0);

	const nodes = registry.liveNodes();

	if (nodes.length === 0) {
		return { shards: [], mergeStrategy: "rrf", timeoutMs: 150 };
	}

	// Broadcast the full query to every live node and merge results via RRF.
	// Each node searches its local shard independently; the gateway re-ranks.
	const shards = nodes.map(({ nodeId, url }) => ({ nodeId, url, terms }));

	return { shards, mergeStrategy: "rrf", timeoutMs: 150 };
}
