import { blake3 } from "@noble/hashes/blake3.js";
import type { NodeId } from "../errors";
import { registry } from "./NodeRegistry";

export interface QueryPlan {
	shards: Array<{ nodeId: NodeId; url: string; terms: string[] }>;
	mergeStrategy: "rrf" | "boolean";
	timeoutMs: number;
}

// blake3 hash of a term as a Uint8Array — matches the Rust node's routing key.
// NodeId on the Rust side is also blake3(ed25519_verifying_key_bytes), so XOR
// distance is computed in the same 32-byte space.
function termKey(term: string): Uint8Array {
	return blake3(new TextEncoder().encode(term));
}

function xorDistance(a: Uint8Array, b: Uint8Array): Uint8Array {
	const result = new Uint8Array(32);
	for (let i = 0; i < 32; i++) {
		result[i] = (a[i] ?? 0) ^ (b[i] ?? 0);
	}
	return result;
}

function bytesLt(a: Uint8Array, b: Uint8Array): boolean {
	for (let i = 0; i < 32; i++) {
		if ((a[i] ?? 0) < (b[i] ?? 0)) return true;
		if ((a[i] ?? 0) > (b[i] ?? 0)) return false;
	}
	return false;
}

// Returns the live node whose nodeId has minimum XOR distance to the term's blake3 key.
// NodeId is stored as a hex string in the registry; we decode it to 32 bytes for comparison.
export function termToNode(
	term: string,
	nodes: Array<{ nodeId: NodeId; url: string }>,
): { nodeId: NodeId; url: string } | undefined {
	if (nodes.length === 0) return undefined;
	const key = termKey(term);
	return nodes.reduce((best, node) => {
		const dBest = xorDistance(key, Buffer.from(best.nodeId, "hex"));
		const dNode = xorDistance(key, Buffer.from(node.nodeId, "hex"));
		return bytesLt(dNode, dBest) ? node : best;
	});
}

// Groups terms by their responsible node, then builds one shard per node.
// Each shard contains only the terms routed to that node, so the node can
// search locally without QUIC fanout.
//
// Falls back to broadcast when there are no live nodes.
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

	if (nodes.length === 1) {
		// Single node — no point routing; send everything.
		const [node] = nodes;
		return {
			shards: [{ nodeId: node.nodeId, url: node.url, terms }],
			mergeStrategy: "rrf",
			timeoutMs: 150,
		};
	}

	// Group terms by responsible node (blake3 XOR closest).
	const shardMap = new Map<
		string,
		{ nodeId: NodeId; url: string; terms: string[] }
	>();
	for (const term of terms) {
		const node = termToNode(term, nodes);
		if (!node) continue;
		const existing = shardMap.get(node.nodeId);
		if (existing) {
			existing.terms.push(term);
		} else {
			shardMap.set(node.nodeId, {
				nodeId: node.nodeId,
				url: node.url,
				terms: [term],
			});
		}
	}

	const shards = [...shardMap.values()];

	// Edge case: if all terms mapped to the same node, skip the overhead.
	return { shards, mergeStrategy: "rrf", timeoutMs: 150 };
}
