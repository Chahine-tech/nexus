import type { NodeId } from "../errors";
import { registry } from "./NodeRegistry";

export interface QueryPlan {
	shards: Array<{ nodeId: NodeId; url: string; terms: string[] }>;
	mergeStrategy: "rrf" | "boolean";
	timeoutMs: number;
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
