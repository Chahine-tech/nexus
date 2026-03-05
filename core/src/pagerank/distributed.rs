use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::network::messages::NodeId;

/// Maximum number of partial scores gossiped per message to bound payload size.
const PAGERANK_GOSSIP_MAX_ENTRIES: usize = 10_000;

/// Partial PageRank contributions from one node, propagated via gossip.
///
/// Merge rule: timestamp-wins per node (last-write-wins). This is intentionally
/// NOT an additive CRDT — summing is done at read time across all peer entries,
/// not at write time. This avoids double-counting when a broadcast is repeated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipPagerank {
    /// Identity of the originating node.
    pub node_id: NodeId,
    /// Partial scores: doc_id → local PageRank contribution.
    /// Only the top `PAGERANK_GOSSIP_MAX_ENTRIES` scores (by value) are included.
    pub partial_scores: HashMap<u32, f32>,
    /// Unix epoch seconds — used for timestamp-wins conflict resolution.
    pub timestamp: u64,
}

impl GossipPagerank {
    /// Builds a `GossipPagerank` message from a full score map.
    ///
    /// Scores below `1e-6` are dropped; the result is capped at
    /// `PAGERANK_GOSSIP_MAX_ENTRIES` entries (highest scores retained).
    pub fn from_scores(node_id: NodeId, scores: HashMap<u32, f32>, timestamp: u64) -> Self {
        let mut entries: Vec<(u32, f32)> = scores
            .into_iter()
            .filter(|(_, s)| *s > 1e-6)
            .collect();

        // Keep the top-K entries by score to bound message size.
        if entries.len() > PAGERANK_GOSSIP_MAX_ENTRIES {
            entries.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            entries.truncate(PAGERANK_GOSSIP_MAX_ENTRIES);
        }

        Self {
            node_id,
            partial_scores: entries.into_iter().collect(),
            timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_scores_drops_tiny_values() {
        let mut scores = HashMap::new();
        scores.insert(0, 0.5);
        scores.insert(1, 1e-8); // below threshold
        scores.insert(2, 0.3);

        let msg = GossipPagerank::from_scores(NodeId([0u8; 32]), scores, 1);
        assert!(!msg.partial_scores.contains_key(&1));
        assert!(msg.partial_scores.contains_key(&0));
        assert!(msg.partial_scores.contains_key(&2));
    }

    #[test]
    fn test_from_scores_caps_at_max_entries() {
        let scores: HashMap<u32, f32> = (0..(PAGERANK_GOSSIP_MAX_ENTRIES + 100) as u32)
            .map(|i| (i, 1.0 / (i + 1) as f32))
            .collect();

        let msg = GossipPagerank::from_scores(NodeId([0u8; 32]), scores, 1);
        assert_eq!(msg.partial_scores.len(), PAGERANK_GOSSIP_MAX_ENTRIES);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut scores = HashMap::new();
        scores.insert(42u32, 0.25f32);
        scores.insert(99u32, 0.75f32);

        let msg = GossipPagerank::from_scores(NodeId([7u8; 32]), scores, 999);
        let bytes = rmp_serde::to_vec(&msg).expect("serialize");
        let decoded: GossipPagerank = rmp_serde::from_slice(&bytes).expect("deserialize");

        assert_eq!(decoded.timestamp, 999);
        assert!((decoded.partial_scores[&42] - 0.25).abs() < 1e-6);
        assert!((decoded.partial_scores[&99] - 0.75).abs() < 1e-6);
    }
}
