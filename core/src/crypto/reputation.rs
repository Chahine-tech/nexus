use dashmap::DashMap;

use crate::network::messages::NodeId;

const SUCCESS_DELTA: i32 = 1;
const FAILURE_DELTA: i32 = -3;
const TRUST_THRESHOLD: i32 = 0;

/// Tracks a simple reputation score per peer node.
///
/// Thread-safe via DashMap — no locking required.
pub struct ReputationStore {
    scores: DashMap<NodeId, i32>,
}

impl ReputationStore {
    pub fn new() -> Self {
        Self { scores: DashMap::new() }
    }

    pub fn record_success(&self, node_id: &NodeId) {
        self.scores
            .entry(node_id.clone())
            .and_modify(|s| *s = s.saturating_add(SUCCESS_DELTA))
            .or_insert(SUCCESS_DELTA);
    }

    pub fn record_failure(&self, node_id: &NodeId) {
        self.scores
            .entry(node_id.clone())
            .and_modify(|s| *s = s.saturating_add(FAILURE_DELTA))
            .or_insert(FAILURE_DELTA);
    }

    pub fn score(&self, node_id: &NodeId) -> i32 {
        self.scores.get(node_id).map(|s| *s).unwrap_or(0)
    }

    pub fn is_trusted(&self, node_id: &NodeId) -> bool {
        self.score(node_id) >= TRUST_THRESHOLD
    }
}

impl Default for ReputationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id(byte: u8) -> NodeId {
        NodeId([byte; 32])
    }

    #[test]
    fn success_increases_score() {
        let store = ReputationStore::new();
        let id = make_id(1);
        assert_eq!(store.score(&id), 0);
        store.record_success(&id);
        assert_eq!(store.score(&id), 1);
        store.record_success(&id);
        assert_eq!(store.score(&id), 2);
    }

    #[test]
    fn failure_decreases_score() {
        let store = ReputationStore::new();
        let id = make_id(2);
        store.record_failure(&id);
        assert_eq!(store.score(&id), -3);
        store.record_failure(&id);
        assert_eq!(store.score(&id), -6);
    }

    #[test]
    fn unknown_node_is_trusted() {
        let store = ReputationStore::new();
        let id = make_id(3);
        assert!(store.is_trusted(&id));
    }

    #[test]
    fn trust_threshold() {
        let store = ReputationStore::new();
        let id = make_id(4);
        store.record_failure(&id); // -3
        assert!(!store.is_trusted(&id));
        store.record_success(&id); // -2
        store.record_success(&id); // -1
        assert!(!store.is_trusted(&id));
        store.record_success(&id); // 0
        assert!(store.is_trusted(&id));
    }
}
