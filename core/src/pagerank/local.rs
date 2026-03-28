use std::collections::{HashMap, HashSet};

/// Damping factor for the standard PageRank formula.
const DAMPING: f32 = 0.85;

/// Maximum iterations before forced termination (safety cap).
const MAX_ITER: usize = 100;

/// Convergence threshold: max absolute delta across all scores per iteration.
const EPSILON: f32 = 1e-6;

/// PageRank computed via power iteration over a local link graph.
///
/// Build the graph with `add_link`, then call `iterate()` to run power iteration.
/// Scores are accessible via `score()` and `ranked()` after `iterate()`.
pub struct LocalPageRank {
    /// Adjacency list: src → set of outbound destinations (deduplication).
    graph: HashMap<u32, HashSet<u32>>,
    /// PageRank scores after the last `iterate()` call. Empty before first call.
    scores: HashMap<u32, f32>,
}

impl LocalPageRank {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self { graph: HashMap::new(), scores: HashMap::new() }
    }

    /// Adds a directed edge `src → dst`.
    ///
    /// Duplicate edges are silently ignored. Both `src` and `dst` are registered
    /// as nodes, so dangling destination nodes receive rank contributions.
    pub fn add_link(&mut self, src: u32, dst: u32) {
        self.graph.entry(src).or_default().insert(dst);
        // Ensure dst exists in the node set even without outbound links.
        self.graph.entry(dst).or_default();
    }

    /// Runs power iteration until convergence or `max_iter` steps.
    ///
    /// `max_iter` is internally capped at `MAX_ITER` (100). Returns the number
    /// of iterations performed. Scores persist and are accessible via `score()`.
    pub fn iterate(&mut self, max_iter: usize) -> usize {
        let nodes: Vec<u32> = self.graph.keys().copied().collect();
        let n = nodes.len();

        if n == 0 {
            return 0;
        }

        let limit = max_iter.min(MAX_ITER);
        let base_rank = 1.0_f32 / n as f32;

        // Pre-compute out-degrees.
        let out_degree: HashMap<u32, usize> = self
            .graph
            .iter()
            .map(|(&node, edges)| (node, edges.len()))
            .collect();

        // Initialize all scores uniformly.
        let mut scores: HashMap<u32, f32> = nodes.iter().map(|&id| (id, base_rank)).collect();

        let mut iters_done = limit;
        for iter in 0..limit {
            // Dangling nodes (out-degree == 0) redistribute their rank uniformly.
            let dangling_sum: f32 = nodes
                .iter()
                .filter(|&&id| out_degree.get(&id).copied().unwrap_or(0) == 0)
                .map(|&id| scores.get(&id).copied().unwrap_or(0.0))
                .sum();

            let dangling_contrib = DAMPING * dangling_sum * base_rank;

            // Next iteration scores: teleportation + dangling contribution.
            let mut next: HashMap<u32, f32> = nodes
                .iter()
                .map(|&id| (id, (1.0 - DAMPING) * base_rank + dangling_contrib))
                .collect();

            // Distribute outbound rank from each non-dangling node.
            for (&src, dsts) in &self.graph {
                let deg = dsts.len();
                if deg == 0 {
                    continue;
                }
                let contrib = DAMPING * scores.get(&src).copied().unwrap_or(0.0) / deg as f32;
                for &dst in dsts {
                    *next.entry(dst).or_insert(0.0) += contrib;
                }
            }

            // Convergence check: max absolute delta across all nodes.
            let max_delta = nodes
                .iter()
                .map(|&id| {
                    let old = scores.get(&id).copied().unwrap_or(0.0);
                    let new = next.get(&id).copied().unwrap_or(0.0);
                    (old - new).abs()
                })
                .fold(0.0_f32, f32::max);

            scores = next;

            if max_delta < EPSILON {
                iters_done = iter + 1;
                break;
            }
        }

        self.scores = scores;
        iters_done
    }

    /// Returns the PageRank score for `doc_id`. Returns `0.0` if unknown or before `iterate()`.
    #[allow(dead_code)]
    pub fn score(&self, doc_id: u32) -> f32 {
        self.scores.get(&doc_id).copied().unwrap_or(0.0)
    }

    /// Returns all `(doc_id, score)` pairs sorted by descending score.
    pub fn ranked(&self) -> Vec<(u32, f32)> {
        let mut v: Vec<(u32, f32)> = self.scores.iter().map(|(&id, &s)| (id, s)).collect();
        v.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }

    /// Returns a clone of the current score map.
    ///
    /// Used by `GossipPagerank` to propagate local contributions to peers.
    pub fn scores_snapshot(&self) -> HashMap<u32, f32> {
        self.scores.clone()
    }
}

impl Default for LocalPageRank {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_iterate_returns_zero() {
        let mut pr = LocalPageRank::new();
        assert_eq!(pr.iterate(10), 0);
        assert_eq!(pr.score(0), 0.0);
    }

    #[test]
    fn test_single_self_loop_converges() {
        let mut pr = LocalPageRank::new();
        pr.add_link(0, 0);
        let iters = pr.iterate(100);
        assert!(iters > 0 && iters <= 100);
        assert!(pr.score(0) > 0.0);
    }

    #[test]
    fn test_two_node_cycle_produces_equal_scores() {
        let mut pr = LocalPageRank::new();
        pr.add_link(0, 1);
        pr.add_link(1, 0);
        pr.iterate(100);
        let s0 = pr.score(0);
        let s1 = pr.score(1);
        assert!((s0 - s1).abs() < 1e-4, "expected equal scores: {s0} vs {s1}");
    }

    #[test]
    fn test_dangling_node_gets_nonzero_score() {
        let mut pr = LocalPageRank::new();
        // Node 0 links to node 1. Node 1 is dangling (no outbound links).
        pr.add_link(0, 1);
        pr.add_link(0, 0); // self-loop to give node 0 some outbound links
        pr.iterate(100);
        assert!(pr.score(1) > 0.0, "dangling node should receive rank");
    }

    #[test]
    fn test_ranked_is_sorted_descending() {
        let mut pr = LocalPageRank::new();
        pr.add_link(0, 1);
        pr.add_link(0, 2);
        pr.add_link(1, 2);
        pr.iterate(100);
        let ranked = pr.ranked();
        for window in ranked.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "ranked not sorted: {} < {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn test_add_link_is_idempotent() {
        let mut pr = LocalPageRank::new();
        pr.add_link(0, 1);
        pr.add_link(0, 1); // duplicate
        // Out-degree of node 0 should still be 1.
        assert_eq!(pr.graph[&0].len(), 1);
    }

    #[test]
    fn test_iterate_converges_before_max_on_simple_graph() {
        let mut pr = LocalPageRank::new();
        pr.add_link(0, 1);
        pr.add_link(1, 0);
        let iters = pr.iterate(100);
        // A 2-node symmetric graph should converge in far fewer than 100 iterations.
        assert!(iters < 100, "expected fast convergence, got {iters} iters");
    }

    #[test]
    fn test_scores_snapshot_matches_score() {
        let mut pr = LocalPageRank::new();
        pr.add_link(0, 1);
        pr.add_link(1, 0);
        pr.iterate(100);
        let snap = pr.scores_snapshot();
        for (&id, &s) in &snap {
            assert_eq!(pr.score(id), s);
        }
    }

    #[test]
    fn test_scores_sum_approximately_one() {
        let mut pr = LocalPageRank::new();
        // Star graph: node 0 links to nodes 1, 2, 3.
        for dst in 1..=3 {
            pr.add_link(0, dst);
        }
        pr.iterate(100);
        let total: f32 = pr.ranked().iter().map(|(_, s)| s).sum();
        assert!((total - 1.0).abs() < 0.01, "scores should sum to ~1.0, got {total}");
    }
}
