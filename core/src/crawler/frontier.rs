use std::collections::{HashSet, VecDeque};
use std::sync::Mutex;

use url::Url;

/// FIFO frontier of URLs to crawl, with a visited set to avoid re-crawling.
///
/// Each entry tracks its crawl depth so `max_depth` is correctly enforced
/// across all BFS levels, not just the first hop.
///
/// No async operations held across lock — `std::sync::Mutex` is correct here.
/// Use behind `Arc<Frontier>` across tokio tasks.
pub struct Frontier {
    /// (url, depth) — depth of the URL when it was enqueued.
    queue: Mutex<VecDeque<(Url, usize)>>,
    visited: Mutex<HashSet<String>>,
    max_depth: usize,
}

impl Frontier {
    pub fn new(max_depth: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            visited: Mutex::new(HashSet::new()),
            max_depth,
        }
    }

    /// Seeds the frontier with an initial URL at depth 0.
    pub fn seed(&self, url: Url) {
        let key = canonical_key(&url);
        let mut visited = self.visited.lock().expect("frontier visited mutex poisoned");
        if visited.insert(key) {
            self.queue.lock().expect("frontier queue mutex poisoned").push_back((url, 0));
        }
    }

    /// Pops the next `(url, depth)` to crawl, or `None` if the queue is empty.
    pub fn pop(&self) -> Option<(Url, usize)> {
        self.queue.lock().expect("frontier queue mutex poisoned").pop_front()
    }

    /// Enqueues `links` discovered at `parent_depth + 1`, filtering already-visited URLs
    /// and URLs that would exceed `max_depth`.
    pub fn enqueue_links(&self, links: Vec<Url>, parent_depth: usize) {
        let child_depth = parent_depth + 1;
        if child_depth >= self.max_depth {
            return;
        }
        let mut visited = self.visited.lock().expect("frontier visited mutex poisoned");
        let mut queue = self.queue.lock().expect("frontier queue mutex poisoned");
        for url in links {
            let key = canonical_key(&url);
            if visited.insert(key) {
                queue.push_back((url, child_depth));
            }
        }
    }

    pub fn visited_count(&self) -> usize {
        self.visited.lock().expect("frontier visited mutex poisoned").len()
    }

    pub fn queue_len(&self) -> usize {
        self.queue.lock().expect("frontier queue mutex poisoned").len()
    }
}

/// Canonical key for deduplication: scheme + host + path (no query/fragment).
fn canonical_key(url: &Url) -> String {
    format!(
        "{}://{}{}",
        url.scheme(),
        url.host_str().unwrap_or(""),
        url.path()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_and_pop() {
        let f = Frontier::new(3);
        let url = Url::parse("https://example.com").unwrap();
        f.seed(url.clone());
        assert_eq!(f.queue_len(), 1);
        let entry = f.pop();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().1, 0); // seed is always depth 0
        assert_eq!(f.queue_len(), 0);
    }

    #[test]
    fn test_no_duplicates() {
        let f = Frontier::new(3);
        let url = Url::parse("https://example.com/page").unwrap();
        f.seed(url.clone());
        // Same URL with query string — same canonical key, should be deduped.
        let url2 = Url::parse("https://example.com/page?ref=1").unwrap();
        f.enqueue_links(vec![url2], 0);
        assert_eq!(f.queue_len(), 1);
    }

    #[test]
    fn test_max_depth_respected() {
        let f = Frontier::new(2);
        let links = vec![Url::parse("https://example.com/deep").unwrap()];
        // parent_depth=1, child_depth=2 == max_depth → should not enqueue
        f.enqueue_links(links, 1);
        assert_eq!(f.queue_len(), 0);
    }

    #[test]
    fn test_depth_propagated() {
        let f = Frontier::new(5);
        let links = vec![Url::parse("https://example.com/level2").unwrap()];
        f.enqueue_links(links, 1); // child_depth = 2
        let entry = f.pop().unwrap();
        assert_eq!(entry.1, 2);
    }
}
