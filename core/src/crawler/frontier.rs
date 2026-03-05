use std::collections::{HashSet, VecDeque};

use tokio::sync::Mutex;
use url::Url;

/// FIFO frontier of URLs to crawl, with a visited set to avoid re-crawling.
///
/// All methods are async — use behind `Arc<Frontier>` across tokio tasks.
pub struct Frontier {
    queue: Mutex<VecDeque<Url>>,
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

    /// Seeds the frontier with an initial URL.
    pub async fn seed(&self, url: Url) {
        let key = canonical_key(&url);
        let mut visited = self.visited.lock().await;
        if visited.insert(key) {
            self.queue.lock().await.push_back(url);
        }
    }

    /// Pops the next URL to crawl, or `None` if the queue is empty.
    pub async fn pop(&self) -> Option<Url> {
        self.queue.lock().await.pop_front()
    }

    /// Enqueues `links` discovered at `depth`, filtering already-visited URLs.
    pub async fn enqueue_links(&self, links: Vec<Url>, depth: usize) {
        if depth >= self.max_depth {
            return;
        }
        let mut visited = self.visited.lock().await;
        let mut queue = self.queue.lock().await;
        for url in links {
            let key = canonical_key(&url);
            if visited.insert(key) {
                queue.push_back(url);
            }
        }
    }

    pub async fn visited_count(&self) -> usize {
        self.visited.lock().await.len()
    }

    pub async fn queue_len(&self) -> usize {
        self.queue.lock().await.len()
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

    #[tokio::test]
    async fn test_seed_and_pop() {
        let f = Frontier::new(3);
        let url = Url::parse("https://example.com").unwrap();
        f.seed(url.clone()).await;
        assert_eq!(f.queue_len().await, 1);
        assert!(f.pop().await.is_some());
        assert_eq!(f.queue_len().await, 0);
    }

    #[tokio::test]
    async fn test_no_duplicates() {
        let f = Frontier::new(3);
        let url = Url::parse("https://example.com/page").unwrap();
        f.seed(url.clone()).await;
        // Same URL with query string — same canonical key, should be deduped.
        let url2 = Url::parse("https://example.com/page?ref=1").unwrap();
        f.enqueue_links(vec![url2], 0).await;
        assert_eq!(f.queue_len().await, 1);
    }

    #[tokio::test]
    async fn test_max_depth_respected() {
        let f = Frontier::new(2);
        let links = vec![Url::parse("https://example.com/deep").unwrap()];
        f.enqueue_links(links, 2).await; // depth == max_depth, should not enqueue
        assert_eq!(f.queue_len().await, 0);
    }
}
