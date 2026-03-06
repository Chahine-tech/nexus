use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use thiserror::Error;
use tokio::sync::Semaphore;
use url::Url;

use crate::crawler::fetcher::{FetchError, Fetcher};
use crate::crawler::frontier::Frontier;
use crate::crawler::robots::RobotsCache;
use crate::node::Node;

#[derive(Debug, Error)]
pub enum CrawlerError {
    #[error("fetch error: {0}")]
    Fetch(#[from] FetchError),

    #[error("invalid seed URL: {0}")]
    InvalidUrl(#[from] url::ParseError),

    #[error("failed to build HTTP client: {0}")]
    ClientBuild(String),
}

/// Crawler tuning knobs.
#[derive(Debug, Clone)]
pub struct CrawlerConfig {
    /// Maximum crawl depth from seed URLs.
    pub max_depth: usize,
    /// Maximum number of concurrent in-flight HTTP requests.
    pub concurrency: usize,
    /// Maximum total pages to crawl (0 = unlimited).
    pub page_limit: u32,
}

impl Default for CrawlerConfig {
    fn default() -> Self {
        Self { max_depth: 3, concurrency: 8, page_limit: 10_000 }
    }
}

/// Async web crawler — fetches pages, respects robots.txt, indexes content.
///
/// Share via `Arc<Crawler>` across tokio tasks.
pub struct Crawler {
    fetcher: Arc<Fetcher>,
    frontier: Arc<Frontier>,
    robots: Arc<RobotsCache>,
    node: Arc<Node>,
    semaphore: Arc<Semaphore>,
    config: CrawlerConfig,
    doc_id: AtomicU32,
}

impl Crawler {
    pub fn new(node: Arc<Node>, config: CrawlerConfig) -> Result<Self, CrawlerError> {
        let fetcher = Fetcher::new().map_err(|e| CrawlerError::ClientBuild(e.to_string()))?;
        Ok(Self {
            fetcher: Arc::new(fetcher),
            frontier: Arc::new(Frontier::new(config.max_depth)),
            robots: Arc::new(RobotsCache::new()),
            node,
            semaphore: Arc::new(Semaphore::new(config.concurrency)),
            config,
            doc_id: AtomicU32::new(0),
        })
    }

    /// Seeds the frontier and starts the crawl loop.
    ///
    /// Fetches robots.txt for each seed origin before enqueuing.
    pub async fn run(&self, seeds: Vec<Url>) -> Result<(), CrawlerError> {
        // Pre-fetch robots.txt for each unique origin.
        for seed in &seeds {
            self.fetch_robots(seed).await;
            if self.robots.is_allowed(seed) {
                self.frontier.seed(seed.clone());
            }
        }

        // BFS crawl loop.
        loop {
            let Some(url) = self.frontier.pop() else {
                break;
            };

            if !self.robots.is_allowed(&url) {
                tracing::debug!(%url, "blocked by robots.txt");
                continue;
            }

            if self.config.page_limit > 0
                && self.doc_id.load(Ordering::Relaxed) >= self.config.page_limit
            {
                tracing::info!("page limit reached, stopping");
                break;
            }

            // acquire_owned only fails if the semaphore is closed, which never happens here.
            let Ok(permit) = self.semaphore.clone().acquire_owned().await else {
                break;
            };

            let fetcher = Arc::clone(&self.fetcher);
            let frontier = Arc::clone(&self.frontier);
            let robots = Arc::clone(&self.robots);
            let node = Arc::clone(&self.node);
            let doc_id = self.doc_id.fetch_add(1, Ordering::Relaxed);

            tokio::spawn(async move {
                let _permit = permit;

                match fetcher.fetch(url.clone()).await {
                    Ok(page) => {
                        tracing::debug!(%url, doc_id, "fetched page");

                        // Index extracted text.
                        node.index_document(doc_id, &page.text);

                        // Prefetch robots for new origins before enqueuing.
                        let mut new_links = Vec::new();
                        for link in page.links {
                            if robots.is_allowed(&link) {
                                new_links.push(link);
                            } else {
                                // Attempt to fetch robots for unknown origin (fail-open).
                                if let Some(robots_url) = RobotsCache::robots_url(&link)
                                    && let Ok(resp) = fetcher.fetch(robots_url).await
                                {
                                    robots.insert(&link, &resp.text);
                                }
                                if robots.is_allowed(&link) {
                                    new_links.push(link);
                                }
                            }
                        }

                        // Depth is unknown at this point (frontier doesn't track it per URL).
                        // We pass depth=1 so enqueue_links applies max_depth filtering at
                        // the frontier level relative to the seed (simplification for week 1).
                        frontier.enqueue_links(new_links, 1);
                    }
                    Err(e) => {
                        tracing::warn!(%url, error = %e, "fetch failed");
                    }
                }
            });
        }

        Ok(())
    }

    /// Fetches and caches robots.txt for `url`'s origin. Fails silently (open).
    async fn fetch_robots(&self, url: &Url) {
        let Some(robots_url) = RobotsCache::robots_url(url) else {
            return;
        };
        match self.fetcher.fetch(robots_url).await {
            Ok(page) => self.robots.insert(url, &page.text),
            Err(e) => {
                tracing::debug!(origin = %url, error = %e, "robots.txt fetch failed, allowing");
            }
        }
    }

    pub fn visited_count(&self) -> usize {
        self.frontier.visited_count()
    }
}
