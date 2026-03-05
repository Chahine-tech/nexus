use dashmap::DashMap;
use url::Url;

/// Minimal robots.txt parser and cache.
///
/// Fetching is delegated to the caller — this struct only parses and caches.
/// Cache is keyed by origin (scheme + host). Thread-safe via DashMap.
pub struct RobotsCache {
    /// origin → list of disallowed path prefixes for NexusBot.
    cache: DashMap<String, Vec<String>>,
}

impl RobotsCache {
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
        }
    }

    /// Returns `true` if `url` is allowed to be crawled.
    ///
    /// Unknown origins (robots.txt not yet fetched) are allowed by default.
    pub fn is_allowed(&self, url: &Url) -> bool {
        let origin = origin_key(url);
        let Some(disallowed) = self.cache.get(&origin) else {
            return true;
        };
        let path = url.path();
        !disallowed.iter().any(|prefix| path.starts_with(prefix.as_str()))
    }

    /// Parses `robots_txt` content and caches the rules for `origin_url`.
    pub fn insert(&self, origin_url: &Url, robots_txt: &str) {
        let disallowed = parse_disallowed(robots_txt);
        self.cache.insert(origin_key(origin_url), disallowed);
    }

    /// Returns the robots.txt URL for a given page URL.
    pub fn robots_url(url: &Url) -> Option<Url> {
        let mut robots = url.clone();
        robots.set_path("/robots.txt");
        robots.set_query(None);
        robots.set_fragment(None);
        Some(robots)
    }
}

impl Default for RobotsCache {
    fn default() -> Self {
        Self::new()
    }
}

fn origin_key(url: &Url) -> String {
    format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""))
}

/// Extracts `Disallow` paths from robots.txt that apply to NexusBot or `*`.
///
/// Respects `User-agent: *` and `User-agent: NexusBot` blocks.
fn parse_disallowed(content: &str) -> Vec<String> {
    let mut disallowed = Vec::new();
    let mut in_relevant_block = false;

    for line in content.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(agent) = line.strip_prefix("User-agent:") {
            let agent = agent.trim().to_lowercase();
            in_relevant_block = agent == "*" || agent == "nexusbot";
        } else if in_relevant_block {
            if let Some(path) = line.strip_prefix("Disallow:") {
                let path = path.trim();
                if !path.is_empty() {
                    disallowed.push(path.to_string());
                }
            }
        }
    }

    disallowed
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROBOTS: &str = "
User-agent: *
Disallow: /admin
Disallow: /private/

User-agent: Googlebot
Disallow: /nogoogle
";

    #[test]
    fn test_disallowed_path_blocked() {
        let cache = RobotsCache::new();
        let origin = Url::parse("https://example.com").unwrap();
        cache.insert(&origin, ROBOTS);
        let url = Url::parse("https://example.com/admin/settings").unwrap();
        assert!(!cache.is_allowed(&url));
    }

    #[test]
    fn test_allowed_path_passes() {
        let cache = RobotsCache::new();
        let origin = Url::parse("https://example.com").unwrap();
        cache.insert(&origin, ROBOTS);
        let url = Url::parse("https://example.com/public/page").unwrap();
        assert!(cache.is_allowed(&url));
    }

    #[test]
    fn test_unknown_origin_allowed() {
        let cache = RobotsCache::new();
        let url = Url::parse("https://unknown.com/anything").unwrap();
        assert!(cache.is_allowed(&url));
    }

    #[test]
    fn test_robots_url() {
        let url = Url::parse("https://example.com/some/page?q=1").unwrap();
        let robots = RobotsCache::robots_url(&url).unwrap();
        assert_eq!(robots.as_str(), "https://example.com/robots.txt");
    }
}
