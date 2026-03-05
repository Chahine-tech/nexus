use reqwest::Client;
use scraper::{ElementRef, Html, Selector};
use thiserror::Error;
use url::Url;

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("invalid URL: {0}")]
    InvalidUrl(#[from] url::ParseError),

    #[error("non-success status {status} for {url}")]
    BadStatus { status: u16, url: String },
}

/// Result of a successful fetch.
pub struct FetchedPage {
    /// Canonical URL after redirects.
    pub url: Url,
    /// Plain text extracted from the HTML body.
    pub text: String,
    /// All hrefs found in <a> tags, resolved to absolute URLs.
    pub links: Vec<Url>,
}

/// Stateless HTTP fetcher — share via `Arc<Fetcher>` across tasks.
pub struct Fetcher {
    client: Client,
}

impl Fetcher {
    pub fn new() -> Result<Self, FetchError> {
        let client = Client::builder()
            .user_agent("NexusBot/0.1 (+https://github.com/nexus)")
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        Ok(Self { client })
    }

    /// Fetches `url`, extracts plain text and outbound links.
    pub async fn fetch(&self, url: Url) -> Result<FetchedPage, FetchError> {
        let response = self.client.get(url.as_str()).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(FetchError::BadStatus {
                status: status.as_u16(),
                url: url.to_string(),
            });
        }
        let final_url = response.url().clone();
        let html = response.text().await?;
        let (text, links) = extract(&html, &final_url);
        Ok(FetchedPage { url: final_url, text, links })
    }
}


// Static selectors — parsed once, valid by construction (literals can't be invalid CSS).
static SKIP_SEL: std::sync::OnceLock<Selector> = std::sync::OnceLock::new();
static BODY_SEL: std::sync::OnceLock<Selector> = std::sync::OnceLock::new();
static ANCHOR_SEL: std::sync::OnceLock<Selector> = std::sync::OnceLock::new();

fn skip_sel() -> &'static Selector {
    SKIP_SEL.get_or_init(|| {
        Selector::parse("script, style, noscript")
            .expect("static CSS selector is valid")
    })
}
fn body_sel() -> &'static Selector {
    BODY_SEL.get_or_init(|| Selector::parse("body").expect("static CSS selector is valid"))
}
fn anchor_sel() -> &'static Selector {
    ANCHOR_SEL
        .get_or_init(|| Selector::parse("a[href]").expect("static CSS selector is valid"))
}

/// Extracts plain text and resolved outbound links from raw HTML.
fn extract(html: &str, base: &Url) -> (String, Vec<Url>) {
    let doc = Html::parse_document(html);

    let text = doc
        .select(body_sel())
        .next()
        .map(|body| {
            body.descendants()
                .filter(|node| {
                    node.value().is_text()
                        && !node.ancestors().any(|a| {
                            ElementRef::wrap(a)
                                .map(|el| skip_sel().matches(&el))
                                .unwrap_or(false)
                        })
                })
                .filter_map(|node| node.value().as_text().map(|t| t.trim()))
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();

    let links = doc
        .select(anchor_sel())
        .filter_map(|el| el.value().attr("href"))
        .filter_map(|href| base.join(href).ok())
        .filter(|u| u.scheme() == "http" || u.scheme() == "https")
        .collect();

    (text, links)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_and_links() {
        let html = r#"<html><body>
            <p>Hello <strong>Rust</strong> world</p>
            <script>alert("skip me")</script>
            <a href="/about">About</a>
            <a href="https://example.com">External</a>
        </body></html>"#;
        let base = Url::parse("https://test.com").unwrap();
        let (text, links) = extract(html, &base);
        assert!(text.contains("Hello"));
        assert!(text.contains("Rust"));
        assert!(!text.contains("skip me"));
        assert_eq!(links.len(), 2);
        assert!(links.iter().any(|u| u.as_str().contains("/about")));
        assert!(links.iter().any(|u| u.as_str().contains("example.com")));
    }

    #[test]
    fn test_extract_ignores_non_http_links() {
        let html = r#"<html><body>
            <a href="mailto:a@b.com">mail</a>
            <a href="ftp://example.com">ftp</a>
            <a href="https://ok.com">ok</a>
        </body></html>"#;
        let base = Url::parse("https://test.com").unwrap();
        let (_, links) = extract(html, &base);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].as_str(), "https://ok.com/");
    }
}
