"""
Media Outlet Scraper MVP — Basque digital news article extraction.

Scrapes Basque news outlets for dialect-labeled text data.
Uses sitemap.xml as primary source, RSS/Atom as fallback, HTML parsing as last resort.

Usage:
    uv run python -m src.data.media_scraper scrape --outlet berria --limit 50
    uv run python -m src.data.media_scraper scrape --outlet all --limit 100
    uv run python -m src.data.media_scraper discover  # check which outlets work

Architecture:
    Outlet config → Sitemap/RSS discovery → Article URL list → HTML fetch → text extraction → labeled output

Output format (JSONL):
    {"outlet": "goierri_hitza", "url": "...", "title": "...", "text": "...",
     "date": "2024-...", "dialect_class": "central", "scraped_at": "2026-..."}
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDIA_CSV = PROJECT_ROOT / "data" / "reference" / "basque_digital_media.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "text" / "media"
USER_AGENT = "ZeineuskiML/0.1 (Basque dialect research; xezpeleta@gmail.com)"

# SSL context that tolerates self-signed certificates (common on small outlets)
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class OutletConfig:
    """Configuration for a single media outlet."""

    name: str  # key used in filenames (snake_case)
    display_name: str  # human-readable
    url: str  # homepage
    dialect_class: str  # e.g., western, central, batua
    municipality: str
    territory: str  # Hegoalde / Iparralde
    type: str  # tokikoa, egunkaria, irratia, ...

    # Discovery strategy
    sitemap_urls: list[str] = field(default_factory=list)
    rss_urls: list[str] = field(default_factory=list)
    atom_urls: list[str] = field(default_factory=list)

    # Extraction rules
    article_link_selector: str = "a"  # CSS selector for article links on listing pages
    article_body_selector: str = "article"  # CSS selector for article body text
    article_title_selector: str = "h1"
    article_date_selector: str = "time"

    # Additional settings
    max_articles_per_run: int = 100
    respect_robots_txt: bool = True
    request_delay: float = 1.0  # seconds between requests


# ── Outlet registry ───────────────────────────────────────────────────────────


def load_outlet_registry() -> dict[str, OutletConfig]:
    """Load outlet configurations from the media CSV."""
    outlets = {}
    with open(MEDIA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = re.sub(r"[^a-z0-9]+", "_", row["name"].lower().strip())
            key = key.strip("_")

            cfg = OutletConfig(
                name=key,
                display_name=row["name"],
                url=row["url"],
                dialect_class=row["dialect_class"],
                municipality=row["municipality"],
                territory=row["territory"],
                type=row["type"],
            )

            # Auto-detect discovery URLs
            base_domain = urlparse(row["url"]).netloc

            # Standard sitemap locations
            cfg.sitemap_urls = [
                f"https://{base_domain}/sitemap.xml",
                f"https://{base_domain}/sitemap_index.xml",
                f"https://{base_domain}/wp-sitemap.xml",  # WordPress
            ]

            # Standard RSS/Atom
            cfg.rss_urls = [
                f"https://{base_domain}/feed/",  # WordPress RSS
                f"https://{base_domain}/rss/",
                f"https://{base_domain}/feed.xml",
            ]
            cfg.atom_urls = [
                f"https://{base_domain}/feed/atom/",
                f"https://{base_domain}/atom.xml",
            ]

            # Outlet-specific overrides (known feeds)
            overrides = {
                "berria": {
                    "rss_urls": [
                        "https://www.berria.eus/uploads/feeds/feed_berria_eu.xml"
                    ],
                    "sitemap_urls": ["https://www.berria.eus/sitemap.xml"],
                    "article_body_selector": "div.article-body, div.c-post__body",
                    "max_articles_per_run": 200,
                },
                "argia": {
                    "rss_urls": ["https://www.argia.eus/rss"],
                    "sitemap_urls": ["https://www.argia.eus/sitemap.xml"],
                    "article_body_selector": "div.article-body, div.page-body",
                    "max_articles_per_run": 200,
                },
                "eitb": {
                    "atom_urls": ["https://www.eitb.eus/eu/rss/sitemaps/last_changes/"],
                    "sitemap_urls": ["https://www.eitb.eus/eu/sitemap.xml"],
                    "article_body_selector": "div.eitb-content, article",
                    "max_articles_per_run": 200,
                },
                "goiena": {
                    "sitemap_urls": ["https://goiena.eus/sitemap.xml"],
                    "article_body_selector": "div.article-body, div.edukia",
                },
                "anboto": {
                    "sitemap_urls": ["https://anboto.org/sitemap.xml"],
                    "article_body_selector": "article, div.content",
                },
            }

            if key in overrides:
                ov = overrides[key]
                if isinstance(ov, dict):
                    for attr, value in ov.items():
                        setattr(cfg, attr, value)

            outlets[key] = cfg

    return outlets


# ── Article discovery ─────────────────────────────────────────────────────────


def discover_from_sitemap(
    sitemap_url: str,
    client: httpx.Client,
    limit: int = 1000,
    prefer_recent: bool = True,
) -> list[str]:
    """Parse a sitemap.xml and return article URLs.

    For sitemap indexes with numbered sub-sitemaps (content-0001..N),
    processes them in reverse order to get most recent content first.
    """
    try:
        resp = client.get(sitemap_url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.debug(f"Sitemap failed {sitemap_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "xml")
    urls: list[str] = []

    # Check if this is a sitemap index
    sitemap_tags = soup.find_all("sitemap")
    if sitemap_tags:
        logger.debug(f"Sitemap index with {len(sitemap_tags)} sub-sitemaps")
        sub_sitemaps = []
        for sm in sitemap_tags:
            loc = sm.find("loc")
            if loc and loc.text:
                sub_sitemaps.append(loc.text)

        # Process newest first (numbered sitemaps usually go old→new)
        if prefer_recent and len(sub_sitemaps) > 1:
            sub_sitemaps.reverse()

        for sm_url in sub_sitemaps:
            urls.extend(
                discover_from_sitemap(
                    sm_url, client, limit - len(urls), prefer_recent=False
                )
            )
            if len(urls) >= limit:
                break
        return urls[:limit]

    # Regular sitemap with <url> entries
    url_tags = list(soup.find_all("url"))
    # Take from end (usually most recent) if prefer_recent
    if prefer_recent:
        url_tags = url_tags[-limit:] if len(url_tags) > limit else url_tags
    for u in url_tags:
        loc = u.find("loc")
        if loc and loc.text:
            urls.append(loc.text)
            if len(urls) >= limit:
                break

    # Filter out non-article URLs (tag pages, category pages, author pages)
    skip_patterns = [
        r"/tag/",
        r"/kategoria/",
        r"/category/",
        r"/egilea/",
        r"/author/",
        r"/page/",
        r"/orria/",
        r"\?p=",
        r"/wp-admin/",
        r"/feed/",
        r"/comments/",
    ]
    urls = [u for u in urls if not any(re.search(p, u) for p in skip_patterns)]

    logger.debug(f"Sitemap {sitemap_url}: {len(urls)} URLs")
    return urls


def discover_from_rss(
    rss_url: str, client: httpx.Client, limit: int = 1000
) -> list[str]:
    """Parse RSS/Atom feed and return article URLs."""
    try:
        resp = client.get(rss_url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.debug(f"RSS failed {rss_url}: {e}")
        return []

    feed = feedparser.parse(resp.text)
    urls = []
    for entry in feed.entries[:limit]:
        if hasattr(entry, "link"):
            urls.append(entry.link)
        elif hasattr(entry, "links"):
            for link in entry.links:
                if link.get("rel") == "alternate" or link.get("type", "").startswith(
                    "text/html"
                ):
                    urls.append(link.href)
                    break

    logger.debug(f"RSS {rss_url}: {len(urls)} entries")
    return urls


def discover_articles(
    cfg: OutletConfig, client: httpx.Client, limit: int | None = None
) -> list[str]:
    """Discover article URLs for an outlet using sitemap → RSS → fallback."""
    limit = limit or cfg.max_articles_per_run
    urls: list[str] = []

    # 1. Try sitemap first (most complete)
    for sitemap_url in cfg.sitemap_urls[:3]:  # try top 3
        new_urls = discover_from_sitemap(sitemap_url, client, limit - len(urls))
        urls.extend(new_urls)
        if len(urls) >= limit:
            break
        time.sleep(0.3)

    # 2. Try RSS/Atom as fallback
    if len(urls) < limit:
        for rss_url in cfg.rss_urls + cfg.atom_urls:
            new_urls = discover_from_rss(rss_url, client, limit - len(urls))
            urls.extend(new_urls)
            if len(urls) >= limit:
                break
            time.sleep(0.3)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique[:limit]


# ── Article text extraction ───────────────────────────────────────────────────


def extract_article_text(html: str, cfg: OutletConfig) -> Optional[str]:
    """Extract clean article body text from HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "aside", "iframe"]):
        tag.decompose()

    # Remove sidebar/navigation/widget blocks
    skip_classes = [
        "social-share",
        "comments",
        "related-posts",
        "sidebar",
        "advertisement",
        "publi",
        "iragarkia",
        "nabigazioa",
        "widget",
        "alboko-menua",
        "goiko-barra",
        "footer",
        "header",
    ]
    for cls in skip_classes:
        for tag in soup.find_all(class_=re.compile(cls, re.I)):
            tag.decompose()

    # Try configured body selector
    body = None
    if cfg.article_body_selector:
        body = soup.select_one(cfg.article_body_selector)

    # Fallback: WordPress / common article containers
    if not body:
        for selector in [
            "div.entry-content",
            "div.post-content",
            "div.article-body",
            "article .content",
            "article",
            "div.content",
            "div.edukia",
            "div[itemprop='articleBody']",
            "main article",
            "main",
        ]:
            body = soup.select_one(selector)
            if body and len(body.get_text(strip=True)) > 200:
                break

    # If still nothing, try extracting all meaningful <p> blocks from the page
    if not body:
        paragraphs = soup.find_all("p")
        good_ps = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            # Skip boilerplate
            skip_phrases = [
                "egin zaitez",
                "harpidedun",
                "zure babesa",
                "kalitatez jaso",
                "cookie",
                "pribatutasun",
                "lege ohar",
                "RSS",
            ]
            if (
                text
                and len(text) > 30
                and not any(s in text.lower() for s in skip_phrases)
            ):
                # Also check the parent isn't a sidebar
                parent = p.parent
                if parent and parent.get("class"):
                    parent_cls = " ".join(str(parent.get("class", ""))).lower()
                    if any(s in parent_cls for s in skip_classes):
                        continue
                good_ps.append(text)
        if good_ps:
            return "\n\n".join(good_ps)
        return None

    if not body:
        return None

    # Extract clean text
    text = body.get_text(separator="\n", strip=True)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    if len(text) < 100:  # too short to be an article
        return None

    return text.strip()


def extract_article_metadata(html: str, cfg: OutletConfig) -> dict:
    """Extract title and date from article HTML."""
    soup = BeautifulSoup(html, "lxml")
    meta = {}

    # Title
    title = soup.find("h1")
    if title:
        meta["title"] = title.get_text(strip=True)
    else:
        # Try og:title
        og_title = soup.find("meta", property="og:title")
        if og_title:
            meta["title"] = str(og_title.get("content", ""))

    # Date
    for selector in [
        "time",
        "meta[property='article:published_time']",
        "span.date",
        "div.date",
        "[datetime]",
    ]:
        date_el = soup.select_one(selector)
        if date_el:
            if date_el.name == "meta":
                meta["date"] = str(date_el.get("content", ""))
            else:
                meta["date"] = str(
                    date_el.get("datetime") or date_el.get_text(strip=True)
                )
            break

    return meta


# ── Main scraper ──────────────────────────────────────────────────────────────


def scrape_outlet(
    cfg: OutletConfig,
    client: httpx.Client,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> list[dict]:
    """Scrape articles from a single outlet. Returns list of result dicts."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scraping {cfg.display_name} ({cfg.dialect_class})...")

    # Discover article URLs
    urls = discover_articles(cfg, client, limit)
    logger.info(f"  Found {len(urls)} article URLs")

    results = []
    scraped_at = datetime.now(timezone.utc).isoformat()

    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(cfg.request_delay)

        try:
            resp = client.get(url, timeout=15, follow_redirects=True)
            if resp.status_code != 200:
                logger.debug(f"  HTTP {resp.status_code}: {url}")
                continue
        except Exception as e:
            logger.debug(f"  Request failed: {url}: {e}")
            continue

        html = resp.text
        text = extract_article_text(html, cfg)
        if not text:
            logger.debug(f"  No text extracted: {url}")
            continue

        meta = extract_article_metadata(html, cfg)

        # Generate article ID from URL
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]

        result = {
            "article_id": url_hash,
            "outlet": cfg.name,
            "outlet_name": cfg.display_name,
            "url": url,
            "title": meta.get("title", ""),
            "text": text,
            "date": meta.get("date", ""),
            "dialect_class": cfg.dialect_class,
            "municipality": cfg.municipality,
            "territory": cfg.territory,
            "scraped_at": scraped_at,
        }
        results.append(result)

        if (i + 1) % 20 == 0:
            logger.info(f"  Scraped {i + 1}/{len(urls)} articles...")

    # Save to JSONL
    if results:
        out_file = output_dir / f"{cfg.name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"  Saved {len(results)} articles → {out_file}")

    return results


def create_client(timeout: int = 30) -> httpx.Client:
    """Create an httpx client with appropriate headers and SSL tolerance."""
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
        verify=SSL_CONTEXT,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def cmd_scrape(
    outlet: str = "berria",
    limit: int = 50,
    output_dir: str | None = None,
) -> None:
    """Scrape articles from one or more outlets.

    Args:
        outlet: Outlet key (e.g., 'berria') or 'all' for all outlets
        limit: Max articles per outlet
        output_dir: Output directory for JSONL files
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    registry = load_outlet_registry()
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    if outlet == "all":
        targets = registry
    elif outlet in registry:
        targets = {outlet: registry[outlet]}
    else:
        available = ", ".join(sorted(registry.keys()))
        logger.error(f"Unknown outlet '{outlet}'. Available: {available}")
        return

    total = 0
    with create_client() as client:
        for key, cfg in targets.items():
            results = scrape_outlet(cfg, client, limit=limit, output_dir=out_dir)
            total += len(results)

    logger.info(f"\nDone. Scraped {total} articles total.")


def cmd_discover() -> None:
    """Test article discovery for all outlets (no scraping)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    registry = load_outlet_registry()
    working = 0
    no_discovery = 0
    fail = 0

    with create_client(timeout=10) as client:
        for key, cfg in sorted(registry.items()):
            urls: list[str] = []

            # Try sitemap
            for sm_url in cfg.sitemap_urls[:1]:
                urls = discover_from_sitemap(sm_url, client, limit=5)
                if urls:
                    break

            # Try RSS
            if not urls:
                for rss_url in cfg.rss_urls[:1]:
                    urls = discover_from_rss(rss_url, client, limit=5)
                    if urls:
                        break

            if urls:
                dialect = cfg.dialect_class
                logger.info(
                    f"  ✓ {cfg.display_name:<35s} {dialect:<10s} {len(urls)} URLs via {'sitemap' if 'sitemap' in str(urls) else 'rss'}"
                )
                working += 1
            elif urls is not None and len(urls) == 0:
                logger.info(
                    f"  - {cfg.display_name:<35s} {cfg.dialect_class:<10s} 0 URLs (empty)"
                )
                no_discovery += 1
            else:
                logger.info(f"  ✗ {cfg.display_name:<35s} {cfg.dialect_class:<10s}")
                fail += 1

            time.sleep(0.3)

    logger.info(
        f"\nWorking: {working}, Empty: {no_discovery}, Failed: {fail}, Total: {len(registry)}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.media_scraper [scrape|discover]")
        sys.exit(1)

    cmd = sys.argv[1]
    kwargs: dict[str, str | int] = {}
    i = 2
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i][2:].replace("-", "_")
            raw = (
                sys.argv[i + 1]
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--")
                else "true"
            )
            # Convert to int if numeric
            try:
                val: str | int = int(raw)
            except ValueError:
                val = raw
            kwargs[key] = val
            i += 2
        else:
            i += 1

    if cmd == "scrape":
        cmd_scrape(**kwargs)  # type: ignore[arg-type]
    elif cmd == "discover":
        cmd_discover()
    else:
        print(f"Unknown command: {cmd}")
