"""
Crawls high-value pages (About, Products/Services, Case Studies, Blog, Careers)
using link classification and per-category caps. Returns (pages, case_study_pages).
"""
import asyncio
import json
import logging
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, unquote

import boto3
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler

from helpers import (
    normalize_website_url,
    clean_markdown_content,
    filter_commercial_urls,
    is_commercially_relevant_url,
)

load_dotenv()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CATEGORY_PATTERNS = {
    "about": re.compile(r"about|about-us|about_us|company|who-we-are|our-story", re.I),
    "products": re.compile(r"product|products", re.I),
    "services": re.compile(r"service|services|solutions|solutions?", re.I),
    "case_studies": re.compile(
        r"case-stud(y|ies)|case_stud(y|ies)|customers?|success-stor(y|ies)|"
        r"resources?|portfolio|client-stories?|implementation[s]?|results?", re.I
    )
}

# Only block non-content file extensions (images, static assets, etc.)
BLOCKLISTED_EXTENSIONS = frozenset({
    "png", "jpg", "jpeg", "gif", "webp", "svg", "ico", "css", "js", "woff", "woff2",
    "ttf", "eot", "pdf", "zip", "xml", "mp4", "mp3", "wav", "avi", "mov",
})

# ---------------------------------------------------------------------------
# Bedrock AI URL filtering
# ---------------------------------------------------------------------------

_bedrock_client = None
DEFAULT_URL_FILTER_MODEL = os.getenv(
    "BEDROCK_URL_FILTER_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
URL_FILTER_PROMPT_PATH = Path(__file__).parent / "prompts" / "url_filtering_prompt.txt"


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    return _bedrock_client


def _load_url_filter_prompt() -> str:
    """Load the URL filtering prompt template from file."""
    if URL_FILTER_PROMPT_PATH.exists():
        return URL_FILTER_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"URL filtering prompt not found: {URL_FILTER_PROMPT_PATH}")


def filter_urls_with_ai(
    urls: list[str],
    max_urls: int = 100,
    model_id: str = DEFAULT_URL_FILTER_MODEL,
) -> list[str]:
    """
    Use AI model to filter URLs based on sales/commercial relevance.
    Returns a list of filtered URLs that are most relevant for sales research.
    """
    if not urls:
        return []

    # If we have very few URLs, return them all
    if len(urls) <= 5:
        return urls

    prompt_template = _load_url_filter_prompt()
    urls_list = "\n".join(urls)
    prompt = prompt_template.format(
        url_count=len(urls),
        max_urls=max_urls,
        urls_list=urls_list,
    )

    client = _get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    })

    try:
        resp = client.invoke_model(modelId=model_id, body=body)
        result = json.loads(resp["body"].read())
        response_text = result["content"][0]["text"].strip()

        # Parse URLs from response (one per line)
        filtered_urls = []
        for line in response_text.split("\n"):
            line = line.strip()
            if line and line.startswith("http"):
                # Only keep URLs that were in the original list
                if line in urls:
                    filtered_urls.append(line)

        logging.info(
            "AI URL filter: %d -> %d URLs (model: %s)",
            len(urls), len(filtered_urls), model_id
        )
        return filtered_urls if filtered_urls else urls[:max_urls]

    except Exception as e:
        logging.warning("AI URL filtering failed, returning original URLs: %s", e)
        return urls[:max_urls]

MAX_PAGES_PER_CATEGORY = 2
MAX_PAGES_CASE_STUDIES = 50
MAX_TOTAL_PAGES = 500
REQUEST_TIMEOUT_MS = 15000
RELATIVE_LINK_PATTERN = re.compile(
    r'(?:href\s*=\s*["\']?|]\s*\(\s*)(/[a-zA-Z0-9][a-zA-Z0-9/._~%-]*)',
    re.I,
)

# -----------------------------------------------------------------------------
# URL and path helpers
# -----------------------------------------------------------------------------


def _path(url: str) -> str:
    """Normalized path for a URL."""
    return (urlparse(url).path or "/").strip() or "/"


def _domain(url: str) -> str:
    """Canonical host for a URL (lowercased, without leading www.)."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _slug_from_url(url: str) -> str:
    """Filesystem-safe slug from URL host."""
    host = _domain(url).replace("www.", "")
    return host.replace(":", "_") or "website"


def _normalize_discovered_url(url: str) -> str:
    """Canonical form for discovered URLs: strip query/fragment and trailing slash (except root)."""
    parsed = urlparse(_normalize_url_fragment(url))
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _markdown_path_for_url(base_dir: Path, url: str) -> Path:
    """Map URL to markdown output path with domain/path hierarchy."""
    parsed = urlparse(url)
    path = parsed.path or "/"
    parts = [p for p in path.split("/") if p]
    host_dir = base_dir / parsed.netloc.replace(":", "_")
    if path == "/" or not parts:
        return host_dir / "index.md"
    if path.endswith("/"):
        return host_dir.joinpath(*parts) / "index.md"
    return host_dir.joinpath(*parts[:-1]) / f"{parts[-1]}.md"


def _json_path_for_url(base_dir: Path, url: str) -> Path:
    """Map URL to JSON output path with domain/path hierarchy."""
    parsed = urlparse(url)
    path = parsed.path or "/"
    parts = [p for p in path.split("/") if p]
    host_dir = base_dir / parsed.netloc.replace(":", "_")
    if path == "/" or not parts:
        return host_dir / "index.json"
    if path.endswith("/"):
        return host_dir.joinpath(*parts) / "index.json"
    return host_dir.joinpath(*parts[:-1]) / f"{parts[-1]}.json"


def _normalize_url_fragment(url: str) -> str:
    """Strip fragment and query from URL string."""
    return url.strip().split("#")[0].split("?")[0].strip()


def _is_blocklisted_path(path: str) -> bool:
    """True if path has a blocklisted file extension (non-content assets)."""
    raw = (path or "").strip()
    decoded = unquote(raw).strip("/").lower()
    parts = decoded.rstrip("/").split("/")
    if parts:
        last = parts[-1]
        ext = last.split(".")[-1].split("?")[0] if "." in last else ""
        if ext in BLOCKLISTED_EXTENSIONS:
            return True
    return False


def _classify_url(path: str) -> Optional[str]:
    """Category key if path matches a high-value pattern, else None."""
    path_lower = (path or "").strip("/").lower()
    for category, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(path_lower):
            return category
    return None


def _is_allowed_same_domain_url(url: str, base_domain: str) -> bool:
    """True if URL is same domain and path is not blocklisted."""
    return _domain(url) == base_domain and not _is_blocklisted_path(_path(url))


def is_useful_same_domain_url(url: str, base_url: str) -> bool:
    """
    True if URL is same-domain and non-blocklisted (no asset extensions).
    Semantic URL filtering is delegated to the AI model via filter_urls_with_ai().
    """
    base_url = normalize_website_url(base_url)
    if not _is_allowed_same_domain_url(url, _domain(base_url)):
        return False
    return True


def _to_absolute_url(href: str, base: str) -> Optional[str]:
    """Resolve href to absolute URL; return None if invalid or not same domain as base."""
    href = _normalize_url_fragment(href)
    if not href or href.startswith(("javascript:", "mailto:")):
        return None
    abs_url = urljoin(base, href)
    return abs_url if _is_allowed_same_domain_url(abs_url, _domain(base)) else None


# -----------------------------------------------------------------------------
# Link collection from content
# -----------------------------------------------------------------------------


def _urls_from_content(content: str, base_url: str) -> set[str]:
    """Extract same-domain, allowed URLs from markdown/HTML (absolute + relative)."""
    base_url = normalize_website_url(base_url)
    domain = _domain(base_url)
    found: set[str] = set()

    for raw in re.findall(r"(https?://[^\s\)\]\"']+)", content):
        u = _normalize_url_fragment(raw)
        if u.startswith("http") and _domain(u) == domain and not _is_blocklisted_path(_path(u)):
            found.add(u)

    for path in RELATIVE_LINK_PATTERN.findall(content):
        path = _normalize_url_fragment(path)
        if not path or path == "/":
            continue
        abs_url = urljoin(base_url, path)
        if _domain(abs_url) == domain and not _is_blocklisted_path(_path(abs_url)):
            found.add(abs_url)

    return found


def get_suburls_from_content(content: str, base_url: str, max_urls: int = 300) -> list[str]:
    """Same-domain links from content; blocklist applied. Category selection is in select_urls_to_crawl."""
    return list(_urls_from_content(content, base_url))[:max_urls]


# -----------------------------------------------------------------------------
# URL selection for crawl
# -----------------------------------------------------------------------------


def _filter_same_domain_allowed(urls: list[str], base_url: str) -> list[str]:
    """Normalize URLs and keep only same-domain, non-blocklisted."""
    base_url = normalize_website_url(base_url)
    base_domain = _domain(base_url)
    out = []
    for u in urls:
        try:
            u = _normalize_url_fragment(u)
            if not u.startswith("http"):
                u = urljoin(base_url, u)
            if _is_allowed_same_domain_url(u, base_domain):
                out.append(u)
        except Exception:
            continue
    return out


def select_urls_to_crawl(
    all_urls: list[str],
    base_url: str,
    max_per_category: int = MAX_PAGES_PER_CATEGORY,
    max_total: int = MAX_TOTAL_PAGES,
    use_ai_filter: bool = True,
    current_iteration: int = 0,
) -> list[str]:
    """
    Select URLs to crawl using AI-based filtering for commercial relevance.
    Homepage is always first. AI filtering starts from iteration 1 onward.
    """
    base_url = normalize_website_url(base_url)
    base_domain = _domain(base_url)
    same_domain = _filter_same_domain_allowed(all_urls, base_url)

    logging.info("Found %d same-domain URLs to filter", len(same_domain))

    seen: set[str] = set()
    selected: list[str] = []
    case_study_list = []

    # Always include homepage first
    if base_url not in seen:
        seen.add(base_url)
        selected.append(base_url)

    # Use AI-based filtering only from iteration 1 onward
    urls_to_filter = [u for u in same_domain if u not in seen]
    if use_ai_filter and current_iteration >= 1 and urls_to_filter:
        filtered_urls = filter_urls_with_ai(urls_to_filter, max_urls=max_total)
        for url in filtered_urls:
            if url not in seen and len(selected) < max_total:
                seen.add(url)
                # Separate case studies
                if _classify_url(_path(url)) == "case_studies":
                    case_study_list.append(url)
                else:
                    selected.append(url)
    else:
        # Fallback to category-based selection
        for url in same_domain:
            if url in seen or len(selected) >= max_total:
                continue
            category = _classify_url(_path(url))
            if category == "case_studies":
                case_study_list.append(url)
            else:
                selected.append(url)
            seen.add(url)

    return selected[:max_total], case_study_list[:MAX_PAGES_CASE_STUDIES]


# -----------------------------------------------------------------------------
# Crawl execution
# -----------------------------------------------------------------------------


def _markdown_from_result(result) -> str:
    """Markdown string from CrawlResult (handles MarkdownGenerationResult or str)."""
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    if hasattr(md, "raw_markdown"):
        return (md.raw_markdown or "").strip()
    return (md or "").strip() if isinstance(md, str) else ""


async def _scrape_one(crawler: AsyncWebCrawler, url: str) -> tuple[str, str]:
    """Return (url, markdown). Empty string on failure; logs exception on failure."""
    try:
        result = await crawler.arun(url=url, timeout=REQUEST_TIMEOUT_MS)
        return (url, _markdown_from_result(result))
    except Exception as e:
        logging.warning("Crawl failed for URL %s: %s", url, e, exc_info=False)
        return (url, "")


async def _scrape_one_with_links(
    crawler: AsyncWebCrawler,
    url: str,
    base_url: str,
    useful_only: bool = True,
) -> tuple[str, str, list[str]]:
    """
    Return (url, markdown, discovered_links).
    discovered_links combines result.internal links and links extracted from markdown.
    """
    try:
        result = await crawler.arun(url=url, timeout=REQUEST_TIMEOUT_MS)
        md = _markdown_from_result(result)
        base = getattr(result, "url", None) or url
        discovered: set[str] = set()

        for link in result.links.get("internal") or []:
            href = link.get("href") or link.get("url") or ""
            abs_url = _to_absolute_url(href, base)
            if not abs_url:
                continue
            if not useful_only or is_useful_same_domain_url(abs_url, base_url):
                discovered.add(abs_url)

        for abs_url in _urls_from_content(md, base_url):
            if not useful_only or is_useful_same_domain_url(abs_url, base_url):
                discovered.add(abs_url)

        return (url, md, list(discovered))
    except Exception as e:
        logging.warning("Crawl failed for URL %s: %s", url, e, exc_info=False)
        return (url, "", [])


async def crawl_multiple_pages(urls: list[str]) -> dict[str, str]:
    """Crawl each URL; return url -> markdown (only non-empty). Failed URLs are logged and skipped."""
    out: dict[str, str] = {}
    if not urls:
        return out
    try:
        async with AsyncWebCrawler() as crawler:
            results = await asyncio.gather(
                *[_scrape_one(crawler, u) for u in urls],
                return_exceptions=True,
            )
        for url, r in zip(urls, results):
            if isinstance(r, Exception):
                logging.warning("Crawl failed for URL %s: %s", url, r, exc_info=False)
                continue
            if isinstance(r, tuple) and len(r) == 2 and r[1]:
                out[r[0]] = r[1]
    except Exception as e:
        logging.exception("crawl_multiple_pages failed: %s", e)
        raise
    return out


async def crawl_pages_with_links(
    urls: list[str],
    base_url: str,
    useful_only: bool = True,
) -> dict[str, dict[str, object]]:
    """
    Crawl each URL and return:
      url -> {"markdown": str, "links": list[str]}
    Only non-empty markdown pages are returned.
    """
    out: dict[str, dict[str, object]] = {}
    if not urls:
        return out
    base_url = normalize_website_url(base_url)
    try:
        async with AsyncWebCrawler() as crawler:
            results = await asyncio.gather(
                *[_scrape_one_with_links(crawler, u, base_url, useful_only=useful_only) for u in urls],
                return_exceptions=True,
            )
        for url, r in zip(urls, results):
            if isinstance(r, Exception):
                logging.warning("Crawl failed for URL %s: %s", url, r, exc_info=False)
                continue
            if isinstance(r, tuple) and len(r) == 3 and r[1]:
                out[r[0]] = {"markdown": r[1], "links": r[2]}
    except Exception as e:
        logging.exception("crawl_pages_with_links failed: %s", e)
        raise
    return out


async def _fetch_homepage_for_discovery(
    crawler: AsyncWebCrawler, website_url: str
) -> tuple[str, list[str]]:
    """Fetch homepage; return (markdown, list of same-domain links from result.links + markdown)."""
    website_url = normalize_website_url(website_url)
    try:
        result = await crawler.arun(url=website_url, timeout=REQUEST_TIMEOUT_MS)
    except Exception:
        return ("", [])

    md = _markdown_from_result(result)
    base = getattr(result, "url", None) or website_url
    found: set[str] = set()

    for link in result.links.get("internal") or []:
        href = link.get("href") or link.get("url") or ""
        abs_url = _to_absolute_url(href, base)
        if abs_url:
            found.add(abs_url)

    found.update(_urls_from_content(md, website_url))
    return md, list(found)


async def _crawl_from_links(
    website_url: str,
    links: list[str],
    homepage_md: str = "",
) -> dict[str, str]:
    """Build to_crawl from links, crawl, ensure homepage in result."""
    to_crawl, case_study_list = select_urls_to_crawl([website_url] + links, website_url)
    pages = await crawl_multiple_pages(to_crawl) if to_crawl else {}
    if website_url not in pages and homepage_md:
        pages[website_url] = homepage_md

    case_study_pages = await crawl_multiple_pages(case_study_list) if case_study_list else {}
    if website_url not in case_study_pages and homepage_md:
        case_study_pages[website_url] = homepage_md
    return pages, case_study_pages


async def _discover_and_crawl(website_url: str) -> dict[str, str]:
    """Fetch homepage for links, then crawl selected URLs (used when no pre-provided markdown)."""
    async with AsyncWebCrawler() as crawler:
        homepage_md, links = await _fetch_homepage_for_discovery(crawler, website_url)
    return await _crawl_from_links(website_url, links, homepage_md)


def _content_fingerprint(text: str) -> str:
    """Cheap fingerprint for near-duplicate detection: lowered, whitespace-collapsed."""
    import hashlib
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(collapsed.encode("utf-8")).hexdigest()


async def crawl_by_iterations(
    website_url: str,
    iterations: int = 2,
    max_pages: int = 2000,
    useful_only: bool = True,
    apply_commercial_filter: bool = True,
    use_ai_url_filter: bool = True,
    clean_markdown: bool = True,
    deduplicate: bool = True,
) -> dict[str, Any]:
    """
    Crawl in iterative waves.
    Iteration 0 crawls the homepage. Iteration N crawls links found at N-1.

    apply_commercial_filter: at iteration >= 1, only enqueue commercially relevant URLs.
    use_ai_url_filter: use AI model to filter URLs based on sales/commercial relevance.
    clean_markdown: run clean_markdown_content on every page's markdown.
    deduplicate: skip pages whose content fingerprint was already seen.
    """
    root = _normalize_discovered_url(normalize_website_url(website_url))
    max_iterations = max(0, iterations)
    max_pages = max(1, max_pages)

    queue: deque[tuple[str, int]] = deque([(root, 0)])
    discovered: set[str] = {root}
    page_iterations: dict[str, int] = {}
    pages: dict[str, str] = {}
    links_by_page: dict[str, list[str]] = {}
    edges: list[dict[str, str]] = []
    iteration_batches: dict[int, list[str]] = {}
    seen_fingerprints: set[str] = set()
    skipped_duplicates: list[str] = []
    filtered_out: dict[int, list[str]] = {}

    while queue and len(pages) < max_pages:
        current_iteration = queue[0][1]
        if current_iteration > max_iterations:
            break

        batch: list[str] = []
        while (
            queue
            and queue[0][1] == current_iteration
            and len(pages) + len(batch) < max_pages
        ):
            url, iteration = queue.popleft()
            batch.append(url)
            page_iterations[url] = iteration

        iteration_batches[current_iteration] = batch
        batch_results = await crawl_pages_with_links(batch, base_url=root, useful_only=useful_only)

        for crawled_url, payload in batch_results.items():
            markdown = payload.get("markdown", "")
            discovered_links = payload.get("links", [])

            if markdown:
                if clean_markdown:
                    markdown = clean_markdown_content(markdown)

                if deduplicate:
                    fp = _content_fingerprint(markdown)
                    if fp in seen_fingerprints:
                        skipped_duplicates.append(crawled_url)
                        markdown = ""
                    else:
                        seen_fingerprints.add(fp)

            if markdown:
                pages[crawled_url] = markdown

            normalized_links: list[str] = []
            for next_url in discovered_links:
                normalized_next = _normalize_discovered_url(next_url)
                normalized_links.append(normalized_next)
                edges.append({"from": crawled_url, "to": normalized_next})

            links_to_enqueue = sorted(set(normalized_links))

            # Apply URL filtering at iteration >= 1
            if current_iteration >= 1 and links_to_enqueue:
                before_count = len(links_to_enqueue)

                if use_ai_url_filter:
                    # Use AI model to filter URLs
                    links_to_enqueue = filter_urls_with_ai(
                        links_to_enqueue,
                        max_urls=min(100, max_pages - len(pages))
                    )
                elif apply_commercial_filter:
                    # Fallback to rule-based commercial filter
                    links_to_enqueue = filter_commercial_urls(links_to_enqueue)

                rejected = before_count - len(links_to_enqueue)
                if rejected:
                    filtered_out.setdefault(current_iteration, []).extend(
                        [u for u in normalized_links if u not in links_to_enqueue]
                    )

            for next_url in links_to_enqueue:
                if next_url not in discovered and current_iteration < max_iterations:
                    discovered.add(next_url)
                    queue.append((next_url, current_iteration + 1))

            links_by_page[crawled_url] = sorted(set(normalized_links))

    iteration_summary: list[dict[str, Any]] = []
    for i in range(0, max_iterations + 1):
        urls_for_iteration = iteration_batches.get(i, [])
        iteration_summary.append(
            {
                "iteration": i,
                "urls_crawled": len(urls_for_iteration),
                "urls": urls_for_iteration,
            }
        )

    return {
        "website_url": root,
        "iterations": max_iterations,
        "max_pages": max_pages,
        "useful_only": useful_only,
        "apply_commercial_filter": apply_commercial_filter,
        "use_ai_url_filter": use_ai_url_filter,
        "clean_markdown": clean_markdown,
        "deduplicate": deduplicate,
        "total_pages_crawled": len(pages),
        "skipped_duplicates": skipped_duplicates,
        "filtered_out_by_iteration": {str(k): v for k, v in filtered_out.items()},
        "pages": pages,
        "links_by_page": links_by_page,
        "page_iterations": page_iterations,
        "crawl_structure": {
            "root": root,
            "iteration_summary": iteration_summary,
            "edges": edges,
        },
    }


def save_iterative_crawl_output(
    crawl_result: dict[str, Any],
    output_dir: str = "crawl_output",
) -> Path:
    """
    Save crawl output with timestamped run directory and hierarchical URL-based files.
    """
    website_url = crawl_result["website_url"]
    pages = crawl_result.get("pages", {})
    links_by_page = crawl_result.get("links_by_page", {})
    page_iterations = crawl_result.get("page_iterations", {})
    crawl_structure = crawl_result.get("crawl_structure", {})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{_slug_from_url(website_url)}_{timestamp}"
    pages_dir = run_dir / "pages_markdown"
    links_dir = run_dir / "links"
    metadata_dir = run_dir / "metadata"
    iterations_dir = run_dir / "iterations"

    for directory in (pages_dir, links_dir, metadata_dir, iterations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    url_to_markdown_file: dict[str, str] = {}
    url_to_links_file: dict[str, str] = {}

    for url, markdown in pages.items():
        md_path = _markdown_path_for_url(pages_dir, url)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = f"# Source URL\n{url}\n\n---\n\n{markdown}".strip() + "\n"
        md_path.write_text(md_content, encoding="utf-8")
        url_to_markdown_file[url] = str(md_path.relative_to(run_dir)).replace("\\", "/")

    for url, links in links_by_page.items():
        links_path = _json_path_for_url(links_dir, url)
        links_path.parent.mkdir(parents=True, exist_ok=True)
        links_payload = {
            "source_url": url,
            "iteration": page_iterations.get(url),
            "links": links,
        }
        links_path.write_text(json.dumps(links_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        url_to_links_file[url] = str(links_path.relative_to(run_dir)).replace("\\", "/")

    iteration_summary = crawl_structure.get("iteration_summary", [])
    for item in iteration_summary:
        iteration_idx = int(item.get("iteration", 0))
        iter_file = iterations_dir / f"iter_{iteration_idx:03d}.json"
        iter_file.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "website_url": website_url,
        "iterations": crawl_result.get("iterations"),
        "max_pages": crawl_result.get("max_pages"),
        "useful_only": crawl_result.get("useful_only"),
        "apply_commercial_filter": crawl_result.get("apply_commercial_filter"),
        "use_ai_url_filter": crawl_result.get("use_ai_url_filter"),
        "clean_markdown": crawl_result.get("clean_markdown"),
        "deduplicate": crawl_result.get("deduplicate"),
        "total_pages_crawled": crawl_result.get("total_pages_crawled"),
        "skipped_duplicates": crawl_result.get("skipped_duplicates", []),
        "pages": [
            {
                "url": url,
                "iteration": page_iterations.get(url),
                "markdown_file": url_to_markdown_file.get(url),
                "links_file": url_to_links_file.get(url),
            }
            for url in pages
        ],
    }

    (metadata_dir / "crawl_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "crawl_structure.json").write_text(
        json.dumps(crawl_structure, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "page_links.json").write_text(
        json.dumps(links_by_page, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    filtered_out = crawl_result.get("filtered_out_by_iteration", {})
    if filtered_out:
        (metadata_dir / "filtered_out_urls.json").write_text(
            json.dumps(filtered_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return run_dir

def remove_common_sections_from_pages(md_pages):

    first_page = md_pages[0]
    remaining_pages = md_pages[1:]

    all_common_sections = []

    common_section = ""
    for each_line in first_page.split("\n"):
        
        if [common_section + each_line in page for page in remaining_pages].count(True) > 2:
            common_section += each_line + "\n"
        else:
            if common_section.lstrip().rstrip() and len(common_section.lstrip().rstrip().split("\n")) > 2:
                all_common_sections.append(common_section.strip())
                # print("Identified common section:\n", common_section)
            common_section = ""
            if [common_section + each_line in page for page in remaining_pages].count(True) > 2:
                common_section += each_line + "\n"
    
    if common_section.lstrip().rstrip() and len(common_section.lstrip().rstrip().split("\n")) > 2:
        # print("Final common section identified at end of page:\n", common_section)
        all_common_sections.append(common_section.strip())

    for i, each_page in enumerate(md_pages):

        updated_page = each_page   # <-- track updates

        for common_sec in all_common_sections:
            cleaned_sec = common_sec.strip()

            if cleaned_sec in updated_page:
                # print(f"\n\nRemoving common section from page {i+1}:\n", cleaned_sec)
                updated_page = updated_page.replace(cleaned_sec, "")
            # else:
            #     print(f"\n\nCommon section not found in page {i+1}:\n", cleaned_sec)

        md_pages[i] = updated_page   # <-- update once after loop
    
    return md_pages


def run_multi_page_crawl(
    website_url: str,
    homepage_markdown: Optional[str] = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Crawl site and return (pages, case_study_pages).
    If homepage_markdown is provided, links are taken from it and not re-fetched.
    """
    website_url = normalize_website_url(website_url)

    if homepage_markdown is not None:
        links = get_suburls_from_content(homepage_markdown, website_url, max_urls=300)
        pages, case_study_pages = asyncio.run(_crawl_from_links(website_url, links, homepage_markdown))
    else:
        pages, case_study_pages = asyncio.run(_discover_and_crawl(website_url))

    return pages, case_study_pages


async def _test_url_selection_for_site(website_url: str) -> None:
    """Print selected URLs per category for a given site."""
    website_url = normalize_website_url(website_url)
    async with AsyncWebCrawler() as crawler:
        _, all_links = await _fetch_homepage_for_discovery(crawler, website_url)
    to_crawl, case_study_list = select_urls_to_crawl([website_url] + all_links, website_url)
    print(f"\n{'='*70}\n  {website_url}\n{'='*70}")
    print(f"  Links (after blocklist): {len(all_links)}  Selected (max {MAX_TOTAL_PAGES}): {len(to_crawl)}\n  ---")
    for i, u in enumerate(to_crawl, 1):
        cat = _classify_url(_path(u)) or "(uncategorized)"
        print(f"  {i:2}. [{cat:12}] {u}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Iterative website crawler with hierarchical output.")
    parser.add_argument("website_url", nargs="?", default="https://www.fissionlabs.com/")
    parser.add_argument("--iterations", type=int, default=2, help="How many crawl iterations to run")
    parser.add_argument("--max-pages", type=int, default=2000, help="Safety cap for pages to crawl")
    parser.add_argument(
        "--all-internal-urls",
        action="store_true",
        help="Include utility/auth/legal/filter URLs",
    )
    parser.add_argument(
        "--output-dir",
        default="crawl_output",
        help="Base output directory for crawl runs",
    )
    parser.add_argument(
        "--no-commercial-filter",
        action="store_true",
        help="Disable commercial-relevance URL filter on iteration >= 1 links",
    )
    parser.add_argument(
        "--no-ai-url-filter",
        action="store_true",
        help="Disable AI-based URL filtering (uses rule-based filtering instead)",
    )
    parser.add_argument(
        "--no-clean-markdown",
        action="store_true",
        help="Disable markdown cleaning (image/link removal, whitespace collapse)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable content de-duplication across pages",
    )
    args = parser.parse_args()

    result = asyncio.run(
        crawl_by_iterations(
            website_url=args.website_url,
            iterations=max(0, args.iterations),
            max_pages=max(1, args.max_pages),
            useful_only=not args.all_internal_urls,
            apply_commercial_filter=not args.no_commercial_filter,
            use_ai_url_filter=not args.no_ai_url_filter,
            clean_markdown=not args.no_clean_markdown,
            deduplicate=not args.no_dedup,
        )
    )
    run_path = save_iterative_crawl_output(result, output_dir=args.output_dir)
    dupes = len(result.get("skipped_duplicates", []))
    filtered = sum(len(v) for v in result.get("filtered_out_by_iteration", {}).values())
    print(
        f"Crawl completed.\n"
        f"  Pages saved   : {result['total_pages_crawled']}\n"
        f"  Iterations    : {result['iterations']}\n"
        f"  Duplicates    : {dupes} skipped\n"
        f"  Filtered out  : {filtered} non-commercial URLs\n"
        f"  Output        : {run_path}"
    )
