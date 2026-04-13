"""
Generate a hierarchical page index tree from crawled website data.

The tree mirrors the URL path structure of the crawled site, with each page
further decomposed into its markdown heading hierarchy. Bedrock (Claude) is
used to produce concise summaries for every node.

Usage:
    python generate_page_index.py crawl_output/fissionlabs.com_20260410_105854
    python generate_page_index.py crawl_output/fissionlabs.com_20260410_105854 --no-summaries
    python generate_page_index.py crawl_output/fissionlabs.com_20260410_105854 --model-id anthropic.claude-3-haiku-20240307-v1:0
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
load_dotenv()

import boto3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bedrock helper
# ---------------------------------------------------------------------------

_bedrock_client = None

SUMMARY_PROMPT = (
    "You are a technical writer. Given the following markdown content from a "
    "web page section, write a concise 1-2 sentence summary (max 120 chars) "
    "that captures the key purpose or topic of this section. "
    "Return ONLY the summary text, nothing else.\n\n"
    "---\n{content}\n---"
)

DEFAULT_SUMMARY_MODEL = os.getenv(
    # "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    "BEDROCK_MODEL_ID", "meta.llama4-maverick-17b-instruct-v1:0"
)
SUMMARY_MAX_TOKENS = 512
SUMMARY_TEMPERATURE = 0.0


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    return _bedrock_client


def summarize_with_bedrock(
    content: str,
    model_id: str = DEFAULT_SUMMARY_MODEL,
) -> str:
    """Call Bedrock Model to produce a short summary of *content*."""
    if not content or not content.strip():
        return ""

    # truncated = content[:3000]
    truncated = content
    prompt = SUMMARY_PROMPT.format(content=truncated)

    client = _get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": SUMMARY_MAX_TOKENS,
        "temperature": SUMMARY_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    })

    try:
        resp = client.invoke_model(modelId=model_id, body=body)
        result = json.loads(resp["body"].read())
        return result["content"][0]["text"].strip()
    except Exception:
        logger.warning("Bedrock summarization failed for chunk, skipping", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Section:
    """A markdown heading and the content below it, before the next heading."""
    level: int
    title: str
    content: str


def _clean_title(raw: str) -> str:
    """Remove markdown image syntax and trailing whitespace from a title."""
    cleaned = _IMAGE_MD_PATTERN.sub("", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


_SKIP_TITLES = frozenset({
    "thank you! your submission has been received!",
    "404",
    "page not found",
})


def extract_title_from_markdown(md: str) -> str:
    """Return the first H1 that isn't the 'Source URL' header or nav boilerplate."""
    for m in _HEADING_RE.finditer(md):
        if len(m.group(1)) == 1:
            title = _clean_title(m.group(2))
            if not title:
                continue
            if title.lower().startswith("source url"):
                continue
            if title.lower() in _SKIP_TITLES:
                continue
            return title
    return ""


def is_error_page(md: str) -> bool:
    """Return True if the page content looks like a 404 / error page."""
    title = extract_title_from_markdown(md)
    if title.lower() in _SKIP_TITLES:
        return True
    lower = md.lower()
    if "404" in lower[:500] and ("not found" in lower[:500] or "page not found" in lower[:500]):
        return True
    return False


def parse_sections(md: str) -> list[Section]:
    """Split markdown into heading-delimited sections."""
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return []

    sections: list[Section] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = _clean_title(m.group(2))
        if not title:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        content = md[start:end].strip()
        sections.append(Section(level=level, title=title, content=content))
    return sections


# ---------------------------------------------------------------------------
# Boilerplate detection and removal
# ---------------------------------------------------------------------------

_NAV_BOILERPLATE_TITLES = frozenset({
    "source url",
    "innovation  \naccelerator",
    "innovation accelerator",
    "book a free consultation to unleash the full potential of digital transformation",
    "thank you! your submission has been received!",
})

_NAV_BOILERPLATE_SUBSTRINGS = (
    "transforming database querying",
    "view file in another tab",
    "oops! something went wrong",
    "fission labs uses cookies",
)

_FOOTER_PATTERN = re.compile(
    r"^(company|services|resources|contact us|©)",
    re.I,
)

_IMAGE_MD_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")


def _is_boilerplate_section(section: Section) -> bool:
    title_lower = section.title.lower().strip()
    if title_lower in _NAV_BOILERPLATE_TITLES:
        return True
    if _FOOTER_PATTERN.match(title_lower):
        return True
    if "![" in section.title:
        return True
    for substr in _NAV_BOILERPLATE_SUBSTRINGS:
        if substr in title_lower:
            return True
    content_lower = section.content.lower()
    if "© " in content_lower and "all rights reserved" in content_lower:
        return True
    if not section.content.strip() and not section.title.strip():
        return True
    return False


def filter_boilerplate(sections: list[Section]) -> list[Section]:
    """Remove common nav/footer/boilerplate sections."""
    return [s for s in sections if not _is_boilerplate_section(s)]


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class IndexNode:
    title: str
    node_id: str = ""
    url: str = ""
    summary: str = ""
    content_preview: str = ""
    children: list[IndexNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"title": self.title, "node_id": self.node_id}
        if self.url:
            d["url"] = self.url
        if self.summary:
            d["summary"] = self.summary
        if self.content_preview:
            d["content_preview"] = self.content_preview
        if self.children:
            d["nodes"] = [c.to_dict() for c in self.children]
        return d


# ---------------------------------------------------------------------------
# Build section sub-tree for a single page
# ---------------------------------------------------------------------------

def _build_section_tree(sections: list[Section], page_url: str = "") -> list[IndexNode]:
    """
    Convert flat list of sections (with heading levels) into a nested tree.
    Each section becomes an IndexNode; children are determined by heading level.
    The page_url is propagated to all section nodes.
    """
    if not sections:
        return []

    root_nodes: list[IndexNode] = []
    stack: list[tuple[int, IndexNode]] = []

    for sec in sections:
        node = IndexNode(
            title=sec.title,
            url=page_url,
            content_preview=sec.content[:200].strip() if sec.content else "",
        )
        while stack and stack[-1][0] >= sec.level:
            stack.pop()
        if stack:
            stack[-1][1].children.append(node)
        else:
            root_nodes.append(node)
        stack.append((sec.level, node))

    return root_nodes


# ---------------------------------------------------------------------------
# Build URL path tree
# ---------------------------------------------------------------------------

def _url_path_segments(url: str) -> list[str]:
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    return segments


def _build_url_tree(
    pages: list[dict],
    crawl_dir: Path,
) -> IndexNode:
    """
    Build a tree where the root is the website homepage and children are
    nested by URL path segments. Each page's internal heading structure
    is attached as deeper children.
    """
    root_url = ""
    for p in pages:
        segs = _url_path_segments(p["url"])
        if not segs:
            root_url = p["url"]
            break

    root = IndexNode(title="", url=root_url or pages[0]["url"] if pages else "")

    path_node_map: dict[str, IndexNode] = {"": root}

    sorted_pages = sorted(pages, key=lambda p: _url_path_segments(p["url"]))

    for page_info in sorted_pages:
        segments = _url_path_segments(page_info["url"])
        if not segments:
            _populate_page_node(root, page_info, crawl_dir)
            continue

        parent_key = ""
        for i, seg in enumerate(segments[:-1]):
            key = "/".join(segments[: i + 1])
            if key not in path_node_map:
                intermediate = IndexNode(
                    title=seg.replace("-", " ").replace("_", " ").title(),
                    url="",
                )
                path_node_map[parent_key].children.append(intermediate)
                path_node_map[key] = intermediate
            parent_key = key

        full_key = "/".join(segments)

        if full_key in path_node_map:
            existing = path_node_map[full_key]
            existing.url = page_info["url"]
            _populate_page_node(existing, page_info, crawl_dir)
        else:
            page_node = IndexNode(title="", url=page_info["url"])
            ok = _populate_page_node(page_node, page_info, crawl_dir)
            if ok:
                path_node_map[parent_key].children.append(page_node)
                path_node_map[full_key] = page_node

    return root


_SOURCE_URL_HEADER_RE = re.compile(
    r"^#\s*Source URL\s*\n.*?\n---\s*\n",
    re.MULTILINE | re.DOTALL,
)

_NAV_BLOCK_RE = re.compile(
    r"^(Company|Services|Resources|Latest Blog|"
    r"###\s+Transforming Database Querying.*?$)\s*\n?",
    re.MULTILINE,
)


def _clean_markdown_for_preview(md: str) -> str:
    """Strip the source-URL header and nav lines inserted by the crawler."""
    md = _SOURCE_URL_HEADER_RE.sub("", md, count=1)
    md = _NAV_BLOCK_RE.sub("", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _populate_page_node(
    node: IndexNode,
    page_info: dict,
    crawl_dir: Path,
    max_section_depth: int = 3,
) -> bool:
    """
    Read the markdown file for a page and fill in title + section children.
    Returns False if the page should be excluded (e.g. 404).
    """
    md_path = crawl_dir / page_info["markdown_file"]
    if not md_path.exists():
        logger.warning("Markdown file not found: %s", md_path)
        return False

    md_content = md_path.read_text(encoding="utf-8")

    if is_error_page(md_content):
        return False

    title = extract_title_from_markdown(md_content)
    if title and not node.title:
        node.title = title
    elif not node.title:
        segments = _url_path_segments(page_info["url"])
        node.title = (
            segments[-1].replace("-", " ").replace("_", " ").title()
            if segments
            else "Home"
        )

    raw_sections = parse_sections(md_content)
    cleaned = filter_boilerplate(raw_sections)

    page_title_lower = node.title.strip().lower()
    cleaned = [
        s for s in cleaned
        if s.title.strip().lower() != page_title_lower
    ]

    # Limit section nesting depth: only keep headings within max_section_depth
    # levels below the shallowest heading found
    if cleaned:
        min_level = min(s.level for s in cleaned)
        cleaned = [s for s in cleaned if s.level <= min_level + max_section_depth - 1]

    section_nodes = _build_section_tree(cleaned, page_url=page_info["url"])
    node.children.extend(section_nodes)

    clean_md = _clean_markdown_for_preview(md_content)
    if clean_md:
        node.content_preview = clean_md[:300].strip()

    return True


# ---------------------------------------------------------------------------
# Assign node IDs and generate summaries
# ---------------------------------------------------------------------------

def _assign_ids(node: IndexNode, counter: list[int] | None = None) -> None:
    if counter is None:
        counter = [0]
    node.node_id = f"{counter[0]:04d}"
    counter[0] += 1
    for child in node.children:
        _assign_ids(child, counter)


def _generate_summaries(
    node: IndexNode,
    model_id: str,
    progress: list[int] | None = None,
    total: list[int] | None = None,
) -> None:
    if progress is None:
        progress = [0]
    if total is None:
        total = [_count_nodes(node)]

    text_for_summary = node.content_preview or node.title
    if text_for_summary:
        node.summary = summarize_with_bedrock(text_for_summary, model_id=model_id)

    progress[0] += 1
    if progress[0] % 5 == 0 or progress[0] == total[0]:
        logger.info("Summaries: %d / %d", progress[0], total[0])

    for child in node.children:
        _generate_summaries(child, model_id, progress, total)


def _count_nodes(node: IndexNode) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_page_index(
    crawl_dir: str | Path,
    *,
    generate_summaries: bool = True,
    model_id: str = DEFAULT_SUMMARY_MODEL,
) -> dict[str, Any]:
    """
    Build the full page index tree from a crawl output directory.

    Returns the tree as a nested dict ready for JSON serialization.
    """
    crawl_dir = Path(crawl_dir)
    manifest_path = crawl_dir / "metadata" / "crawl_manifest.json"
    if not manifest_path.exists():
        manifest_path = crawl_dir / "crawl_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No crawl_manifest.json found in {crawl_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = manifest["pages"]

    logger.info("Building page index for %d pages ...", len(pages))

    root = _build_url_tree(pages, crawl_dir)

    _assign_ids(root)

    total = _count_nodes(root)
    logger.info("Tree built with %d nodes", total)

    if generate_summaries:
        logger.info("Generating summaries via Bedrock (%s) ...", model_id)
        start = time.time()
        _generate_summaries(root, model_id)
        elapsed = time.time() - start
        logger.info("Summaries completed in %.1fs", elapsed)

    tree_dict = root.to_dict()

    output_path = crawl_dir / "page_index.json"
    output_path.write_text(
        json.dumps(tree_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Page index written to %s", output_path)

    return tree_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hierarchical page index from crawled website data."
    )
    parser.add_argument(
        "crawl_dir",
        help="Path to the crawl output directory (containing metadata/crawl_manifest.json)",
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip Bedrock summary generation (useful for quick structural preview)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_SUMMARY_MODEL,
        help="Bedrock model ID for summary generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    args = _parse_args()
    tree = generate_page_index(
        args.crawl_dir,
        generate_summaries=not args.no_summaries,
        model_id=args.model_id,
    )
    def _tree_summary(node_dict, depth=0, lines=None):
        if lines is None:
            lines = []
        indent = "  " * depth
        title = node_dict["title"][:60]
        nid = node_dict["node_id"]
        url_str = f'  [{node_dict["url"]}]' if node_dict.get("url") else ""
        summary_str = f'  — {node_dict["summary"][:80]}' if node_dict.get("summary") else ""
        lines.append(f"{indent}{nid}: {title}{url_str}{summary_str}")
        for child in node_dict.get("nodes", []):
            _tree_summary(child, depth + 1, lines)
        return lines

    lines = _tree_summary(tree)
    preview = "\n".join(lines[:80])
    print(preview)
    if len(lines) > 80:
        print(f"\n... and {len(lines) - 80} more nodes")
    print(f"\nTotal nodes: {len(lines)}")
    print(f"Full output at: {Path(args.crawl_dir) / 'page_index.json'}")
