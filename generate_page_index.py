"""
Generate a hierarchical page index tree from crawled website data.

The tree mirrors the URL path structure of the crawled site. Each node represents
a page (not individual sections/headings). Bedrock is used to produce concise
summaries for every page node.

Usage:
    python generate_page_index.py crawl_output/fissionlabs.com_20260410_105854
    python generate_page_index.py crawl_output/fissionlabs.com_20260410_105854 --no-summaries
    python generate_page_index.py crawl_output/fissionlabs.com_20260410_105854 --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bedrock helper
# ---------------------------------------------------------------------------

_bedrock_client = None
_failed_summary_models: set[str] = set()

SUMMARY_PROMPT = (
    "You are a technical writer. Given the following markdown content from a "
    "web page, write a concise summary "
    "that captures the key purpose or topic of this page. "
    "Return ONLY the summary text, nothing else.\n\n"
    "Page Title: {title}\n\n"
    "---\n{content}\n---"
)

DEFAULT_SUMMARY_MODEL = os.getenv(
    "SUMMARY_BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
SUMMARY_MAX_TOKENS = 512
SUMMARY_TEMPERATURE = 0.0
MAX_PARALLEL_SUMMARIES = int(os.getenv("MAX_PARALLEL_SUMMARIES", "10"))


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    return _bedrock_client


def summarize_with_bedrock(
    title: str,
    content: str,
    model_id: str = DEFAULT_SUMMARY_MODEL,
) -> str:
    """Call Bedrock Model to produce a short summary of page content."""
    if not content or not content.strip():
        return ""

    if model_id in _failed_summary_models:
        return ""

    truncated = content[:4000]
    prompt = SUMMARY_PROMPT.format(title=title, content=truncated)

    client = _get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": SUMMARY_MAX_TOKENS,
        "temperature": SUMMARY_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    })

    candidate_model_ids = [model_id]
    if model_id.startswith("anthropic."):
        candidate_model_ids.append(f"us.{model_id}")

    last_error: Exception | None = None
    for candidate in candidate_model_ids:
        try:
            resp = client.invoke_model(modelId=candidate, body=body)
            result = json.loads(resp["body"].read())
            text = result.get("content", [{}])[0].get("text", "")
            if text:
                return text.strip()
        except ClientError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue

    _failed_summary_models.add(model_id)
    logger.warning(
        "Bedrock summarization disabled for model '%s' after failure. "
        "Use an inference-profile model ID (e.g. us.anthropic....). Error: %s",
        model_id,
        last_error,
    )
    return ""


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_IMAGE_MD_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")

_SKIP_TITLES = frozenset({
    "thank you! your submission has been received!",
    "404",
    "page not found",
})


def _clean_title(raw: str) -> str:
    """Remove markdown image syntax and trailing whitespace from a title."""
    cleaned = _IMAGE_MD_PATTERN.sub("", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


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


_SOURCE_URL_HEADER_RE = re.compile(
    r"^#\s*Source URL\s*\n.*?\n---\s*\n",
    re.MULTILINE | re.DOTALL,
)


def _clean_markdown_for_summary(md: str) -> str:
    """Strip the source-URL header for summary generation."""
    md = _SOURCE_URL_HEADER_RE.sub("", md, count=1)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class IndexNode:
    title: str
    node_id: str = ""
    url: str = ""
    markdown_file: str = ""
    summary: str = ""
    children: list[IndexNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"title": self.title, "node_id": self.node_id}
        if self.url:
            d["url"] = self.url
        if self.markdown_file:
            d["markdown_file"] = self.markdown_file
        if self.summary:
            d["summary"] = self.summary
        if self.children:
            d["nodes"] = [c.to_dict() for c in self.children]
        return d


# ---------------------------------------------------------------------------
# Build URL path tree (one node per page)
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
    nested by URL path segments. Each node represents a single page.
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


def _populate_page_node(
    node: IndexNode,
    page_info: dict,
    crawl_dir: Path,
) -> bool:
    """
    Read the markdown file for a page and fill in title + markdown_file path.
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

    node.markdown_file = page_info["markdown_file"]
    return True


# ---------------------------------------------------------------------------
# Assign node IDs and generate summaries (parallel)
# ---------------------------------------------------------------------------

def _assign_ids(node: IndexNode, counter: list[int] | None = None) -> None:
    if counter is None:
        counter = [0]
    node.node_id = f"{counter[0]:04d}"
    counter[0] += 1
    for child in node.children:
        _assign_ids(child, counter)


def _collect_nodes_for_summary(node: IndexNode, nodes_list: list[IndexNode]) -> None:
    """Collect all nodes that need summaries (have markdown_file)."""
    if node.markdown_file:
        nodes_list.append(node)
    for child in node.children:
        _collect_nodes_for_summary(child, nodes_list)


def _generate_single_summary(
    node: IndexNode,
    crawl_dir: Path,
    model_id: str,
) -> tuple[str, str]:
    """Generate summary for a single node. Returns (node_id, summary)."""
    if not node.markdown_file:
        return (node.node_id, "")

    md_path = crawl_dir / node.markdown_file
    if not md_path.exists():
        return (node.node_id, "")

    md_content = md_path.read_text(encoding="utf-8")
    clean_content = _clean_markdown_for_summary(md_content)

    summary = summarize_with_bedrock(node.title, clean_content, model_id=model_id)
    return (node.node_id, summary)


def _generate_summaries_parallel(
    root: IndexNode,
    crawl_dir: Path,
    model_id: str,
    max_workers: int = MAX_PARALLEL_SUMMARIES,
) -> None:
    """Generate summaries for all nodes in parallel using ThreadPoolExecutor."""
    nodes_list: list[IndexNode] = []
    _collect_nodes_for_summary(root, nodes_list)

    if not nodes_list:
        return

    node_id_to_node = {n.node_id: n for n in nodes_list}
    total = len(nodes_list)
    completed = 0

    logger.info("Generating summaries for %d pages (max %d parallel)...", total, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_single_summary, node, crawl_dir, model_id): node.node_id
            for node in nodes_list
        }

        for future in as_completed(futures):
            node_id = futures[future]
            try:
                _, summary = future.result()
                if summary:
                    node_id_to_node[node_id].summary = summary
            except Exception as e:
                logger.warning("Summary generation failed for node %s: %s", node_id, e)

            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.info("Summaries: %d / %d", completed, total)


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
    max_parallel: int = MAX_PARALLEL_SUMMARIES,
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
    logger.info("Tree built with %d nodes (1 per page)", total)

    if generate_summaries:
        logger.info("Generating summaries via Bedrock (%s) ...", model_id)
        start = time.time()
        _generate_summaries_parallel(root, crawl_dir, model_id, max_workers=max_parallel)
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
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL_SUMMARIES,
        help=f"Maximum parallel summary requests (default: {MAX_PARALLEL_SUMMARIES})",
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
        max_parallel=args.max_parallel,
    )

    def _tree_summary(node_dict, depth=0, lines=None):
        if lines is None:
            lines = []
        indent = "  " * depth
        title = node_dict["title"][:60]
        nid = node_dict["node_id"]
        url_str = f'  [{node_dict["url"]}]' if node_dict.get("url") else ""
        md_file = f'  -> {node_dict["markdown_file"]}' if node_dict.get("markdown_file") else ""
        summary_str = f'  — {node_dict["summary"][:80]}' if node_dict.get("summary") else ""
        lines.append(f"{indent}{nid}: {title}{url_str}{md_file}{summary_str}")
        for child in node_dict.get("nodes", []):
            _tree_summary(child, depth + 1, lines)
        return lines

    lines = _tree_summary(tree)
    preview = "\n".join(lines[:50])
    print(preview)
    if len(lines) > 50:
        print(f"\n... and {len(lines) - 50} more nodes")
    print(f"\nTotal nodes: {len(lines)}")
    print(f"Full output at: {Path(args.crawl_dir) / 'page_index.json'}")
