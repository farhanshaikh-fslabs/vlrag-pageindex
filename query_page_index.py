"""
Tree-based retrieval engine for crawled website data.

Uses a two-stage approach:
  1. LLM-guided tree search  — a reasoning model navigates the page index
     tree to identify which nodes are most relevant to the user's query.
  2. Full-content retrieval   — selected nodes are resolved back to their
     complete markdown text from the crawl source files.
  3. Answer generation        — the retrieved context is sent to a generation
     model (Llama 4 Maverick on Bedrock) to produce the final answer.

Usage:
    python query_page_index.py crawl_output/fissionlabs.com_20260410_105854 "What AI services does Fission Labs offer?"
    python query_page_index.py crawl_output/fissionlabs.com_20260410_105854 --interactive
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv

from helpers import parse_model_json_response

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------

_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        region = os.getenv("AWS_REGION", "us-east-1")
        _bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return _bedrock_client


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

TREE_SEARCH_MODEL = os.getenv(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
ANSWER_MODEL = "us.meta.llama4-maverick-17b-instruct-v1:0"

# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------


def load_page_index(crawl_dir: Path) -> dict:
    index_path = crawl_dir / "page_index.json"
    # Fallback to old location for backward compatibility
    if not index_path.exists():
        index_path = crawl_dir / "metadata" / "page_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No page_index.json found in {crawl_dir}. "
            "Run generate_page_index.py first."
        )
    return json.loads(index_path.read_text(encoding="utf-8"))


def load_manifest(crawl_dir: Path) -> dict:
    manifest_path = crawl_dir / "metadata" / "crawl_manifest.json"
    if not manifest_path.exists():
        manifest_path = crawl_dir / "crawl_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_node_map(tree: dict) -> dict[str, dict]:
    """Flatten the tree into a {node_id: node_dict} mapping."""
    node_map: dict[str, dict] = {}

    def _walk(node: dict):
        node_map[node["node_id"]] = node
        for child in node.get("nodes", []):
            _walk(child)

    _walk(tree)
    return node_map


def build_url_to_markdown_path(crawl_dir: Path) -> dict[str, Path]:
    """Map each crawled URL to its local markdown file path."""
    manifest = load_manifest(crawl_dir)
    return {
        p["url"]: crawl_dir / p["markdown_file"]
        for p in manifest["pages"]
    }


def strip_tree_for_search(tree: dict, *, include_hints: bool = True) -> dict:
    """
    Return a lightweight copy of the tree suitable for the LLM search prompt.
    Keeps: node_id, title, summary, url (if present), and nested nodes.
    Drops: content_preview (too long for prompt context).
    """
    compact: dict[str, Any] = {
        "title": tree["title"],
        "node_id": tree["node_id"],
    }
    if tree.get("url"):
        compact["url"] = tree["url"]
    if tree.get("summary"):
        compact["summary"] = tree["summary"]
    if include_hints and tree.get("content_preview"):
        compact["hint"] = tree["content_preview"][:120]
    children = tree.get("nodes", [])
    if children:
        compact["nodes"] = [
            strip_tree_for_search(c, include_hints=include_hints)
            for c in children
        ]
    return compact


# ---------------------------------------------------------------------------
# Content resolution — node_id -> full markdown text
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _build_parent_map(tree: dict) -> dict[str, str]:
    """Return {child_node_id: parent_node_id} for every node in the tree."""
    parent_map: dict[str, str] = {}

    def _walk(node: dict):
        for child in node.get("nodes", []):
            parent_map[child["node_id"]] = node["node_id"]
            _walk(child)

    _walk(tree)
    return parent_map


def _find_page_url(node_id: str, node_map: dict, parent_map: dict) -> str | None:
    """Walk up from node_id through parent_map until we find a node with a URL."""
    current = node_id
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        node = node_map.get(current)
        if node and node.get("url"):
            return node["url"]
        current = parent_map.get(current)
    return None


def extract_section_content(
    md_text: str,
    section_title: str,
) -> str:
    """
    Extract the full content of a section identified by its heading title.
    Returns everything from the heading to the next heading of equal or
    higher level.
    """
    matches = list(_HEADING_RE.finditer(md_text))
    for i, m in enumerate(matches):
        heading_text = re.sub(r"!\[.*?\]\(.*?\)", "", m.group(2)).strip()
        heading_text = re.sub(r"\s+", " ", heading_text)
        if heading_text.lower() == section_title.lower():
            level = len(m.group(1))
            start = m.start()
            end = len(md_text)
            for j in range(i + 1, len(matches)):
                if len(matches[j].group(1)) <= level:
                    end = matches[j].start()
                    break
            return md_text[start:end].strip()
    return ""


def resolve_node_content(
    node_id: str,
    node_map: dict[str, dict],
    parent_map: dict[str, str],
    url_to_md_path: dict[str, Path],
) -> str:
    """
    Resolve a node_id to its full markdown content.

    For page-level nodes (those with a URL), returns the entire cleaned
    page markdown. For section-level nodes, extracts just that section
    from the parent page.
    """
    node = node_map.get(node_id)
    if not node:
        return ""

    page_url = _find_page_url(node_id, node_map, parent_map)
    if not page_url:
        return node.get("content_preview", "")

    md_path = url_to_md_path.get(page_url)
    if not md_path or not md_path.exists():
        return node.get("content_preview", "")

    md_text = md_path.read_text(encoding="utf-8")

    if node.get("url"):
        return md_text

    section_text = extract_section_content(md_text, node["title"])
    if section_text:
        return section_text

    return node.get("content_preview", "")


# ---------------------------------------------------------------------------
# Stage 1: LLM-guided tree search
# ---------------------------------------------------------------------------

TREE_SEARCH_PROMPT = """\
You are an expert information retrieval system. You are given a user question \
and a hierarchical tree index of a website's content. Each node has a node_id, \
title, and optionally a summary or hint about its content.

Your task is to identify the nodes most likely to contain information relevant \
to answering the question. Think step-by-step:
1. Understand what the question is asking.
2. Scan the tree structure — consider both page-level nodes (with URLs) and \
   section-level nodes within pages.
3. Select the most relevant nodes. Prefer specific sections over entire pages \
   when possible, but include the page node if the whole page is relevant.
4. Select between 1 and 10 nodes. Be selective — only pick nodes that are \
   genuinely relevant.

Question: {query}

Website tree index:
{tree_json}

Reply with ONLY a JSON object in this exact format:
{{
    "reasoning": "<your step-by-step thinking about which nodes are relevant>",
    "node_ids": ["node_id_1", "node_id_2", ...]
}}"""


def tree_search(
    query: str,
    tree: dict,
    model_id: str = TREE_SEARCH_MODEL,
) -> dict:
    """
    Use an LLM to navigate the tree and pick relevant node_ids.
    Returns {"reasoning": str, "node_ids": list[str]}.
    """
    compact_tree = strip_tree_for_search(tree, include_hints=True)
    tree_json = json.dumps(compact_tree, indent=2, ensure_ascii=False)

    # Progressively reduce detail to fit within context window
    if len(tree_json) > 150_000:
        compact_tree = strip_tree_for_search(tree, include_hints=False)
        tree_json = json.dumps(compact_tree, indent=2, ensure_ascii=False)

    if len(tree_json) > 150_000:
        def _trim_summaries(node):
            if "summary" in node:
                node["summary"] = node["summary"][:60]
            for c in node.get("nodes", []):
                _trim_summaries(c)
        _trim_summaries(compact_tree)
        tree_json = json.dumps(compact_tree, indent=2, ensure_ascii=False)

    prompt = TREE_SEARCH_PROMPT.format(query=query, tree_json=tree_json)

    logger.info("Tree search: sending query to %s ...", model_id)
    start = time.time()

    if "claude" in model_id.lower() or "anthropic" in model_id.lower():
        raw = _invoke_claude(prompt, model_id, max_tokens=1024)
    else:
        formatted = _build_llama_prompt(
            "You are an expert information retrieval system.", prompt
        )
        raw = _invoke_llama(formatted, model_id, max_gen_len=1024)

    elapsed = time.time() - start
    logger.info("Tree search completed in %.1fs", elapsed)

    try:
        result = parse_model_json_response(raw)
    except Exception:
        logger.error("Failed to parse tree search response:\n%s", raw)
        result = {"reasoning": raw, "node_ids": []}

    if "node_list" in result and "node_ids" not in result:
        result["node_ids"] = result.pop("node_list")

    return result


# ---------------------------------------------------------------------------
# Stage 2: retrieve content for selected nodes
# ---------------------------------------------------------------------------

def retrieve_context(
    node_ids: list[str],
    tree: dict,
    crawl_dir: Path,
    max_context_chars: int = 15_000,
) -> list[dict]:
    """
    Resolve each node_id to its full content and return a list of
    {node_id, title, url, content} dicts.
    """
    node_map = build_node_map(tree)
    parent_map = _build_parent_map(tree)
    url_to_md_path = build_url_to_markdown_path(crawl_dir)

    chunks: list[dict] = []
    total_chars = 0

    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            logger.warning("Node %s not found in tree, skipping", nid)
            continue

        content = resolve_node_content(nid, node_map, parent_map, url_to_md_path)
        if not content:
            continue

        if total_chars + len(content) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 200:
                content = content[:remaining] + "\n\n[... truncated ...]"
            else:
                break

        page_url = _find_page_url(nid, node_map, parent_map) or ""

        chunks.append({
            "node_id": nid,
            "title": node["title"],
            "url": page_url,
            "content": content,
        })
        total_chars += len(content)

    return chunks


# ---------------------------------------------------------------------------
# Stage 3: answer generation with Llama 4 Maverick
# ---------------------------------------------------------------------------


def _build_llama_prompt(system: str, user: str) -> str:
    """Format a prompt using the Llama instruct chat template."""
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


_LLAMA_STOP_TOKENS = re.compile(
    r"<\|eot_id\|>|<\|end_of_text\|>|<\|start_header_id\|>.*"
)


def _invoke_llama(prompt: str, model_id: str, max_gen_len: int = 2048) -> str:
    client = _get_bedrock_client()
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": max_gen_len,
        "temperature": 0.3,
    })
    resp = client.invoke_model(modelId=model_id, body=body)
    result = json.loads(resp["body"].read())
    text = result.get("generation", "")
    return _LLAMA_STOP_TOKENS.sub("", text).strip()


def _invoke_claude(prompt: str, model_id: str, max_tokens: int = 2048) -> str:
    client = _get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = client.invoke_model(modelId=model_id, body=body)
    raw = json.loads(resp["body"].read())
    return raw["content"][0]["text"]


def generate_answer(
    query: str,
    context_chunks: list[dict],
    model_id: str = ANSWER_MODEL,
) -> str:
    """Send the retrieved context to the answer model and return the response."""
    context_parts: list[str] = []
    for chunk in context_chunks:
        header = f"--- [{chunk['title']}]"
        if chunk.get("url"):
            header += f" ({chunk['url']})"
        header += " ---"
        context_parts.append(f"{header}\n{chunk['content']}")

    context_text = "\n\n".join(context_parts)

    system_msg = (
        "You are a helpful assistant that answers questions about a company's "
        "website. Use ONLY the provided context to answer. If the context does "
        "not contain enough information, say so clearly. Be concise, "
        "well-structured, and cite the source page URL when relevant."
    )
    user_msg = f"Question: {query}\n\nContext:\n{context_text}"

    logger.info("Answer generation: sending to %s ...", model_id)
    start = time.time()

    try:
        if "llama" in model_id.lower() or "meta" in model_id.lower():
            formatted = _build_llama_prompt(system_msg, user_msg)
            answer = _invoke_llama(formatted, model_id)
        elif "claude" in model_id.lower() or "anthropic" in model_id.lower():
            full_prompt = f"{system_msg}\n\n{user_msg}"
            answer = _invoke_claude(full_prompt, model_id)
        else:
            formatted = _build_llama_prompt(system_msg, user_msg)
            answer = _invoke_llama(formatted, model_id)
    except Exception:
        logger.warning(
            "Primary model invocation failed, falling back to Claude",
            exc_info=True,
        )
        full_prompt = f"{system_msg}\n\n{user_msg}"
        answer = _invoke_claude(
            full_prompt, "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )

    elapsed = time.time() - start
    logger.info("Answer generated in %.1fs", elapsed)
    return answer.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def query_website(
    query: str,
    crawl_dir: str | Path,
    *,
    tree_search_model: str = TREE_SEARCH_MODEL,
    answer_model: str = ANSWER_MODEL,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    End-to-end: tree search -> content retrieval -> answer generation.

    Returns:
        {
            "query": str,
            "reasoning": str,
            "selected_nodes": [{node_id, title, url}],
            "answer": str,
            "timing": {search_ms, retrieval_ms, answer_ms, total_ms},
        }
    """
    crawl_dir = Path(crawl_dir)
    t0 = time.time()

    tree = load_page_index(crawl_dir)

    # Stage 1: tree search
    t1 = time.time()
    search_result = tree_search(query, tree, model_id=tree_search_model)
    t2 = time.time()

    node_ids = search_result.get("node_ids", [])
    reasoning = search_result.get("reasoning", "")

    if verbose:
        print(f"\n{'='*60}")
        print("TREE SEARCH REASONING:")
        print(f"{'='*60}")
        for line in textwrap.wrap(reasoning, width=80):
            print(f"  {line}")
        print(f"\nSelected {len(node_ids)} nodes: {node_ids}")

    # Stage 2: content retrieval
    context_chunks = retrieve_context(node_ids, tree, crawl_dir)
    t3 = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print("RETRIEVED CONTEXT:")
        print(f"{'='*60}")
        for chunk in context_chunks:
            print(f"\n  [{chunk['node_id']}] {chunk['title']}")
            if chunk.get("url"):
                print(f"  URL: {chunk['url']}")
            preview = chunk["content"][:200].replace("\n", " ")
            print(f"  Preview: {preview}...")

    # Stage 3: answer generation
    if not context_chunks:
        answer = (
            "I couldn't find relevant information in the crawled website data "
            "to answer this question."
        )
        t4 = t3
    else:
        answer = generate_answer(query, context_chunks, model_id=answer_model)
        t4 = time.time()

    total_ms = int((t4 - t0) * 1000)

    return {
        "query": query,
        "reasoning": reasoning,
        "selected_nodes": [
            {
                "node_id": c["node_id"],
                "title": c["title"],
                "url": c.get("url", ""),
            }
            for c in context_chunks
        ],
        "answer": answer,
        "timing": {
            "search_ms": int((t2 - t1) * 1000),
            "retrieval_ms": int((t3 - t2) * 1000),
            "answer_ms": int((t4 - t3) * 1000),
            "total_ms": total_ms,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: dict) -> None:
    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    for line in textwrap.wrap(result["answer"], width=80):
        print(f"  {line}")

    print(f"\n{'-'*60}")
    print("Sources:")
    for node in result["selected_nodes"]:
        url_str = f"  ({node['url']})" if node.get("url") else ""
        print(f"  * [{node['node_id']}] {node['title']}{url_str}")

    t = result["timing"]
    print(f"\n{'-'*60}")
    print(
        f"Timing: search={t['search_ms']}ms  retrieval={t['retrieval_ms']}ms  "
        f"answer={t['answer_ms']}ms  total={t['total_ms']}ms"
    )


def _interactive_loop(crawl_dir: Path, args: argparse.Namespace) -> None:
    print(f"\nTree-based retrieval engine for: {crawl_dir.name}")
    print(f"Tree search model:  {args.tree_search_model}")
    print(f"Answer model:       {args.answer_model}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = query_website(
            query,
            crawl_dir,
            tree_search_model=args.tree_search_model,
            answer_model=args.answer_model,
            verbose=args.verbose,
        )
        _print_result(result)
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query crawled website data using tree-based retrieval."
    )
    parser.add_argument(
        "crawl_dir",
        help="Path to the crawl output directory",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive query loop",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show reasoning and retrieved context",
    )
    parser.add_argument(
        "--tree-search-model",
        default=TREE_SEARCH_MODEL,
        help=f"Bedrock model for tree search (default: {TREE_SEARCH_MODEL})",
    )
    parser.add_argument(
        "--answer-model",
        default=ANSWER_MODEL,
        help=f"Bedrock model for answer generation (default: {ANSWER_MODEL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    args = _parse_args()
    crawl_dir = Path(args.crawl_dir)

    if args.interactive or args.query is None:
        _interactive_loop(crawl_dir, args)
    else:
        result = query_website(
            args.query,
            crawl_dir,
            tree_search_model=args.tree_search_model,
            answer_model=args.answer_model,
            verbose=args.verbose,
        )
        _print_result(result)
