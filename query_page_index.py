"""
Tree-based retrieval engine for crawled website data.

Uses a two-stage approach:
  1. LLM-guided tree search  — a reasoning model navigates the page index
     tree to identify which pages are most relevant to the user's query.
  2. Full-content retrieval   — selected nodes are resolved to their full
     markdown content using the markdown_file path stored in each node.
  3. Answer generation        — the retrieved markdown content is sent to a
     generation model to produce the final answer.

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
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
ANSWER_MODEL = os.getenv(
    "ANSWER_BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------


def load_page_index(crawl_dir: Path) -> dict:
    index_path = crawl_dir / "page_index.json"
    if not index_path.exists():
        index_path = crawl_dir / "metadata" / "page_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No page_index.json found in {crawl_dir}. "
            "Run generate_page_index.py first."
        )
    return json.loads(index_path.read_text(encoding="utf-8"))


def build_node_map(tree: dict) -> dict[str, dict]:
    """Flatten the tree into a {node_id: node_dict} mapping."""
    node_map: dict[str, dict] = {}

    def _walk(node: dict):
        node_map[node["node_id"]] = node
        for child in node.get("nodes", []):
            _walk(child)

    _walk(tree)
    return node_map


def strip_tree_for_search(tree: dict) -> dict:
    """
    Return a lightweight copy of the tree suitable for the LLM search prompt.
    Keeps: node_id, title, summary, url, markdown_file.
    """
    compact: dict[str, Any] = {
        "title": tree["title"],
        "node_id": tree["node_id"],
    }
    if tree.get("url"):
        compact["url"] = tree["url"]
    if tree.get("summary"):
        compact["summary"] = tree["summary"]
    if tree.get("markdown_file"):
        compact["markdown_file"] = tree["markdown_file"]
    children = tree.get("nodes", [])
    if children:
        compact["nodes"] = [strip_tree_for_search(c) for c in children]
    return compact


# ---------------------------------------------------------------------------
# Content retrieval — node_id -> full markdown file content
# ---------------------------------------------------------------------------

def retrieve_markdown_content(
    node: dict,
    crawl_dir: Path,
) -> str:
    """
    Retrieve the full markdown content for a node using its markdown_file path.
    """
    md_file = node.get("markdown_file")
    if not md_file:
        return ""

    md_path = crawl_dir / md_file
    if not md_path.exists():
        logger.warning("Markdown file not found: %s", md_path)
        return ""

    return md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage 1: LLM-guided tree search
# ---------------------------------------------------------------------------

TREE_SEARCH_PROMPT = """\
You are an expert information retrieval system. You are given a user question \
and a hierarchical tree index of a website's content. Each node represents a page \
with a node_id, title, URL, and a summary of its content.

Your task is to identify ALL pages that might contain information relevant \
to answering the question. Think step-by-step:
1. Understand what the question is asking.
2. Scan the tree structure and read the summaries to find relevant pages.
3. Select ALL relevant pages (up to 15 pages). Include every page that could \
   potentially help answer the question - it's better to include more pages \
   than to miss important information.

Question: {query}

Website page index:
{tree_json}

Reply with ONLY a JSON object in this exact format:
{{
    "reasoning": "<your step-by-step thinking about which pages are relevant>",
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
    compact_tree = strip_tree_for_search(tree)
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
        raw = _invoke_claude(prompt, model_id, max_tokens=10240)
    else:
        formatted = _build_llama_prompt(
            "You are an expert information retrieval system.", prompt
        )
        raw = _invoke_llama(formatted, model_id, max_gen_len=10240)

    elapsed = time.time() - start
    logger.info("Tree search completed in %.1fs", elapsed)

    try:
        result = parse_model_json_response(raw)
    except Exception:
        logger.warning("JSON parsing failed, attempting regex extraction...")
        # Fallback: extract node_ids using regex
        node_ids = _extract_node_ids_regex(raw)
        if node_ids:
            logger.info("Regex extraction found %d node_ids", len(node_ids))
            result = {"reasoning": raw, "node_ids": node_ids}
        else:
            logger.error("Failed to parse tree search response:\n%s", raw)
            result = {"reasoning": raw, "node_ids": []}

    if "node_list" in result and "node_ids" not in result:
        result["node_ids"] = result.pop("node_list")

    return result


def _extract_node_ids_regex(raw_response: str) -> list[str]:
    """
    Fallback extraction of node_ids from response using regex.
    Looks for patterns like "node_ids": ["0001", "0002", ...]
    """
    # Try to find the node_ids array
    pattern = r'"node_ids"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, raw_response, re.DOTALL)
    if match:
        ids_str = match.group(1)
        # Extract all quoted strings
        ids = re.findall(r'"(\d{4})"', ids_str)
        return ids
    return []


# ---------------------------------------------------------------------------
# Stage 2: retrieve full markdown content for selected nodes
# ---------------------------------------------------------------------------

def retrieve_context(
    node_ids: list[str],
    tree: dict,
    crawl_dir: Path,
    max_context_chars: int = 300_000,
) -> list[dict]:
    """
    Resolve each node_id to its full markdown content and return a list of
    {node_id, title, url, markdown_file, content} dicts.
    """
    node_map = build_node_map(tree)

    chunks: list[dict] = []
    total_chars = 0

    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            logger.warning("Node %s not found in tree, skipping", nid)
            continue

        content = retrieve_markdown_content(node, crawl_dir)
        if not content:
            logger.warning("No markdown content for node %s, skipping", nid)
            continue

        if total_chars + len(content) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 500:
                content = content[:remaining] + "\n\n[... truncated ...]"
            else:
                logger.info("Context limit reached, stopping at %d pages", len(chunks))
                break

        chunks.append({
            "node_id": nid,
            "title": node.get("title", ""),
            "url": node.get("url", ""),
            "markdown_file": node.get("markdown_file", ""),
            "content": content,
        })
        total_chars += len(content)

    return chunks


# ---------------------------------------------------------------------------
# Stage 3: answer generation
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


def _invoke_llama(prompt: str, model_id: str, max_gen_len: int = 10240) -> str:
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


def _invoke_claude(prompt: str, model_id: str, max_tokens: int = 10240) -> str:
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
    """Send the retrieved markdown content to the answer model and return the response."""
    context_parts: list[str] = []
    for chunk in context_chunks:
        header = f"--- Page: {chunk['title']}"
        if chunk.get("url"):
            header += f" ({chunk['url']})"
        header += " ---"
        context_parts.append(f"{header}\n\n{chunk['content']}")

    context_text = "\n\n".join(context_parts)

    system_msg = (
        "You are a helpful assistant that answers questions about a company's "
        "website. Use ONLY the provided page content to answer. If the content does "
        "not contain enough information, say so clearly. Be concise, "
        "well-structured, and cite the source page URL when relevant."
    )
    user_msg = f"Question: {query}\n\nPage Content:\n{context_text}"

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
    max_context_chars: int = 300_000,
) -> dict[str, Any]:
    """
    End-to-end: tree search -> content retrieval -> answer generation.

    Returns:
        {
            "query": str,
            "reasoning": str,
            "selected_nodes": [{node_id, title, url, markdown_file}],
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
        print(f"\nSelected {len(node_ids)} pages: {node_ids}")

    # Stage 2: content retrieval (full markdown files)
    context_chunks = retrieve_context(
        node_ids, tree, crawl_dir, max_context_chars=max_context_chars
    )
    t3 = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print("RETRIEVED PAGES:")
        print(f"{'='*60}")
        for chunk in context_chunks:
            print(f"\n  [{chunk['node_id']}] {chunk['title']}")
            if chunk.get("url"):
                print(f"  URL: {chunk['url']}")
            if chunk.get("markdown_file"):
                print(f"  File: {chunk['markdown_file']}")
            print(f"  Content length: {len(chunk['content'])} chars")

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
                "markdown_file": c.get("markdown_file", ""),
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
        file_str = f"  -> {node['markdown_file']}" if node.get("markdown_file") else ""
        print(f"  * [{node['node_id']}] {node['title']}{url_str}{file_str}")

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
            max_context_chars=args.max_context,
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
    parser.add_argument(
        "--max-context",
        type=int,
        default=300000,
        help="Maximum characters of page content to send to answer model (default: 300000)",
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
