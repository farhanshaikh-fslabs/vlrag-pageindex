"""
End-to-end orchestrator: crawl a website, build a page index, and query it.

Usage:
    # Full pipeline: crawl + index + query
    python orchestrator.py https://example.com "What does the company do?"

    # Crawl + index only (no query)
    python orchestrator.py https://example.com --no-query

    # Index + query on an existing crawl
    python orchestrator.py --crawl-dir crawl_output/example.com_20260410_105854 "What does the company do?"

    # Query-only on an existing crawl that already has page_index.json
    python orchestrator.py --crawl-dir crawl_output/example.com_20260410_105854 --skip-index "What does the company do?"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_MODEL = os.getenv(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
DEFAULT_TREE_SEARCH_MODEL = os.getenv(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
DEFAULT_ANSWER_MODEL = "us.meta.llama4-maverick-17b-instruct-v1:0"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def crawl_website(
    website_url: str,
    *,
    output_dir: str = "crawl_output",
    depth: int = 2,
    max_pages: int = 2000,
    useful_only: bool = True,
    remove_common_sections: bool = True,
) -> Path:
    """Stage 1: Crawl the website and write markdown files."""
    from run_full_website_pipeline import run_pipeline

    logger.info("Stage 1: Crawling %s (depth=%d, max_pages=%d) ...", website_url, depth, max_pages)
    start = time.time()

    crawl_dir = run_pipeline(
        website_url=website_url,
        output_dir=output_dir,
        depth=depth,
        max_pages=max_pages,
        useful_only=useful_only,
        remove_common_sections=remove_common_sections,
    )

    elapsed = time.time() - start
    logger.info("Stage 1 complete: %d pages crawled in %.1fs -> %s", _count_pages(crawl_dir), elapsed, crawl_dir)
    return crawl_dir


def build_index(
    crawl_dir: str | Path,
    *,
    generate_summaries: bool = True,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
) -> dict[str, Any]:
    """Stage 2: Build the page index tree from crawled data."""
    from generate_page_index import generate_page_index

    logger.info("Stage 2: Building page index (summaries=%s, model=%s) ...", generate_summaries, summary_model)
    start = time.time()

    tree = generate_page_index(
        crawl_dir,
        generate_summaries=generate_summaries,
        model_id=summary_model,
    )

    elapsed = time.time() - start
    logger.info("Stage 2 complete: index built in %.1fs", elapsed)
    return tree


def query(
    question: str,
    crawl_dir: str | Path,
    *,
    tree_search_model: str = DEFAULT_TREE_SEARCH_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    verbose: bool = False,
) -> dict[str, Any]:
    """Stage 3: Query the page index."""
    from query_page_index import query_website

    logger.info("Stage 3: Querying with search_model=%s, answer_model=%s ...", tree_search_model, answer_model)
    return query_website(
        question,
        crawl_dir,
        tree_search_model=tree_search_model,
        answer_model=answer_model,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run(
    website_url: str | None = None,
    crawl_dir: str | Path | None = None,
    question: str | None = None,
    *,
    output_dir: str = "crawl_output",
    depth: int = 2,
    max_pages: int = 2000,
    useful_only: bool = True,
    remove_common_sections: bool = True,
    generate_summaries: bool = True,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    tree_search_model: str = DEFAULT_TREE_SEARCH_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    skip_index: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run the full pipeline or any subset of stages.

    Args:
        website_url:        URL to crawl (skip if crawl_dir is provided).
        crawl_dir:          Existing crawl output directory (skip crawling).
        question:           Question to query (skip if None).
        output_dir:         Base output directory for crawl data.
        depth:              Crawl depth.
        max_pages:          Max pages to crawl.
        useful_only:        Filter to commercially useful pages.
        remove_common_sections: Strip repeated nav/footer sections.
        generate_summaries: Generate LLM summaries for each node.
        summary_model:      Bedrock model for summary generation.
        tree_search_model:  Bedrock model for tree search.
        answer_model:       Bedrock model for answer generation.
        skip_index:         Skip index generation (use existing page_index.json).
        verbose:            Print detailed output during query.

    Returns:
        {
            "crawl_dir": str,
            "pages_crawled": int,
            "index_nodes": int,
            "query_result": {...} or None,
            "timing": {"crawl_s", "index_s", "query_s", "total_s"},
        }
    """
    t0 = time.time()
    result: dict[str, Any] = {
        "crawl_dir": None,
        "pages_crawled": 0,
        "index_nodes": 0,
        "query_result": None,
        "timing": {},
    }

    # --- Stage 1: Crawl ---
    t_crawl_start = time.time()
    if crawl_dir:
        crawl_dir = Path(crawl_dir)
        result["crawl_dir"] = str(crawl_dir)
        result["pages_crawled"] = _count_pages(crawl_dir)
        logger.info("Using existing crawl: %s (%d pages)", crawl_dir, result["pages_crawled"])
    elif website_url:
        crawl_dir = crawl_website(
            website_url,
            output_dir=output_dir,
            depth=depth,
            max_pages=max_pages,
            useful_only=useful_only,
            remove_common_sections=remove_common_sections,
        )
        result["crawl_dir"] = str(crawl_dir)
        result["pages_crawled"] = _count_pages(crawl_dir)
    else:
        raise ValueError("Either website_url or crawl_dir must be provided")
    t_crawl_end = time.time()
    result["timing"]["crawl_s"] = round(t_crawl_end - t_crawl_start, 1)

    # --- Stage 2: Index ---
    t_index_start = time.time()
    if not skip_index:
        tree = build_index(
            crawl_dir,
            generate_summaries=generate_summaries,
            summary_model=summary_model,
        )
        result["index_nodes"] = _count_tree_nodes(tree)
    else:
        index_path = Path(crawl_dir) / "page_index.json"
        if not index_path.exists():
            index_path = Path(crawl_dir) / "metadata" / "page_index.json"
        if index_path.exists():
            tree = json.loads(index_path.read_text(encoding="utf-8"))
            result["index_nodes"] = _count_tree_nodes(tree)
            logger.info("Using existing page_index.json (%d nodes)", result["index_nodes"])
        else:
            raise FileNotFoundError(f"No page_index.json found in {crawl_dir}")
    t_index_end = time.time()
    result["timing"]["index_s"] = round(t_index_end - t_index_start, 1)

    # --- Stage 3: Query ---
    t_query_start = time.time()
    if question:
        result["query_result"] = query(
            question,
            crawl_dir,
            tree_search_model=tree_search_model,
            answer_model=answer_model,
            verbose=verbose,
        )
    t_query_end = time.time()
    result["timing"]["query_s"] = round(t_query_end - t_query_start, 1)

    result["timing"]["total_s"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_pages(crawl_dir: Path) -> int:
    for name in ("metadata/crawl_manifest.json", "crawl_manifest.json"):
        p = crawl_dir / name
        if p.exists():
            manifest = json.loads(p.read_text(encoding="utf-8"))
            return len(manifest.get("pages", []))
    return 0


def _count_tree_nodes(tree: dict) -> int:
    count = 1
    for child in tree.get("nodes", []):
        count += _count_tree_nodes(child)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate the full pipeline: crawl -> index -> query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Crawl, index, and query
  python orchestrator.py https://example.com "What does the company do?"

  # Crawl and index only
  python orchestrator.py https://example.com --no-query

  # Index and query existing crawl
  python orchestrator.py --crawl-dir crawl_output/site_20260410 "Tell me about services"

  # Query only (skip index)
  python orchestrator.py --crawl-dir crawl_output/site_20260410 --skip-index "Tell me about services"
""",
    )

    source = parser.add_argument_group("data source (provide one)")
    source.add_argument("website_url", nargs="?", default=None, help="URL to crawl (or question if --crawl-dir is set)")
    source.add_argument("--crawl-dir", default=None, help="Existing crawl output directory")

    parser.add_argument("--question", "-q", default=None, help="Question to ask")
    parser.add_argument("--no-query", action="store_true", help="Skip the query stage")

    crawl_opts = parser.add_argument_group("crawl options")
    crawl_opts.add_argument("--output-dir", default="crawl_output")
    crawl_opts.add_argument("--depth", type=int, default=2)
    crawl_opts.add_argument("--max-pages", type=int, default=2000)

    index_opts = parser.add_argument_group("index options")
    index_opts.add_argument("--skip-index", action="store_true", help="Use existing page_index.json")
    index_opts.add_argument("--no-summaries", action="store_true", help="Skip LLM summary generation")
    index_opts.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL, help="Model for summary generation")

    query_opts = parser.add_argument_group("query options")
    query_opts.add_argument("--tree-search-model", default=DEFAULT_TREE_SEARCH_MODEL, help="Model for tree search")
    query_opts.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL, help="Model for answer generation")
    query_opts.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def _print_summary(result: dict) -> None:
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Crawl dir:     {result['crawl_dir']}")
    print(f"  Pages crawled: {result['pages_crawled']}")
    print(f"  Index nodes:   {result['index_nodes']}")
    t = result["timing"]
    print(f"  Timing:        crawl={t['crawl_s']}s  index={t['index_s']}s  query={t['query_s']}s  total={t['total_s']}s")

    qr = result.get("query_result")
    if qr:
        print(f"\n{'-'*60}")
        print("ANSWER:")
        print(f"{'-'*60}")
        print(f"  {qr['answer'][:500]}")
        if len(qr["answer"]) > 500:
            print("  ...")
        print(f"\n  Sources: {len(qr['selected_nodes'])} nodes")
        for n in qr["selected_nodes"]:
            url_str = f"  ({n['url']})" if n.get("url") else ""
            print(f"    * [{n['node_id']}] {n['title'][:50]}{url_str}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    args = _parse_args()

    if not args.website_url and not args.crawl_dir:
        print("Error: provide either a website URL or --crawl-dir")
        raise SystemExit(1)

    # When --crawl-dir is used, treat the positional arg as the question
    website_url = args.website_url
    question = args.question
    if args.crawl_dir and args.website_url and not args.question:
        question = args.website_url
        website_url = None

    if args.no_query:
        question = None

    result = run(
        website_url=website_url,
        crawl_dir=args.crawl_dir,
        question=question,
        output_dir=args.output_dir,
        depth=args.depth,
        max_pages=args.max_pages,
        generate_summaries=not args.no_summaries,
        summary_model=args.summary_model,
        tree_search_model=args.tree_search_model,
        answer_model=args.answer_model,
        skip_index=args.skip_index,
        verbose=args.verbose,
    )

    _print_summary(result)
