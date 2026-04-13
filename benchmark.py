"""
Benchmark different model combinations for tree-search and answer generation.

Runs a fixed set of queries against a crawled website using every combination
of tree-search model x answer model, then writes a comparison report.

Usage:
    python benchmark.py crawl_output/fissionlabs.com_20260410_105854
    python benchmark.py crawl_output/fissionlabs.com_20260410_105854 --queries queries.json
    python benchmark.py crawl_output/fissionlabs.com_20260410_105854 --output-dir bench_results
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from query_page_index import query_website, load_page_index, build_node_map

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model pools
# ---------------------------------------------------------------------------

DEFAULT_TREE_SEARCH_MODELS = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]

DEFAULT_ANSWER_MODELS = [
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]

DEFAULT_QUERIES = [
    "What services does this company offer?",
    "What case studies or success stories are available?",
    "What is the company's mission and background?",
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_single(
    query_text: str,
    crawl_dir: Path,
    tree_search_model: str,
    answer_model: str,
) -> dict[str, Any]:
    """Run a single query with a specific model combination and capture results."""
    try:
        result = query_website(
            query_text,
            crawl_dir,
            tree_search_model=tree_search_model,
            answer_model=answer_model,
            verbose=False,
        )
        return {
            "status": "ok",
            "query": query_text,
            "tree_search_model": tree_search_model,
            "answer_model": answer_model,
            "reasoning": result["reasoning"],
            "answer": result["answer"],
            "selected_nodes": result["selected_nodes"],
            "num_nodes_selected": len(result["selected_nodes"]),
            "timing": result["timing"],
            "answer_length": len(result["answer"]),
        }
    except Exception as e:
        logger.error("Failed: query=%r  search=%s  answer=%s  error=%s", query_text, tree_search_model, answer_model, e)
        return {
            "status": "error",
            "query": query_text,
            "tree_search_model": tree_search_model,
            "answer_model": answer_model,
            "error": str(e),
            "timing": {},
        }


def run_benchmark(
    crawl_dir: str | Path,
    *,
    queries: list[str] | None = None,
    tree_search_models: list[str] | None = None,
    answer_models: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run all query x model combinations and produce a benchmark report.

    Args:
        crawl_dir:            Path to crawl output directory.
        queries:              List of test queries.
        tree_search_models:   List of Bedrock model IDs for tree search.
        answer_models:        List of Bedrock model IDs for answer generation.
        output_dir:           Where to write benchmark results.

    Returns:
        Full benchmark results dict, also saved to output_dir.
    """
    crawl_dir = Path(crawl_dir)
    queries = queries or DEFAULT_QUERIES
    tree_search_models = tree_search_models or DEFAULT_TREE_SEARCH_MODELS
    answer_models = answer_models or DEFAULT_ANSWER_MODELS

    # Pre-validate index exists
    tree = load_page_index(crawl_dir)
    total_nodes = len(build_node_map(tree))

    combos = list(product(tree_search_models, answer_models))
    total_runs = len(queries) * len(combos)

    logger.info(
        "Benchmark: %d queries x %d model combos = %d runs",
        len(queries), len(combos), total_runs,
    )
    logger.info("  Tree search models: %s", tree_search_models)
    logger.info("  Answer models:      %s", answer_models)

    all_results: list[dict] = []
    t0 = time.time()
    run_idx = 0

    for q in queries:
        for search_model, ans_model in combos:
            run_idx += 1
            logger.info(
                "[%d/%d] query=%r  search=%s  answer=%s",
                run_idx, total_runs, q[:50], search_model.split(".")[-1], ans_model.split(".")[-1],
            )
            result = run_single(q, crawl_dir, search_model, ans_model)
            all_results.append(result)

    total_time = time.time() - t0

    report = _build_report(
        all_results,
        crawl_dir=str(crawl_dir),
        total_nodes=total_nodes,
        total_time_s=round(total_time, 1),
        queries=queries,
        tree_search_models=tree_search_models,
        answer_models=answer_models,
    )

    # Save results
    if output_dir is None:
        output_dir = crawl_dir / "benchmark"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"bench_{timestamp}.json"
    results_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Benchmark results written to %s", results_path)

    return report


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(
    results: list[dict],
    *,
    crawl_dir: str,
    total_nodes: int,
    total_time_s: float,
    queries: list[str],
    tree_search_models: list[str],
    answer_models: list[str],
) -> dict[str, Any]:
    """Aggregate raw results into a structured comparison report."""
    ok_results = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] == "error"]

    # Per-combo aggregates
    combo_stats: dict[str, dict] = {}
    for r in ok_results:
        key = f"{r['tree_search_model']} | {r['answer_model']}"
        if key not in combo_stats:
            combo_stats[key] = {
                "tree_search_model": r["tree_search_model"],
                "answer_model": r["answer_model"],
                "runs": 0,
                "total_search_ms": 0,
                "total_answer_ms": 0,
                "total_ms": 0,
                "avg_nodes_selected": 0,
                "avg_answer_length": 0,
            }
        s = combo_stats[key]
        s["runs"] += 1
        s["total_search_ms"] += r["timing"].get("search_ms", 0)
        s["total_answer_ms"] += r["timing"].get("answer_ms", 0)
        s["total_ms"] += r["timing"].get("total_ms", 0)
        s["avg_nodes_selected"] += r["num_nodes_selected"]
        s["avg_answer_length"] += r["answer_length"]

    for s in combo_stats.values():
        n = s["runs"] or 1
        s["avg_search_ms"] = round(s["total_search_ms"] / n)
        s["avg_answer_ms"] = round(s["total_answer_ms"] / n)
        s["avg_total_ms"] = round(s["total_ms"] / n)
        s["avg_nodes_selected"] = round(s["avg_nodes_selected"] / n, 1)
        s["avg_answer_length"] = round(s["avg_answer_length"] / n)

    # Per-query comparison
    query_comparisons: list[dict] = []
    for q in queries:
        q_results = [r for r in ok_results if r["query"] == q]
        query_comparisons.append({
            "query": q,
            "results": [
                {
                    "tree_search_model": r["tree_search_model"],
                    "answer_model": r["answer_model"],
                    "answer_preview": r["answer"][:300],
                    "num_nodes": r["num_nodes_selected"],
                    "nodes": [n["title"][:60] for n in r["selected_nodes"]],
                    "timing_ms": r["timing"].get("total_ms", 0),
                }
                for r in q_results
            ],
        })

    return {
        "meta": {
            "crawl_dir": crawl_dir,
            "total_nodes_in_index": total_nodes,
            "num_queries": len(queries),
            "tree_search_models": tree_search_models,
            "answer_models": answer_models,
            "total_runs": len(results),
            "successful_runs": len(ok_results),
            "failed_runs": len(err_results),
            "total_time_s": total_time_s,
            "timestamp": datetime.now().isoformat(),
        },
        "combo_stats": list(combo_stats.values()),
        "query_comparisons": query_comparisons,
        "raw_results": results,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    """Print a human-readable summary of the benchmark report."""
    meta = report["meta"]
    print(f"\n{'='*70}")
    print("BENCHMARK REPORT")
    print(f"{'='*70}")
    print(f"  Crawl dir:       {meta['crawl_dir']}")
    print(f"  Index nodes:     {meta['total_nodes_in_index']}")
    print(f"  Queries tested:  {meta['num_queries']}")
    print(f"  Total runs:      {meta['total_runs']} ({meta['successful_runs']} ok, {meta['failed_runs']} failed)")
    print(f"  Total time:      {meta['total_time_s']}s")

    print(f"\n{'-'*70}")
    print("MODEL COMBINATION STATS")
    print(f"{'-'*70}")
    for s in report["combo_stats"]:
        search_short = s["tree_search_model"].split(".")[-1]
        answer_short = s["answer_model"].split(".")[-1]
        print(f"\n  Search: {search_short}")
        print(f"  Answer: {answer_short}")
        print(f"    Avg search:    {s['avg_search_ms']}ms")
        print(f"    Avg answer:    {s['avg_answer_ms']}ms")
        print(f"    Avg total:     {s['avg_total_ms']}ms")
        print(f"    Avg nodes:     {s['avg_nodes_selected']}")
        print(f"    Avg ans len:   {s['avg_answer_length']} chars")

    print(f"\n{'-'*70}")
    print("PER-QUERY COMPARISON")
    print(f"{'-'*70}")
    for qc in report["query_comparisons"]:
        print(f"\n  Q: {qc['query']}")
        for r in qc["results"]:
            answer_short = r["answer_model"].split(".")[-1]
            preview = r["answer_preview"][:120].replace("\n", " ")
            print(f"    [{answer_short}] ({r['timing_ms']}ms, {r['num_nodes']} nodes)")
            print(f"      {preview}...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark different model combinations for tree-based retrieval."
    )
    parser.add_argument(
        "crawl_dir",
        help="Path to the crawl output directory (must have page_index.json)",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="Path to JSON file with a list of query strings",
    )
    parser.add_argument(
        "--tree-search-models",
        nargs="+",
        default=None,
        help="Bedrock model IDs for tree search stage",
    )
    parser.add_argument(
        "--answer-models",
        nargs="+",
        default=None,
        help="Bedrock model IDs for answer generation stage",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for benchmark results (default: <crawl_dir>/benchmark)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    args = _parse_args()

    queries = None
    if args.queries:
        queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))

    report = run_benchmark(
        args.crawl_dir,
        queries=queries,
        tree_search_models=args.tree_search_models,
        answer_models=args.answer_models,
        output_dir=args.output_dir,
    )

    print_report(report)
