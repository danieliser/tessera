"""Benchmark embedding models for code search quality.

Indexes tessera's own codebase with each candidate model, runs a standard
query set, and scores results by relevance. Produces a comparison table.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tessera.db import ProjectDB
from tessera.embeddings import FastembedClient
from tessera.indexer import IndexerPipeline
from tessera.search import hybrid_search

# --- Config ---

MODELS = [
    ("BAAI/bge-small-en-v1.5", "BGE-Small"),
    ("BAAI/bge-base-en-v1.5", "BGE-Base"),
    ("jinaai/jina-embeddings-v2-base-code", "Jina-Code"),
    ("nomic-ai/nomic-embed-text-v1.5", "Nomic-1.5"),
    ("snowflake/snowflake-arctic-embed-m", "Arctic-M"),
]

# Queries with expected file/symbol hits (ground truth)
QUERIES = [
    (
        "hybrid search combining keyword and semantic results",
        ["search.py"],
        "Core hybrid search function",
    ),
    (
        "embedding client with caching",
        ["embeddings.py"],
        "Embedding client implementation",
    ),
    (
        "tree-sitter AST parsing for multiple languages",
        ["parser.py"],
        "Parser orchestration",
    ),
    (
        "scope-gated authentication tokens",
        ["auth.py"],
        "Auth/scope system",
    ),
    (
        "incremental re-indexing changed files",
        ["_pipeline.py", "indexer"],
        "Incremental indexer",
    ),
    (
        "SQLite FTS5 full text search",
        ["_project.py", "search.py"],
        "FTS5 integration",
    ),
    (
        "reciprocal rank fusion merge",
        ["search.py"],
        "RRF algorithm",
    ),
    (
        "FAISS vector similarity index",
        ["_project.py"],
        "FAISS index management",
    ),
    (
        "MCP server tool registration",
        ["_app.py", "server"],
        "MCP server setup",
    ),
    (
        "PageRank graph scoring for code symbols",
        ["ppr.py", "graph"],
        "PPR/graph ranking",
    ),
    (
        "chunk code into overlapping windows",
        ["chunker.py"],
        "Code chunking",
    ),
    (
        "normalize BM25 scores to 0-1 range",
        ["search.py"],
        "Score normalization",
    ),
    (
        "cross-encoder reranking after search",
        ["embeddings.py", "_search.py"],
        "Reranker integration",
    ),
    (
        "extract code snippet with ancestor context",
        ["search.py"],
        "Snippet extraction",
    ),
    (
        "session token creation and validation",
        ["auth.py"],
        "Session management",
    ),
]

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def index_with_model(model_name: str, label: str, base_dir: str) -> tuple[ProjectDB, float, str]:
    """Index the project with a specific embedding model. Returns (db, elapsed)."""
    # Each model gets its own base_dir so DBs don't collide
    model_base = os.path.join(base_dir, label)
    os.makedirs(model_base, exist_ok=True)

    # Override ProjectDB storage location
    ProjectDB.base_dir = model_base

    client = FastembedClient(model_name=model_name)

    pipeline = IndexerPipeline(
        project_path=PROJECT_PATH,
        embedding_client=client,
    )
    pipeline.register()

    start = time.perf_counter()
    stats = pipeline.index_project()
    elapsed = time.perf_counter() - start

    print(f"  {label}: {stats.files_processed} files, {stats.chunks_created} chunks, "
          f"{stats.chunks_embedded} embedded in {elapsed:.1f}s")

    client.close()

    # Reopen DB for querying (keep base_dir set — caller manages it)
    db = ProjectDB(PROJECT_PATH)
    return db, elapsed, model_base


def run_queries(db: ProjectDB, client: FastembedClient, queries: list) -> list[dict]:
    """Run all queries against a DB and score results."""
    results = []
    for query, expected_files, desc in queries:
        raw = client.embed_query(query)
        query_embedding = np.array(raw, dtype=np.float32)

        hits = hybrid_search(
            query, query_embedding, db,
            graph=None, limit=10,
            source_type=None, search_types=None,
            advanced_fts=False, rrf_weights=None,
        )

        hit_files = [h.get("file_path", "") for h in hits]

        top1_match = any(exp in hit_files[0] for exp in expected_files) if hits else False
        top3_match = any(
            any(exp in f for exp in expected_files)
            for f in hit_files[:3]
        )
        top5_match = any(
            any(exp in f for exp in expected_files)
            for f in hit_files[:5]
        )
        top10_match = any(
            any(exp in f for exp in expected_files)
            for f in hit_files[:10]
        )

        mrr = 0.0
        for rank, f in enumerate(hit_files, 1):
            if any(exp in f for exp in expected_files):
                mrr = 1.0 / rank
                break

        top_score = hits[0].get("score", 0) if hits else 0

        results.append({
            "query": query[:60],
            "desc": desc,
            "top1": top1_match,
            "top3": top3_match,
            "top5": top5_match,
            "top10": top10_match,
            "mrr": mrr,
            "top_score": top_score,
            "top_file": os.path.basename(hit_files[0]) if hits else "—",
            "num_hits": len(hits),
        })

    return results


def main():
    print("=" * 80)
    print("Tessera Embedding Model Benchmark")
    print("=" * 80)
    print(f"Project: {PROJECT_PATH}")
    print(f"Models:  {len(MODELS)}")
    print(f"Queries: {len(QUERIES)}")
    print()

    base_dir = tempfile.mkdtemp(prefix="tessera_bench_")
    print(f"Working dir: {base_dir}")

    all_model_results = {}
    model_times = {}
    model_bases = {}  # label -> base_dir for DB reopening

    for model_name, label in MODELS:
        print(f"\n--- {label} ({model_name}) ---")
        try:
            db, elapsed, model_base = index_with_model(model_name, label, base_dir)
            model_times[label] = elapsed
            model_bases[label] = model_base

            # Query with same base_dir so we hit the right DB
            ProjectDB.base_dir = model_base
            query_client = FastembedClient(model_name=model_name)
            results = run_queries(db, query_client, QUERIES)
            all_model_results[label] = results
            query_client.close()
            db.close()
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_model_results[label] = None
            model_times[label] = -1

    # Reset base_dir
    ProjectDB.base_dir = None

    # --- Report ---
    labels = [l for _, l in MODELS if all_model_results.get(l)]
    if not labels:
        print("\nNo models completed successfully.")
        return

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    header = f"{'Metric':<25}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("-" * len(header))

    # Top-1 accuracy
    row = f"{'Top-1 Accuracy':<25}"
    for label in labels:
        results = all_model_results[label]
        acc = sum(1 for r in results if r["top1"]) / len(results) * 100
        row += f"{acc:>13.0f}%"
    print(row)

    # Top-3 accuracy
    row = f"{'Top-3 Accuracy':<25}"
    for label in labels:
        results = all_model_results[label]
        acc = sum(1 for r in results if r["top3"]) / len(results) * 100
        row += f"{acc:>13.0f}%"
    print(row)

    # Top-5 accuracy
    row = f"{'Top-5 Accuracy':<25}"
    for label in labels:
        results = all_model_results[label]
        acc = sum(1 for r in results if r["top5"]) / len(results) * 100
        row += f"{acc:>13.0f}%"
    print(row)

    # Top-10 accuracy
    row = f"{'Top-10 Accuracy':<25}"
    for label in labels:
        results = all_model_results[label]
        acc = sum(1 for r in results if r["top10"]) / len(results) * 100
        row += f"{acc:>13.0f}%"
    print(row)

    # MRR
    row = f"{'Mean Reciprocal Rank':<25}"
    for label in labels:
        results = all_model_results[label]
        mrr = sum(r["mrr"] for r in results) / len(results)
        row += f"{mrr:>14.3f}"
    print(row)

    # Index time
    row = f"{'Index Time (s)':<25}"
    for label in labels:
        row += f"{model_times[label]:>13.1f}s"
    print(row)

    # Per-query breakdown
    print(f"\n{'PER-QUERY BREAKDOWN'}")
    print("-" * 80)
    for i, (query, expected, desc) in enumerate(QUERIES):
        print(f"\n  Q{i+1}: {desc}")
        print(f"      Expected in: {expected}")
        for label in labels:
            results = all_model_results[label]
            r = results[i]
            rank_str = "MISS" if r["mrr"] == 0 else f"rank {int(1/r['mrr'])}"
            marker = "+" if r["top3"] else ("-" if r["top10"] else "X")
            print(f"    [{marker}] {label:<15} → {r['top_file']:<30} {rank_str:<10} score={r['top_score']:.3f}")

    # Save raw results
    output_path = os.path.join(PROJECT_PATH, "docs", "research", "embedding-benchmark-results.json")
    with open(output_path, "w") as f:
        json.dump({
            "models": {l: m for m, l in MODELS},
            "times": model_times,
            "results": {l: r for l, r in all_model_results.items() if r},
        }, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")

    # Cleanup
    shutil.rmtree(base_dir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
