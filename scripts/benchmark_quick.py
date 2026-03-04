"""Quick benchmark: Gateway-Nomic only, VEC vs HYBRID comparison.

Validates that HYBRID mode no longer degrades results after RRF weight rebalance
and scripts/ directory exclusion.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tessera.db import ProjectDB
from tessera.embeddings import EmbeddingClient
from tessera.indexer import IndexerPipeline
from tessera.search import SearchType, hybrid_search

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

QUERIES = [
    ("hybrid search combining keyword and semantic results", ["search.py"], "Core hybrid search"),
    ("embedding client with caching", ["embeddings.py"], "Embedding client"),
    ("tree-sitter AST parsing for multiple languages", ["parser.py"], "Parser orchestration"),
    ("scope-gated authentication tokens", ["auth.py"], "Auth/scope system"),
    ("incremental re-indexing changed files", ["_pipeline.py"], "Incremental indexer"),
    ("SQLite FTS5 full text search", ["_project.py", "search.py"], "FTS5 integration"),
    ("reciprocal rank fusion merge", ["search.py"], "RRF algorithm"),
    ("FAISS vector similarity index", ["_project.py"], "FAISS index management"),
    ("MCP server tool registration", ["_app.py"], "MCP server setup"),
    ("PageRank graph scoring for code symbols", ["ppr.py"], "PPR/graph ranking"),
    ("chunk code into overlapping windows", ["chunker.py"], "Code chunking"),
    ("normalize BM25 scores to 0-1 range", ["search.py"], "Score normalization"),
    ("cross-encoder reranking after search", ["embeddings.py", "_search.py"], "Reranker integration"),
    ("extract code snippet with ancestor context", ["search.py"], "Snippet extraction"),
    ("session token creation and validation", ["auth.py"], "Session management"),
]


def run_search(db, client, queries, search_types=None):
    results = []
    for query, expected_files, desc in queries:
        raw = client.embed_query(query)
        query_embedding = np.array(raw, dtype=np.float32)

        hits = hybrid_search(
            query, query_embedding, db,
            graph=None, limit=10,
            source_type=None, search_types=search_types,
            advanced_fts=False, rrf_weights=None,
        )

        hit_files = [os.path.basename(h.get("file_path", "")) for h in hits]

        top1 = any(exp in hit_files[0] for exp in expected_files) if hits else False
        top3 = any(any(exp in f for exp in expected_files) for f in hit_files[:3])
        top5 = any(any(exp in f for exp in expected_files) for f in hit_files[:5])
        top10 = any(any(exp in f for exp in expected_files) for f in hit_files[:10])

        mrr = 0.0
        for rank, f in enumerate(hit_files, 1):
            if any(exp in f for exp in expected_files):
                mrr = 1.0 / rank
                break

        results.append({
            "desc": desc, "top1": top1, "top3": top3, "top5": top5, "top10": top10,
            "mrr": mrr, "top_file": hit_files[0] if hits else "—",
            "top3_files": hit_files[:3],
        })
    return results


def print_results(label, results):
    n = len(results)
    top1 = sum(1 for r in results if r["top1"]) / n * 100
    top3 = sum(1 for r in results if r["top3"]) / n * 100
    top5 = sum(1 for r in results if r["top5"]) / n * 100
    top10 = sum(1 for r in results if r["top10"]) / n * 100
    mrr = sum(r["mrr"] for r in results) / n
    print(f"\n  {label}:")
    print(f"    Top-1: {top1:.0f}%  Top-3: {top3:.0f}%  Top-5: {top5:.0f}%  Top-10: {top10:.0f}%  MRR: {mrr:.3f}")

    for i, r in enumerate(results):
        rank_str = "MISS" if r["mrr"] == 0 else f"rank {int(1/r['mrr'])}"
        marker = "+" if r["top3"] else ("-" if r["top10"] else "X")
        top3 = ", ".join(r["top3_files"])
        print(f"    [{marker}] Q{i+1}: {r['desc']:<22} {rank_str:<8} [{top3}]")


def main():
    print("=" * 70)
    print("Quick Benchmark: Gateway-Nomic — VEC vs HYBRID")
    print("=" * 70)

    base_dir = tempfile.mkdtemp(prefix="tessera_quick_bench_")
    print(f"Working dir: {base_dir}")

    client = EmbeddingClient(
        endpoint="http://localhost:8800/v1/embeddings",
        model="nomic-embed",
    )

    model_base = os.path.join(base_dir, "gateway")
    os.makedirs(model_base, exist_ok=True)
    ProjectDB.base_dir = model_base

    pipeline = IndexerPipeline(project_path=PROJECT_PATH, embedding_client=client)
    pipeline.register()

    start = time.perf_counter()
    stats = pipeline.index_project()
    elapsed = time.perf_counter() - start
    print(f"\nIndexed: {stats.files_processed} files, {stats.chunks_created} chunks, "
          f"{stats.chunks_embedded} embedded in {elapsed:.1f}s")

    db = ProjectDB(PROJECT_PATH)

    print("\nRunning VEC-only queries...")
    vec_results = run_search(db, client, QUERIES, search_types=[SearchType.VEC])

    print("Running HYBRID queries...")
    hybrid_results = run_search(db, client, QUERIES, search_types=None)

    print_results("VEC-ONLY", vec_results)
    print_results("HYBRID", hybrid_results)

    client.close()
    db.close()
    ProjectDB.base_dir = None
    shutil.rmtree(base_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
