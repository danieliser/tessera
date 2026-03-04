"""Benchmark embedding models using VECTOR-ONLY search (no FTS5 keyword mixing).

Isolates the semantic embedding quality by bypassing keyword search entirely.
This tells us whether model differences are real or masked by FTS5 dominance.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tessera.db import ProjectDB
from tessera.embeddings import FastembedClient
from tessera.search import hybrid_search, SearchType

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Use the DBs already built by the previous benchmark
BENCH_DIR = None  # Will be auto-detected

MODELS = [
    ("BAAI/bge-small-en-v1.5", "BGE-Small"),
    ("BAAI/bge-base-en-v1.5", "BGE-Base"),
    ("jinaai/jina-embeddings-v2-base-code", "Jina-Code"),
    ("nomic-ai/nomic-embed-text-v1.5", "Nomic-1.5"),
    ("snowflake/snowflake-arctic-embed-m", "Arctic-M"),
]

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


def find_bench_dir() -> str:
    """Find the most recent benchmark temp directory."""
    import glob
    candidates = glob.glob("/var/folders/43/yjckkxz56_578nc9tpxhg4yh0000gn/T/tessera_bench_*")
    # Pick the one with the most subdirs (most models indexed)
    best = None
    best_count = 0
    for c in candidates:
        subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
        if len(subdirs) > best_count:
            best = c
            best_count = len(subdirs)
    if not best:
        raise FileNotFoundError("No benchmark directories found. Run benchmark_embeddings.py first.")
    print(f"Using existing benchmark dir: {best} ({best_count} models)")
    return best


def run_vec_only(db: ProjectDB, client: FastembedClient, queries: list) -> list[dict]:
    """Run vector-only search (SearchType.VEC) to isolate embedding quality."""
    results = []
    for query, expected_files, desc in queries:
        raw = client.embed_query(query)
        query_embedding = np.array(raw, dtype=np.float32)

        # Force VEC-only search — no FTS5
        hits = hybrid_search(
            query, query_embedding, db,
            graph=None, limit=10,
            source_type=None,
            search_types=[SearchType.VEC],
            advanced_fts=False, rrf_weights=None,
        )

        hit_files = [os.path.basename(h.get("file_path", "")) for h in hits]
        hit_paths = [h.get("file_path", "") for h in hits]

        top1_match = any(exp in hit_files[0] for exp in expected_files) if hits else False
        top3_match = any(any(exp in f for exp in expected_files) for f in hit_files[:3])
        top5_match = any(any(exp in f for exp in expected_files) for f in hit_files[:5])
        top10_match = any(any(exp in f for exp in expected_files) for f in hit_files[:10])

        mrr = 0.0
        for rank, f in enumerate(hit_files, 1):
            if any(exp in f for exp in expected_files):
                mrr = 1.0 / rank
                break

        top_score = hits[0].get("score", 0) if hits else 0

        results.append({
            "desc": desc,
            "top1": top1_match,
            "top3": top3_match,
            "top5": top5_match,
            "top10": top10_match,
            "mrr": mrr,
            "top_score": top_score,
            "top_file": hit_files[0] if hits else "—",
            "top3_files": hit_files[:3],
        })

    return results


def run_lex_only(db: ProjectDB, queries: list) -> list[dict]:
    """Run keyword-only search (SearchType.LEX) as baseline."""
    results = []
    for query, expected_files, desc in queries:
        hits = hybrid_search(
            query, None, db,
            graph=None, limit=10,
            source_type=None,
            search_types=[SearchType.LEX],
            advanced_fts=False, rrf_weights=None,
        )

        hit_files = [os.path.basename(h.get("file_path", "")) for h in hits]

        top1_match = any(exp in hit_files[0] for exp in expected_files) if hits else False
        top3_match = any(any(exp in f for exp in expected_files) for f in hit_files[:3])
        top5_match = any(any(exp in f for exp in expected_files) for f in hit_files[:5])
        top10_match = any(any(exp in f for exp in expected_files) for f in hit_files[:10])

        mrr = 0.0
        for rank, f in enumerate(hit_files, 1):
            if any(exp in f for exp in expected_files):
                mrr = 1.0 / rank
                break

        top_score = hits[0].get("score", 0) if hits else 0

        results.append({
            "desc": desc,
            "top1": top1_match,
            "top3": top3_match,
            "top5": top5_match,
            "top10": top10_match,
            "mrr": mrr,
            "top_score": top_score,
            "top_file": hit_files[0] if hits else "—",
            "top3_files": hit_files[:3],
        })

    return results


def main():
    bench_dir = find_bench_dir()

    print("=" * 90)
    print("Vector-Only vs Keyword-Only Search Comparison")
    print("=" * 90)
    print(f"Isolating semantic signal by running VEC-only (no FTS5)\n")

    all_results = {}

    # First: keyword-only baseline (use any model's DB — FTS5 is identical)
    first_label = MODELS[0][1]
    first_base = os.path.join(bench_dir, first_label)
    if os.path.isdir(first_base):
        ProjectDB.base_dir = first_base
        db = ProjectDB(PROJECT_PATH)
        print("--- LEX-only (keyword baseline) ---")
        lex_results = run_lex_only(db, QUERIES)
        all_results["LEX-only"] = lex_results
        db.close()

    # Then: vec-only for each model
    for model_name, label in MODELS:
        model_base = os.path.join(bench_dir, label)
        if not os.path.isdir(model_base):
            print(f"  Skipping {label} — no index found")
            continue

        ProjectDB.base_dir = model_base
        db = ProjectDB(PROJECT_PATH)
        client = FastembedClient(model_name=model_name)

        print(f"--- VEC-only: {label} ---")
        vec_results = run_vec_only(db, client, QUERIES)
        all_results[f"VEC:{label}"] = vec_results

        client.close()
        db.close()

    ProjectDB.base_dir = None

    # --- Summary Table ---
    labels = list(all_results.keys())
    print("\n" + "=" * 90)
    print("RESULTS: Vector-Only vs Keyword-Only")
    print("=" * 90)

    header = f"{'Metric':<22}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("-" * len(header))

    for metric_name, key in [("Top-1 Accuracy", "top1"), ("Top-3 Accuracy", "top3"),
                              ("Top-5 Accuracy", "top5"), ("Top-10 Accuracy", "top10")]:
        row = f"{metric_name:<22}"
        for label in labels:
            results = all_results[label]
            acc = sum(1 for r in results if r[key]) / len(results) * 100
            row += f"{acc:>13.0f}%"
        print(row)

    row = f"{'MRR':<22}"
    for label in labels:
        results = all_results[label]
        mrr = sum(r["mrr"] for r in results) / len(results)
        row += f"{mrr:>14.3f}"
    print(row)

    # Per-query detail
    print(f"\n{'PER-QUERY DETAIL'}")
    print("-" * 90)

    for i, (query, expected, desc) in enumerate(QUERIES):
        print(f"\n  Q{i+1}: {desc} — expected: {expected}")
        for label in labels:
            r = all_results[label][i]
            rank_str = "MISS" if r["mrr"] == 0 else f"rank {int(1/r['mrr'])}"
            marker = "+" if r["top3"] else ("-" if r["top10"] else "X")
            top3 = ", ".join(r["top3_files"])
            print(f"    [{marker}] {label:<22} {rank_str:<8} top3=[{top3}]")

    # Save
    output_path = os.path.join(PROJECT_PATH, "docs", "research", "vec-only-benchmark-results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
