"""Clean embedding benchmark: HTTP gateway vs fastembed models, site/ excluded.

Re-indexes with site/ directory excluded, compares vec-only and hybrid modes.
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
from tessera.embeddings import EmbeddingClient, FastembedClient
from tessera.indexer import IndexerPipeline
from tessera.search import SearchType, hybrid_search

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS = [
    ("http", "nomic-embed", "Gateway-Nomic", "http://localhost:8800/v1/embeddings"),
    ("fastembed", "BAAI/bge-small-en-v1.5", "BGE-Small", None),
    ("fastembed", "nomic-ai/nomic-embed-text-v1.5", "Nomic-FE", None),
    ("fastembed", "jinaai/jina-embeddings-v2-base-code", "Jina-Code", None),
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


def create_client(provider, model_name, endpoint):
    if provider == "http":
        return EmbeddingClient(endpoint=endpoint, model=model_name)
    return FastembedClient(model_name=model_name)


def index_project(client, label, base_dir):
    model_base = os.path.join(base_dir, label)
    os.makedirs(model_base, exist_ok=True)
    ProjectDB.base_dir = model_base

    pipeline = IndexerPipeline(project_path=PROJECT_PATH, embedding_client=client)
    pipeline.register()

    start = time.perf_counter()
    stats = pipeline.index_project()
    elapsed = time.perf_counter() - start

    print(f"  {label}: {stats.files_processed} files, {stats.chunks_created} chunks, "
          f"{stats.chunks_embedded} embedded in {elapsed:.1f}s")

    db = ProjectDB(PROJECT_PATH)
    return db, elapsed, model_base


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


def print_table(all_results, labels, title):
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")

    header = f"{'Metric':<22}" + "".join(f"{l:>16}" for l in labels)
    print(header)
    print("-" * len(header))

    for metric_name, key in [("Top-1 Accuracy", "top1"), ("Top-3 Accuracy", "top3"),
                              ("Top-5 Accuracy", "top5"), ("Top-10 Accuracy", "top10")]:
        row = f"{metric_name:<22}"
        for label in labels:
            r = all_results[label]
            acc = sum(1 for x in r if x[key]) / len(r) * 100
            row += f"{acc:>15.0f}%"
        print(row)

    row = f"{'MRR':<22}"
    for label in labels:
        r = all_results[label]
        mrr = sum(x["mrr"] for x in r) / len(r)
        row += f"{mrr:>16.3f}"
    print(row)

    print(f"\n{'PER-QUERY'}")
    print("-" * 90)
    for i, (_, expected, desc) in enumerate(QUERIES):
        print(f"\n  Q{i+1}: {desc} — expected: {expected}")
        for label in labels:
            r = all_results[label][i]
            rank_str = "MISS" if r["mrr"] == 0 else f"rank {int(1/r['mrr'])}"
            marker = "+" if r["top3"] else ("-" if r["top10"] else "X")
            top3 = ", ".join(r["top3_files"])
            print(f"    [{marker}] {label:<16} {rank_str:<8} [{top3}]")


def main():
    print("=" * 90)
    print("CLEAN Embedding Benchmark (site/ excluded)")
    print("=" * 90)

    base_dir = tempfile.mkdtemp(prefix="tessera_clean_bench_")
    print(f"Working dir: {base_dir}\n")

    vec_results = {}
    hybrid_results = {}
    labels = []

    for provider, model_name, label, endpoint in MODELS:
        print(f"\n--- {label} ({model_name}) ---")
        try:
            client = create_client(provider, model_name, endpoint)
            db, elapsed, model_base = index_project(client, label, base_dir)

            ProjectDB.base_dir = model_base

            print(f"  Running VEC-only queries...")
            vec_results[label] = run_search(db, client, QUERIES, search_types=[SearchType.VEC])

            print(f"  Running HYBRID queries...")
            hybrid_results[label] = run_search(db, client, QUERIES, search_types=None)

            labels.append(label)
            client.close()
            db.close()
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    ProjectDB.base_dir = None

    if not labels:
        print("No models completed.")
        return

    print_table(vec_results, labels, "VECTOR-ONLY RESULTS (semantic signal isolation)")
    print_table(hybrid_results, labels, "HYBRID RESULTS (keyword + semantic)")

    shutil.rmtree(base_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
