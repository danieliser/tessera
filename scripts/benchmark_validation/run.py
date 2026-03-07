"""Validation benchmark runner — multi-codebase, multi-tier search evaluation.

Usage:
    uv run python scripts/benchmark_validation/run.py --tier quick
    uv run python scripts/benchmark_validation/run.py --tier standard --codebase nextjs
    uv run python scripts/benchmark_validation/run.py --tier full --reranker-local
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tessera.db import ProjectDB
from tessera.embeddings import EmbeddingClient, FastembedClient, FastembedReranker, HTTPReranker
from tessera.search import hybrid_search

ENDPOINT = "http://localhost:8800/v1/embeddings"
RERANK_ENDPOINT = "http://localhost:8800/v1/rerank"

# Codebase configs: (git_url, ref, paths_to_index, query_module)
CODEBASES = {
    "nextjs": {
        "git_url": "https://github.com/vercel/next.js.git",
        "ref": "v16.1.6",
        "paths": ["packages/next/src", "docs"],
        "queries": "queries_nextjs",
    },
}

BENCH_ROOT = os.path.expanduser("~/.tessera/benchmarks/validation")


def ensure_codebase(name: str) -> str:
    """Clone/checkout the codebase at the pinned ref. Returns local path."""
    config = CODEBASES[name]
    local_dir = os.path.join(BENCH_ROOT, "repos", name)

    if os.path.isdir(local_dir):
        # Verify correct ref
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=local_dir, capture_output=True, text=True,
        )
        if result.returncode == 0 and config["ref"] in result.stdout.strip():
            print(f"  Codebase {name}: already at {config['ref']}")
            return local_dir

    if not os.path.isdir(local_dir):
        print(f"  Cloning {name} (sparse, {config['ref']})...")
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", config["ref"],
            "--sparse", config["git_url"], local_dir,
        ], check=True, capture_output=True)

    # Sparse checkout only the paths we need
    subprocess.run(
        ["git", "sparse-checkout", "set"] + config["paths"],
        cwd=local_dir, check=True, capture_output=True,
    )
    print(f"  Codebase {name}: ready at {config['ref']}")
    return local_dir


def index_codebase(name: str, codebase_path: str, embed_client, model_key: str,
                   reindex: bool = False) -> ProjectDB:
    """Index a codebase with Tessera. Returns the ProjectDB."""
    config = CODEBASES[name]
    index_dir = os.path.join(BENCH_ROOT, model_key, name)
    os.makedirs(index_dir, exist_ok=True)

    ProjectDB.base_dir = index_dir
    db = ProjectDB(codebase_path)

    # Check if already indexed
    stamp = os.path.join(index_dir, ".indexed")
    if os.path.exists(stamp) and not reindex:
        print(f"  Index {name}/{model_key}: cached")
        return db

    # Drop a .tesseraignore to exclude noise files
    ignore_path = os.path.join(codebase_path, ".tesseraignore")
    if not os.path.exists(ignore_path):
        with open(ignore_path, "w") as f:
            f.write("pnpm-lock.yaml\nyarn.lock\npackage-lock.json\n")
            f.write("**/compiled/\n**/vendored/\n")
            f.write("*.wasm\n*.woff2\n*.ttf\n*.css\n")
            f.write("errors.json\n")

    print(f"  Indexing {name} with {model_key}...")
    from tessera.indexer import IndexerPipeline
    indexer = IndexerPipeline(
        project_path=codebase_path,
        project_db=db,
        embedding_client=embed_client,
        languages=["typescript", "javascript"],
    )
    stats = indexer.index_project_sync()

    with open(stamp, "w") as f:
        f.write(f"{config['ref']}\n")

    print(f"  Index {name}/{model_key}: done ({stats.chunks_created} chunks, "
          f"{stats.files_processed} files)")
    return db


def evaluate_hits(hit_files: list[str], expected: list[str]) -> dict:
    """Compute MRR and top-k scores."""
    first_hit_rank = None
    for i, path in enumerate(hit_files[:10]):
        basename = os.path.basename(path)
        for exp in expected:
            if exp in basename or exp in path:
                if first_hit_rank is None:
                    first_hit_rank = i + 1
                break

    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
    top1 = 1 if first_hit_rank == 1 else 0
    top3 = 1 if first_hit_rank and first_hit_rank <= 3 else 0
    top5 = 1 if first_hit_rank and first_hit_rank <= 5 else 0
    top10 = 1 if first_hit_rank and first_hit_rank <= 10 else 0

    return {"mrr": mrr, "top1": top1, "top3": top3, "top5": top5, "top10": top10}


def run_queries(queries, db, embed_client, reranker, rerank_pool: int = 40,
                smart_routing: bool = False, graph=None,
                keyword_weight: float | None = None):
    """Run queries against a single DB, return per-query results."""
    results = []

    for query_text, expected_files, desc, category, tier in queries:
        t0 = time.perf_counter()

        raw = embed_client.embed_query(query_text)
        query_embedding = np.array(raw, dtype=np.float32)

        # Smart routing: filter code queries to code chunks only
        # Use larger pool for code (more candidates needed), smaller for doc
        source_type = None
        pool = rerank_pool
        if smart_routing:
            if category == "code":
                source_type = ["code"]
                pool = max(rerank_pool, 100)

        hits = hybrid_search(
            query_text, query_embedding, db,
            graph=graph,
            limit=pool, advanced_fts=False,
            source_type=source_type,
            keyword_weight=keyword_weight,
        )

        # Rerank
        if reranker and hits:
            docs = [h.get("content", "")[:512] for h in hits[:rerank_pool]]
            reranked = reranker.rerank(query_text, docs, top_k=10)
            if reranked and reranked[0][1] > 0:
                hits = [hits[idx] for idx, _ in reranked if idx < len(hits)]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        hit_files = [h.get("file_path", "") for h in hits]
        scores = evaluate_hits(hit_files, expected_files)
        scores["desc"] = desc
        scores["category"] = category
        scores["tier"] = tier
        scores["latency_ms"] = elapsed_ms
        scores["top_file"] = os.path.basename(hit_files[0]) if hit_files else "—"

        # Print per-query result
        rank_str = f"rank {int(1/scores['mrr'])}" if scores["mrr"] > 0 else "MISS"
        status = "[+]" if scores["mrr"] >= 0.5 else "[-]" if scores["mrr"] > 0 else "[X]"
        print(f"  {status} {desc:50s} {rank_str:8s} [{scores['top_file']}]")

        results.append(scores)

    return results


def print_summary(results: list[dict], label: str):
    """Print category-level summary."""
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    total_mrr = 0
    total_n = 0
    for cat in ["code", "doc", "cross"]:
        if cat not in categories:
            continue
        cat_results = categories[cat]
        mrr = sum(r["mrr"] for r in cat_results) / len(cat_results)
        top1 = sum(r["top1"] for r in cat_results) / len(cat_results)
        top3 = sum(r["top3"] for r in cat_results) / len(cat_results)
        top5 = sum(r["top5"] for r in cat_results) / len(cat_results)
        top10 = sum(r["top10"] for r in cat_results) / len(cat_results)
        avg_ms = sum(r["latency_ms"] for r in cat_results) / len(cat_results)
        print(f"  {cat:6s}  MRR: {mrr:.3f}  Top1: {top1:.0%}  Top3: {top3:.0%}  "
              f"Top5: {top5:.0%}  Top10: {top10:.0%}  "
              f"({len(cat_results)} queries, {avg_ms:.0f}ms avg)")
        total_mrr += mrr
        total_n += 1

    if total_n:
        blended = total_mrr / total_n
        overall_mrr = sum(r["mrr"] for r in results) / len(results)
        print(f"  {'BLEND':6s}  MRR: {blended:.3f}  (overall: {overall_mrr:.3f}, "
              f"{len(results)} queries)")


def main():
    parser = argparse.ArgumentParser(description="Tessera validation benchmark")
    parser.add_argument("--tier", default="quick",
                        choices=["quick", "standard", "full"])
    parser.add_argument("--codebase", default="nextjs",
                        choices=list(CODEBASES.keys()))
    parser.add_argument("--model", default="bge-small",
                        choices=["bge-small", "bge-base", "nomic-embed",
                                 "jina-code", "nomic-text"])
    parser.add_argument("--http", action="store_true",
                        help="Use HTTP embedding endpoint instead of local fastembed")
    parser.add_argument("--reranker", default="Xenova/ms-marco-MiniLM-L-6-v2",
                        help="Reranker model name")
    parser.add_argument("--reranker-local", action="store_true",
                        help="Use FastembedReranker (local ONNX)")
    parser.add_argument("--reranker-http", action="store_true",
                        help="Use HTTP reranker via gateway")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--rerank-pool", type=int, default=40)
    parser.add_argument("--smart-routing", action="store_true",
                        help="Filter by source type per query category")
    parser.add_argument("--graph", action="store_true",
                        help="Enable PPR graph boost in hybrid search")
    parser.add_argument("--keyword-weight", type=float, default=None,
                        help="FTS5 keyword weight override (default: auto)")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Tessera Validation Benchmark — {args.codebase} / {args.tier}")
    print("=" * 80)

    # Load queries
    import importlib
    query_mod = importlib.import_module(
        f"benchmark_validation.{CODEBASES[args.codebase]['queries']}"
    )
    queries = query_mod.get_queries(args.tier)
    n_code = sum(1 for q in queries if q[3] == "code")
    n_doc = sum(1 for q in queries if q[3] == "doc")
    n_cross = sum(1 for q in queries if q[3] == "cross")
    print(f"  Queries: {len(queries)} ({n_code} code, {n_doc} doc, {n_cross} cross)")

    # Embedding client
    MODELS = {
        "bge-small": ("BAAI/bge-small-en-v1.5", "BGE-small-384d"),
        "bge-base": ("BAAI/bge-base-en-v1.5", "BGE-base-768d"),
        "nomic-embed": ("nomic-embed", "Nomic-Embed-Base"),
        "jina-code": ("jinaai/jina-embeddings-v2-base-code", "Jina-Code-768d"),
        "nomic-text": ("nomic-ai/nomic-embed-text-v1.5", "Nomic-Text-v1.5-768d"),
    }
    model_name, model_label = MODELS[args.model]
    if args.http:
        embed_client = EmbeddingClient(endpoint=ENDPOINT, model=model_name)
        model_label += " (HTTP)"
    else:
        embed_client = FastembedClient(model_name=model_name)
    _ = embed_client.embed_single("warmup")
    print(f"  Embedding: {model_label}")

    # Reranker
    reranker = None
    if not args.no_reranker:
        if args.reranker_http:
            reranker = HTTPReranker(endpoint=RERANK_ENDPOINT, model=args.reranker)
            print(f"  Reranker: {args.reranker} (HTTP)")
        else:
            reranker = FastembedReranker(model_name=args.reranker)
            print(f"  Reranker: {args.reranker} (local ONNX)")
        # Warmup
        try:
            reranker.rerank("test", ["test doc"], top_k=1)
        except Exception as e:
            print(f"  Reranker warmup failed: {e}")
            reranker = None

    # Ensure codebase
    print(f"\n  Setting up codebase...")
    codebase_path = ensure_codebase(args.codebase)

    # Index
    print(f"  Setting up index...")
    db = index_codebase(
        args.codebase, codebase_path, embed_client,
        model_key=args.model, reindex=args.reindex,
    )

    # Load graph if requested
    graph = None
    if args.graph:
        from tessera.graph import load_project_graph
        try:
            graph = load_project_graph(db, project_id=1)
            lcc = graph.largest_cc_size / graph.n_symbols if graph.n_symbols else 0
            print(f"  Graph: {graph.n_symbols} symbols, {graph.edge_count} edges, "
                  f"LCC {lcc:.1%}")
        except (ValueError, Exception) as e:
            print(f"  Graph: failed to load ({e})")

    # Run
    print(f"\n  Running {len(queries)} queries...\n")
    results = run_queries(queries, db, embed_client, reranker,
                          rerank_pool=args.rerank_pool,
                          smart_routing=args.smart_routing,
                          graph=graph,
                          keyword_weight=args.keyword_weight)

    # Summary
    reranker_label = args.reranker.split("/")[-1] if not args.no_reranker else "none"
    label = f"{args.model} + {reranker_label} | {args.codebase} / {args.tier}"
    print_summary(results, label)


if __name__ == "__main__":
    main()
