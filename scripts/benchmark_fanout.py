"""Fan-out benchmark: dual-model search with reranker fusion.

Runs each query through TWO embedding models simultaneously,
merges candidates, and lets the cross-encoder reranker pick winners.

- CodeRankEmbed (137M, code-specialized) → code candidates
- BGE-small (67MB, general) → doc/general candidates
- Jina reranker → final ordering

Uses pre-built indexes from benchmark_mixed.py runs.

Usage:
    uv run python scripts/benchmark_fanout.py
    uv run python scripts/benchmark_fanout.py --smart   # route by category
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tessera.db import ProjectDB
from tessera.embeddings import EmbeddingClient, FastembedClient, HTTPReranker
from tessera.search import hybrid_search, SearchType

# Reuse queries and helpers from the mixed benchmark
from benchmark_mixed import (
    QUERIES, PM_CORE, PM_PRO, PM_DOCS,
    evaluate_hits, try_load_graph, print_comparison,
)

ENDPOINT = "http://localhost:8800/v1/embeddings"
RERANK_ENDPOINT = "http://localhost:8800/v1/rerank"


def load_dbs(model_key: str):
    """Load pre-built indexes for a model."""
    bench_root = os.path.expanduser(f"~/.tessera/benchmarks/mixed_{model_key}")
    targets = [("core", PM_CORE), ("pro", PM_PRO), ("docs", PM_DOCS)]
    dbs = []
    for label, path in targets:
        idx_base = os.path.join(bench_root, label)
        if not os.path.isdir(idx_base):
            print(f"  ERROR: No index for {model_key}/{label}. Run benchmark_mixed.py first.")
            return None
        ProjectDB.base_dir = idx_base
        db = ProjectDB(path)
        dbs.append((label, db))
    return dbs


def search_with_model(query, client, dbs, search_types=None, src_filter=None,
                       use_hyde=False, keyword_weight=None, graphs=None):
    """Search all DBs with a single model, return raw hits."""
    raw = client.embed_single(query) if use_hyde else client.embed_query(query)
    query_embedding = np.array(raw, dtype=np.float32)

    all_hits = []
    for label, db in dbs:
        graph = graphs.get(label) if graphs else None
        hits = hybrid_search(
            query, query_embedding, db,
            graph=graph, limit=10,
            source_type=src_filter, search_types=search_types,
            advanced_fts=False, keyword_weight=keyword_weight,
        )
        for h in hits:
            h["_source"] = label
            all_hits.append(h)
    return all_hits


def run_fanout(code_client, code_dbs, gen_client, gen_dbs, reranker,
               mode="fanout", graphs_code=None, graphs_gen=None):
    """Fan-out: search both models, merge candidates, rerank."""
    results = []
    total_ms = 0.0

    for query, expected_files, desc, category in QUERIES:
        t0 = time.perf_counter()

        if mode == "smart":
            # Smart routing: both models for code, general-only for docs, both for cross
            if category == "code":
                # Fan-out: both models generate candidates, reranker picks
                code_hits = search_with_model(
                    query, code_client, code_dbs,
                    search_types=[SearchType.VEC], src_filter=["code"],
                    use_hyde=False, graphs=graphs_code,
                )
                gen_hits = search_with_model(
                    query, gen_client, gen_dbs,
                    search_types=[SearchType.VEC], src_filter=["code"],
                    use_hyde=False, graphs=graphs_gen,
                )
            elif category == "doc":
                code_hits = []
                gen_hits = search_with_model(
                    query, gen_client, gen_dbs,
                    search_types=[SearchType.VEC], src_filter=["markdown", "json"],
                    use_hyde=True, graphs=graphs_gen,
                )
            else:  # cross — code model only with hybrid + reranker
                code_hits = search_with_model(
                    query, code_client, code_dbs,
                    use_hyde=False, graphs=graphs_code,
                )
                gen_hits = []
        else:
            # Pure fan-out: BOTH models, no filter, merge everything
            code_hits = search_with_model(
                query, code_client, code_dbs,
                search_types=[SearchType.VEC], use_hyde=False,
                graphs=graphs_code,
            )
            gen_hits = search_with_model(
                query, gen_client, gen_dbs,
                use_hyde=False, graphs=graphs_gen,
            )

        # Interleave: alternate code/gen hits, dedup by file_path+content
        seen = set()
        merged = []
        code_iter = iter(code_hits)
        gen_iter = iter(gen_hits)
        code_done = gen_done = False

        while len(merged) < 30:
            if not code_done:
                try:
                    h = next(code_iter)
                    key = (h.get("file_path", ""), h.get("content", "")[:100])
                    if key not in seen:
                        seen.add(key)
                        h["_model"] = "code"
                        merged.append(h)
                except StopIteration:
                    code_done = True
            if not gen_done:
                try:
                    h = next(gen_iter)
                    key = (h.get("file_path", ""), h.get("content", "")[:100])
                    if key not in seen:
                        seen.add(key)
                        h["_model"] = "gen"
                        merged.append(h)
                except StopIteration:
                    gen_done = True
            if code_done and gen_done:
                break

        # Rerank the merged candidates
        if reranker and merged:
            docs = [h.get("content", "")[:512] for h in merged[:20]]
            reranked = reranker.rerank(query, docs, top_k=10)
            if reranked and reranked[0][1] > 0:
                reordered = []
                for orig_idx, score in reranked:
                    if orig_idx < len(merged):
                        item = merged[orig_idx]
                        item["rerank_score"] = score
                        reordered.append(item)
                merged = reordered

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        hit_files = [h.get("file_path", "") for h in merged]
        scores = evaluate_hits(hit_files, expected_files)
        scores["desc"] = desc
        scores["category"] = category
        scores["top_file"] = os.path.basename(hit_files[0]) if hit_files else "—"
        scores["top3_files"] = [os.path.basename(f) for f in hit_files[:3]]
        scores["latency_ms"] = elapsed_ms
        results.append(scores)

    avg_ms = total_ms / len(QUERIES)
    return results, avg_ms


def main():
    parser = argparse.ArgumentParser(description="Fan-out dual-model benchmark")
    parser.add_argument("--smart", action="store_true",
                        help="Smart routing: code model for code queries, general for docs")
    parser.add_argument("--code-model", default="coderank",
                        choices=["coderank"], help="Code-specialized model")
    parser.add_argument("--gen-model", default="bge-small",
                        choices=["bge-small", "bge-base"], help="General-purpose model")
    parser.add_argument("--gen-provider", default="fastembed",
                        choices=["fastembed", "http"])
    args = parser.parse_args()

    print("=" * 100)
    print("Fan-Out Benchmark: Dual-Model Search + Reranker Fusion")
    print("=" * 100)

    # Code model (always HTTP)
    code_client = EmbeddingClient(endpoint=ENDPOINT, model="code-rank-embed")
    try:
        v = code_client.embed_single("test")
        print(f"  Code model: CodeRankEmbed — {len(v)}d (HTTP)")
    except Exception as e:
        print(f"  ERROR: CodeRankEmbed unavailable ({e})")
        return

    # General model
    GEN_FASTEMBED = {
        "bge-small": ("BAAI/bge-small-en-v1.5", "BGE-small-384d"),
        "bge-base": ("BAAI/bge-base-en-v1.5", "BGE-base-768d"),
    }
    if args.gen_provider == "fastembed":
        model_name, model_label = GEN_FASTEMBED[args.gen_model]
        gen_client = FastembedClient(model_name=model_name)
        _ = gen_client.embed_single("warmup")
    else:
        gen_client = EmbeddingClient(endpoint=ENDPOINT, model="nomic-embed")
        model_label = "Nomic-768d"
    v = gen_client.embed_single("test")
    print(f"  General model: {model_label} — {len(v)}d")

    # Reranker
    reranker = HTTPReranker(endpoint=RERANK_ENDPOINT, model="jina-reranker")
    try:
        test = reranker.rerank("test", ["test doc"], top_k=1)
        if test and test[0][1] > 0:
            print(f"  Reranker: OK (jina-reranker via HTTP)")
        else:
            print("  Reranker: zero scores, disabling")
            reranker = None
    except Exception as e:
        print(f"  Reranker: UNAVAILABLE ({e})")
        reranker = None

    # Load indexes
    print(f"\n  Loading indexes...")
    code_dbs = load_dbs(args.code_model)
    gen_dbs = load_dbs(args.gen_model)
    if not code_dbs or not gen_dbs:
        return
    print(f"  Code index: mixed_{args.code_model}")
    print(f"  Gen index: mixed_{args.gen_model}")

    # Run modes
    all_results = {}
    labels = []

    # 1. Fan-out (both models, no routing)
    print(f"\n  Running fan-out modes...")
    fanout_results, fanout_ms = run_fanout(
        code_client, code_dbs, gen_client, gen_dbs, reranker, mode="fanout",
    )
    all_results["FANOUT+rerank"] = fanout_results
    labels.append("FANOUT+rerank")
    print(f"    FANOUT+rerank: avg {fanout_ms:.0f}ms/query")

    # 2. Smart routing (code→coderank, doc→bge, cross→both)
    smart_results, smart_ms = run_fanout(
        code_client, code_dbs, gen_client, gen_dbs, reranker, mode="smart",
    )
    all_results["SMART+rerank"] = smart_results
    labels.append("SMART+rerank")
    print(f"    SMART+rerank: avg {smart_ms:.0f}ms/query")

    # 3. Baselines for comparison: single-model bests
    # CodeRankEmbed alone with reranker
    cr_results = []
    for query, expected_files, desc, category in QUERIES:
        hits = search_with_model(query, code_client, code_dbs, use_hyde=False)
        if reranker and hits:
            docs = [h.get("content", "")[:512] for h in hits[:20]]
            reranked = reranker.rerank(query, docs, top_k=10)
            if reranked and reranked[0][1] > 0:
                hits = [hits[idx] for idx, _ in reranked if idx < len(hits)]
        hit_files = [h.get("file_path", "") for h in hits]
        scores = evaluate_hits(hit_files, expected_files)
        scores["desc"] = desc
        scores["category"] = category
        scores["top_file"] = os.path.basename(hit_files[0]) if hit_files else "—"
        scores["top3_files"] = [os.path.basename(f) for f in hit_files[:3]]
        scores["latency_ms"] = 0
        cr_results.append(scores)
    all_results["CodeRank+rr"] = cr_results
    labels.append("CodeRank+rr")
    print(f"    CodeRank+rr: baseline")

    # BGE-small alone with reranker
    bge_results = []
    for query, expected_files, desc, category in QUERIES:
        hits = search_with_model(query, gen_client, gen_dbs, use_hyde=False)
        if reranker and hits:
            docs = [h.get("content", "")[:512] for h in hits[:20]]
            reranked = reranker.rerank(query, docs, top_k=10)
            if reranked and reranked[0][1] > 0:
                hits = [hits[idx] for idx, _ in reranked if idx < len(hits)]
        hit_files = [h.get("file_path", "") for h in hits]
        scores = evaluate_hits(hit_files, expected_files)
        scores["desc"] = desc
        scores["category"] = category
        scores["top_file"] = os.path.basename(hit_files[0]) if hit_files else "—"
        scores["top3_files"] = [os.path.basename(f) for f in hit_files[:3]]
        scores["latency_ms"] = 0
        bge_results.append(scores)
    all_results["BGE-small+rr"] = bge_results
    labels.append("BGE-small+rr")
    print(f"    BGE-small+rr: baseline")

    # Print results
    print_comparison(all_results, labels)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for label in labels:
        res = all_results[label]
        code_mrr = sum(r["mrr"] for r in res if r["category"] == "code") / 10
        doc_mrr = sum(r["mrr"] for r in res if r["category"] == "doc") / 10
        cross_mrr = sum(r["mrr"] for r in res if r["category"] == "cross") / 10
        blended = (code_mrr + doc_mrr + cross_mrr) / 3
        print(f"  {label:20s}  Code: {code_mrr:.3f}  Doc: {doc_mrr:.3f}  "
              f"Cross: {cross_mrr:.3f}  Blended: {blended:.3f}")

    # Close
    code_client.close()
    gen_client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
