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

# Boilerplate files that dominate reranker input without being useful results
BOILERPLATE_FILES = {
    "readme.txt", "readme.md", "readme.rst",
    "changelog.md", "changelog.txt", "changes.md",
    "license.md", "license.txt", "license",
    "contributing.md", "code_of_conduct.md",
}


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
               mode="fanout", graphs_code=None, graphs_gen=None,
               filter_boilerplate=False, rerank_pool=20, gen_weight=1):
    """Fan-out: search both models, merge candidates, rerank.

    Args:
        filter_boilerplate: Deprioritize readme/changelog/license before reranking.
        rerank_pool: Number of candidates to feed the reranker (default 20).
        gen_weight: How many gen hits per code hit during interleaving (default 1).
                    Use 2 for cross queries to prioritize general model results.
    """
    results = []
    total_ms = 0.0

    for query, expected_files, desc, category in QUERIES:
        t0 = time.perf_counter()

        # Per-query gen weight (v3 uses 2:1 for cross)
        q_gen_weight = gen_weight

        if mode in ("smart", "smart_v2", "smart_v3"):
            # Smart routing: both models for code, general-only for docs
            if category == "code":
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
                q_gen_weight = 1  # code: equal weighting
            elif category == "doc":
                code_hits = []
                gen_hits = search_with_model(
                    query, gen_client, gen_dbs,
                    search_types=[SearchType.VEC], src_filter=["markdown", "json"],
                    use_hyde=True, graphs=graphs_gen,
                )
            elif mode == "smart_v3":
                # Cross v3: fan-out both + hybrid + HyDE on gen + graph +
                # boilerplate filter + 2:1 gen weighting + larger pool
                code_hits = search_with_model(
                    query, code_client, code_dbs,
                    search_types=[SearchType.VEC, SearchType.LEX],
                    use_hyde=False, graphs=graphs_code,
                )
                gen_hits = search_with_model(
                    query, gen_client, gen_dbs,
                    search_types=[SearchType.VEC, SearchType.LEX],
                    use_hyde=True, graphs=graphs_gen,
                )
                q_gen_weight = 2  # cross: favor gen model 2:1
            elif mode == "smart_v2":
                # Cross v2: fan-out both + hybrid + HyDE on gen + graph
                code_hits = search_with_model(
                    query, code_client, code_dbs,
                    search_types=[SearchType.VEC, SearchType.LEX],
                    use_hyde=False, graphs=graphs_code,
                )
                gen_hits = search_with_model(
                    query, gen_client, gen_dbs,
                    search_types=[SearchType.VEC, SearchType.LEX],
                    use_hyde=True, graphs=graphs_gen,
                )
            else:  # cross v1 — code model only with hybrid + reranker
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

        # Interleave: weighted code/gen hits, dedup by file_path+content
        seen = set()
        merged = []
        code_iter = iter(code_hits)
        gen_iter = iter(gen_hits)
        code_done = gen_done = False
        merge_limit = max(30, rerank_pool + 10)

        while len(merged) < merge_limit:
            # One code hit per round
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
            # q_gen_weight gen hits per round (2:1 for cross v3)
            for _ in range(q_gen_weight):
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

        # Filter boilerplate before reranking (push to end, don't remove)
        if filter_boilerplate and merged:
            priority = []
            deprioritized = []
            for h in merged:
                fname = os.path.basename(h.get("file_path", "")).lower()
                if fname in BOILERPLATE_FILES:
                    deprioritized.append(h)
                else:
                    priority.append(h)
            merged = priority + deprioritized

        # Rerank the merged candidates
        if reranker and merged:
            docs = [h.get("content", "")[:512] for h in merged[:rerank_pool]]
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
    parser.add_argument("--reranker", default="jina-reranker",
                        help="Reranker model (HTTP gateway model name or fastembed model path)")
    parser.add_argument("--reranker-local", action="store_true",
                        help="Use FastembedReranker (local ONNX) instead of HTTP gateway")
    parser.add_argument("--rerank-pool", type=int, default=20,
                        help="Number of candidates to feed reranker (default 20)")
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
    if args.reranker_local:
        from tessera.embeddings import FastembedReranker
        reranker = FastembedReranker(model_name=args.reranker)
        try:
            test = reranker.rerank("test", ["test doc"], top_k=1)
            print(f"  Reranker: OK ({args.reranker} via fastembed/ONNX)")
        except Exception as e:
            print(f"  Reranker: UNAVAILABLE ({e})")
            reranker = None
    else:
        reranker = HTTPReranker(endpoint=RERANK_ENDPOINT, model=args.reranker)
        try:
            test = reranker.rerank("test", ["test doc"], top_k=1)
            if test and test[0][1] > 0:
                print(f"  Reranker: OK ({args.reranker} via HTTP)")
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

    # 2. Smart routing v1 (code→both, doc→bge, cross→coderank only)
    smart_results, smart_ms = run_fanout(
        code_client, code_dbs, gen_client, gen_dbs, reranker, mode="smart",
    )
    all_results["SMART+rerank"] = smart_results
    labels.append("SMART+rerank")
    print(f"    SMART+rerank: avg {smart_ms:.0f}ms/query")

    # 3. Smart v2 (cross: fan-out both + hybrid + HyDE on gen + graph)
    # Load graphs for v2/v3
    graphs_code = {}
    graphs_gen = {}
    for label, db in code_dbs:
        graphs_code[label] = try_load_graph(db)
    for label, db in gen_dbs:
        graphs_gen[label] = try_load_graph(db)
    smart2_results, smart2_ms = run_fanout(
        code_client, code_dbs, gen_client, gen_dbs, reranker, mode="smart_v2",
        graphs_code=graphs_code, graphs_gen=graphs_gen,
        rerank_pool=args.rerank_pool,
    )
    all_results["SMARTv2+rerank"] = smart2_results
    labels.append("SMARTv2+rerank")
    print(f"    SMARTv2+rerank: avg {smart2_ms:.0f}ms/query")

    # 4. Smart v3 (v2 + boilerplate filter + 2:1 gen weight + larger pool)
    smart3_results, smart3_ms = run_fanout(
        code_client, code_dbs, gen_client, gen_dbs, reranker, mode="smart_v3",
        graphs_code=graphs_code, graphs_gen=graphs_gen,
        filter_boilerplate=True, rerank_pool=args.rerank_pool,
    )
    all_results["SMARTv3+rerank"] = smart3_results
    labels.append("SMARTv3+rerank")
    print(f"    SMARTv3+rerank: avg {smart3_ms:.0f}ms/query")

    # 5. Baselines for comparison: single-model bests
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
