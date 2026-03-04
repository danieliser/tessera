"""Batch benchmark: run all fastembed model tiers and collect aggregate results.

Usage:
    uv run python scripts/benchmark_all_models.py
    uv run python scripts/benchmark_all_models.py --rerankers   # also compare rerankers
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tessera.db import ProjectDB
from tessera.embeddings import FastembedClient, FastembedReranker
from tessera.indexer import IndexerPipeline
from tessera.search import SearchType, hybrid_search

PM_CORE = os.path.expanduser("~/Projects/ProContent/ProductCode/popup-maker")
PM_PRO = os.path.expanduser("~/Projects/ProContent/ProductCode/popup-maker-pro")

QUERIES = [
    ("popup rendering in WordPress footer hook", ["Popups.php"], "Frontend rendering"),
    ("register popup and popup_theme custom post types", ["PostTypes.php"], "Post type registration"),
    ("popup data model with triggers conditions and settings", ["Popup.php"], "Popup model"),
    ("singleton registry for popup trigger types", ["Triggers.php"], "Trigger registry"),
    ("exit intent mouse detection trigger", ["Triggers.php", "entry--plugin-init.php"], "Exit intent trigger"),
    ("condition callback evaluation for page targeting", ["ConditionCallbacks.php", "Conditions.php"], "Condition callbacks"),
    ("check if popup is loadable on current page", ["conditionals.php"], "Popup conditionals"),
    ("popup cookie expiration session hours days", ["Cookies.php", "PopupCookie.php"], "Cookie system"),
    ("newsletter subscription form AJAX submission handler", ["Newsletters.php", "Subscribe.php"], "Newsletter AJAX"),
    ("Gravity Forms integration for popup form submission", ["GravityForms.php"], "Gravity Forms integration"),
    ("shortcode to trigger popup on click element", ["PopupTrigger.php"], "Trigger shortcode"),
    ("pum_subscribe email subscription shortcode", ["Subscribe.php"], "Subscribe shortcode"),
    ("REST API endpoint for license validation", ["RestAPI.php", "License.php"], "License REST API"),
    ("AJAX analytics tracking popup open and conversion events", ["Analytics.php"], "Analytics tracking"),
    ("admin settings page for global plugin options", ["Settings.php"], "Admin settings"),
    ("Gutenberg block editor integration for popups", ["BlockEditor.php"], "Block editor"),
    ("Pimple dependency injection container service registration", ["Container.php", "Core.php"], "DI container"),
    ("FluentCRM add tag to contact on popup conversion", ["AddTag.php", "FluentCRM.php"], "FluentCRM tagging"),
    ("popup schedule date range recurring daily weekly", ["scheduling.php", "scheduled-actions.php"], "Scheduling"),
    ("attribution model calculation for conversion tracking", ["Attribution.php", "Conversions.php"], "Attribution service"),
]

# Models to benchmark — (key, fastembed_name, label, size_mb, dim)
EMBEDDING_MODELS = [
    ("arctic-xs", "Snowflake/snowflake-arctic-embed-xs", "Arctic-XS", 90, 384),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6", 90, 384),
    ("bge-small", "BAAI/bge-small-en-v1.5", "BGE-small", 67, 384),
    ("jina-small", "jinaai/jina-embeddings-v2-small-en", "Jina-small", 120, 512),
    ("nomic-q", "nomic-ai/nomic-embed-text-v1.5-Q", "Nomic-Q", 130, 768),
    ("arctic-s", "snowflake/snowflake-arctic-embed-s", "Arctic-S", 130, 384),
    ("bge-base", "BAAI/bge-base-en-v1.5", "BGE-base", 210, 768),
    ("gte-base", "thenlper/gte-base", "GTE-base", 440, 768),
    ("arctic-m", "snowflake/snowflake-arctic-embed-m", "Arctic-M", 430, 768),
    ("nomic", "nomic-ai/nomic-embed-text-v1.5", "Nomic-full", 520, 768),
    ("jina-code", "jinaai/jina-embeddings-v2-base-code", "Jina-Code", 640, 768),
    ("mxbai-large", "mixedbread-ai/mxbai-embed-large-v1", "MxBAI-large", 640, 1024),
]

RERANKER_MODELS = [
    ("ms-marco", "Xenova/ms-marco-MiniLM-L-6-v2", "MiniLM-L6-rr", 80),
    ("ms-marco-12", "Xenova/ms-marco-MiniLM-L-12-v2", "MiniLM-L12-rr", 120),
    ("jina-tiny", "jinaai/jina-reranker-v1-tiny-en", "Jina-tiny-rr", 130),
    ("jina-turbo", "jinaai/jina-reranker-v1-turbo-en", "Jina-turbo-rr", 150),
]


def evaluate_hits(hit_files, expected_files):
    top1 = any(exp in hit_files[0] for exp in expected_files) if hit_files else False
    top3 = any(any(exp in f for exp in expected_files) for f in hit_files[:3])
    top5 = any(any(exp in f for exp in expected_files) for f in hit_files[:5])
    top10 = any(any(exp in f for exp in expected_files) for f in hit_files[:10])
    mrr = 0.0
    for rank, f in enumerate(hit_files, 1):
        if any(exp in f for exp in expected_files):
            mrr = 1.0 / rank
            break
    return {"top1": top1, "top3": top3, "top5": top5, "top10": top10, "mrr": mrr}


def run_queries(client, db_core, db_pro, search_types=None, reranker=None):
    results = []
    for query, expected_files, desc in QUERIES:
        needs_embedding = search_types is None or SearchType.VEC in search_types
        query_embedding = None
        if needs_embedding:
            raw = client.embed_query(query)
            query_embedding = np.array(raw, dtype=np.float32)

        hits_core = hybrid_search(
            query, query_embedding, db_core, graph=None, limit=10,
            source_type=["code"], search_types=search_types,
            advanced_fts=False, rrf_weights=None,
        )
        hits_pro = hybrid_search(
            query, query_embedding, db_pro, graph=None, limit=10,
            source_type=["code"], search_types=search_types,
            advanced_fts=False, rrf_weights=None,
        )

        all_hits = hits_core + hits_pro
        all_hits.sort(key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)

        # Rerank if available
        if reranker and all_hits:
            docs = [h.get("content", "")[:512] for h in all_hits[:20]]
            reranked = reranker.rerank(query, docs, top_k=10)
            if reranked and reranked[0][1] > 0:
                reordered = []
                for orig_idx, score in reranked:
                    if orig_idx < len(all_hits):
                        reordered.append(all_hits[orig_idx])
                all_hits = reordered

        hit_files = [os.path.basename(h.get("file_path", "")) for h in all_hits]
        scores = evaluate_hits(hit_files, expected_files)
        results.append(scores)
    return results


def aggregate(results):
    n = len(results)
    return {
        "top1": sum(1 for r in results if r["top1"]) / n * 100,
        "top3": sum(1 for r in results if r["top3"]) / n * 100,
        "top5": sum(1 for r in results if r["top5"]) / n * 100,
        "top10": sum(1 for r in results if r["top10"]) / n * 100,
        "mrr": sum(r["mrr"] for r in results) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerankers", action="store_true", help="Also compare reranker models")
    parser.add_argument("--models", nargs="*", help="Specific model keys to test (default: all)")
    args = parser.parse_args()

    for path, label in [(PM_CORE, "Core"), (PM_PRO, "Pro")]:
        if not os.path.isdir(path):
            print(f"ERROR: {label} not found at {path}")
            return

    models_to_test = EMBEDDING_MODELS
    if args.models:
        models_to_test = [m for m in EMBEDDING_MODELS if m[0] in args.models]

    # Default reranker for comparison
    print("Loading default reranker (jina-reranker-v1-tiny-en)...")
    default_reranker = FastembedReranker(model_name="jinaai/jina-reranker-v1-tiny-en")

    all_rows = []

    for model_key, model_name, label, size_mb, dim in models_to_test:
        print(f"\n{'='*80}")
        print(f"  {label} ({model_name}, ~{size_mb}MB, {dim}d)")
        print(f"{'='*80}")

        try:
            client = FastembedClient(model_name=model_name)
            test_vec = client.embed_single("test")
            actual_dim = len(test_vec)
            print(f"  Loaded: {actual_dim}d vectors")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        base_dir = tempfile.mkdtemp(prefix=f"tessera_bench_{model_key}_")

        # Index both repos
        core_base = os.path.join(base_dir, "core")
        os.makedirs(core_base, exist_ok=True)
        ProjectDB.base_dir = core_base
        pipeline_core = IndexerPipeline(project_path=PM_CORE, embedding_client=client)
        pipeline_core.register()
        t0 = time.perf_counter()
        stats_core = pipeline_core.index_project()

        pro_base = os.path.join(base_dir, "pro")
        os.makedirs(pro_base, exist_ok=True)
        ProjectDB.base_dir = pro_base
        pipeline_pro = IndexerPipeline(project_path=PM_PRO, embedding_client=client)
        pipeline_pro.register()
        stats_pro = pipeline_pro.index_project()
        index_time = time.perf_counter() - t0
        total_chunks = stats_core.chunks_embedded + stats_pro.chunks_embedded
        print(f"  Indexed: {total_chunks} chunks in {index_time:.0f}s")

        # Open DBs
        ProjectDB.base_dir = core_base
        db_core = ProjectDB(PM_CORE)
        ProjectDB.base_dir = pro_base
        db_pro = ProjectDB(PM_PRO)

        # VEC+code (no reranker)
        vec_results = run_queries(client, db_core, db_pro, [SearchType.VEC])
        vec_agg = aggregate(vec_results)

        # VEC+code+rerank
        rr_results = run_queries(client, db_core, db_pro, [SearchType.VEC], default_reranker)
        rr_agg = aggregate(rr_results)

        row = {
            "key": model_key, "label": label, "size_mb": size_mb, "dim": actual_dim,
            "index_s": index_time,
            "vec_mrr": vec_agg["mrr"], "vec_top1": vec_agg["top1"],
            "vec_top3": vec_agg["top3"], "vec_top10": vec_agg["top10"],
            "rr_mrr": rr_agg["mrr"], "rr_top1": rr_agg["top1"],
            "rr_top3": rr_agg["top3"], "rr_top10": rr_agg["top10"],
        }
        all_rows.append(row)

        print(f"  VEC+code:   MRR={vec_agg['mrr']:.3f}  Top1={vec_agg['top1']:.0f}%  Top3={vec_agg['top3']:.0f}%  Top10={vec_agg['top10']:.0f}%")
        print(f"  +rerank:    MRR={rr_agg['mrr']:.3f}  Top1={rr_agg['top1']:.0f}%  Top3={rr_agg['top3']:.0f}%  Top10={rr_agg['top10']:.0f}%")

        db_core.close()
        db_pro.close()
        client.close()
        ProjectDB.base_dir = None
        shutil.rmtree(base_dir, ignore_errors=True)

    # Reranker comparison (using best embedding model)
    reranker_rows = []
    if args.rerankers and all_rows:
        best = max(all_rows, key=lambda r: r["vec_mrr"])
        print(f"\n\n{'#'*80}")
        print(f"# Reranker comparison (using {best['label']} embeddings)")
        print(f"{'#'*80}")

        best_model = next(m for m in EMBEDDING_MODELS if m[0] == best["key"])
        client = FastembedClient(model_name=best_model[1])

        base_dir = tempfile.mkdtemp(prefix="tessera_bench_reranker_")
        core_base = os.path.join(base_dir, "core")
        os.makedirs(core_base, exist_ok=True)
        ProjectDB.base_dir = core_base
        pipeline = IndexerPipeline(project_path=PM_CORE, embedding_client=client)
        pipeline.register()
        pipeline.index_project()

        pro_base = os.path.join(base_dir, "pro")
        os.makedirs(pro_base, exist_ok=True)
        ProjectDB.base_dir = pro_base
        pipeline = IndexerPipeline(project_path=PM_PRO, embedding_client=client)
        pipeline.register()
        pipeline.index_project()

        ProjectDB.base_dir = core_base
        db_core = ProjectDB(PM_CORE)
        ProjectDB.base_dir = pro_base
        db_pro = ProjectDB(PM_PRO)

        for rr_key, rr_name, rr_label, rr_size in RERANKER_MODELS:
            print(f"\n  {rr_label} ({rr_name}, ~{rr_size}MB)")
            try:
                reranker = FastembedReranker(model_name=rr_name)
                results = run_queries(client, db_core, db_pro, [SearchType.VEC], reranker)
                agg = aggregate(results)
                reranker_rows.append({
                    "key": rr_key, "label": rr_label, "size_mb": rr_size,
                    "mrr": agg["mrr"], "top1": agg["top1"],
                    "top3": agg["top3"], "top10": agg["top10"],
                })
                print(f"    MRR={agg['mrr']:.3f}  Top1={agg['top1']:.0f}%  Top3={agg['top3']:.0f}%  Top10={agg['top10']:.0f}%")
            except Exception as e:
                print(f"    FAILED: {e}")

        db_core.close()
        db_pro.close()
        client.close()
        ProjectDB.base_dir = None
        shutil.rmtree(base_dir, ignore_errors=True)

    # Summary table
    print(f"\n\n{'='*120}")
    print("EMBEDDING MODEL COMPARISON (sorted by VEC+rerank MRR)")
    print(f"{'='*120}")
    print(f"{'Model':<16} {'Size':>6} {'Dim':>5} {'Index':>7} | {'VEC MRR':>8} {'Top1':>5} {'Top3':>5} {'T10':>5} | {'+ Rerank':>8} {'Top1':>5} {'Top3':>5} {'T10':>5}")
    print("-" * 120)

    for row in sorted(all_rows, key=lambda r: r["rr_mrr"], reverse=True):
        print(f"{row['label']:<16} {row['size_mb']:>5}M {row['dim']:>5}d {row['index_s']:>5.0f}s "
              f"| {row['vec_mrr']:>8.3f} {row['vec_top1']:>4.0f}% {row['vec_top3']:>4.0f}% {row['vec_top10']:>4.0f}% "
              f"| {row['rr_mrr']:>8.3f} {row['rr_top1']:>4.0f}% {row['rr_top3']:>4.0f}% {row['rr_top10']:>4.0f}%")

    if reranker_rows:
        print(f"\n{'='*80}")
        print(f"RERANKER COMPARISON (with {best['label']} embeddings)")
        print(f"{'='*80}")
        print(f"{'Reranker':<20} {'Size':>6} | {'MRR':>8} {'Top1':>5} {'Top3':>5} {'Top10':>5}")
        print("-" * 80)
        for row in sorted(reranker_rows, key=lambda r: r["mrr"], reverse=True):
            print(f"{row['label']:<20} {row['size_mb']:>5}M | {row['mrr']:>8.3f} {row['top1']:>4.0f}% {row['top3']:>4.0f}% {row['top10']:>4.0f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
