"""Cross-test: top embedding models × top reranker models.

Runs a matrix of the best embedders against all rerankers to find the
optimal (embedder, reranker) pair for local deployment.

Usage:
    uv run python scripts/benchmark_cross.py
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

# Top embedding models from phase 1 benchmarks
EMBEDDERS = [
    ("bge-small", "BAAI/bge-small-en-v1.5", "BGE-small", 67, 384),
    ("nomic-q", "nomic-ai/nomic-embed-text-v1.5-Q", "Nomic-Q", 130, 768),
    ("bge-base", "BAAI/bge-base-en-v1.5", "BGE-base", 210, 768),
    ("gte-base", "thenlper/gte-base", "GTE-base", 440, 768),
]

# All reranker models
RERANKERS = [
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


def run_queries(client, db_core, db_pro, reranker=None):
    results = []
    for query, expected_files, _desc in QUERIES:
        raw = client.embed_query(query)
        query_embedding = np.array(raw, dtype=np.float32)

        hits_core = hybrid_search(
            query, query_embedding, db_core, graph=None, limit=10,
            source_type=["code"], search_types=[SearchType.VEC],
            advanced_fts=False, rrf_weights=None,
        )
        hits_pro = hybrid_search(
            query, query_embedding, db_pro, graph=None, limit=10,
            source_type=["code"], search_types=[SearchType.VEC],
            advanced_fts=False, rrf_weights=None,
        )

        all_hits = hits_core + hits_pro
        all_hits.sort(key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)

        if reranker and all_hits:
            docs = [h.get("content", "")[:512] for h in all_hits[:20]]
            reranked = reranker.rerank(query, docs, top_k=10)
            if reranked and reranked[0][1] > 0:
                reordered = []
                for orig_idx, _score in reranked:
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
    for path, label in [(PM_CORE, "Core"), (PM_PRO, "Pro")]:
        if not os.path.isdir(path):
            print(f"ERROR: {label} not found at {path}")
            return

    # Pre-load all rerankers
    print("Loading rerankers...")
    reranker_instances = {}
    for rr_key, rr_name, rr_label, _rr_size in RERANKERS:
        try:
            reranker_instances[rr_key] = FastembedReranker(model_name=rr_name)
            print(f"  {rr_label}: loaded")
        except Exception as e:
            print(f"  {rr_label}: FAILED ({e})")

    # Matrix: each embedder × each reranker
    matrix = []  # list of dicts

    for emb_key, emb_name, emb_label, emb_size, emb_dim in EMBEDDERS:
        print(f"\n{'='*80}")
        print(f"  {emb_label} ({emb_name}, ~{emb_size}MB, {emb_dim}d)")
        print(f"{'='*80}")

        try:
            client = FastembedClient(model_name=emb_name)
            test_vec = client.embed_single("test")
            actual_dim = len(test_vec)
            print(f"  Loaded: {actual_dim}d vectors")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        base_dir = tempfile.mkdtemp(prefix=f"tessera_cross_{emb_key}_")
        core_base = os.path.join(base_dir, "core")
        os.makedirs(core_base, exist_ok=True)
        ProjectDB.base_dir = core_base
        pipeline_core = IndexerPipeline(project_path=PM_CORE, embedding_client=client)
        pipeline_core.register()
        t0 = time.perf_counter()
        stats_core = pipeline_core.index_project_sync()

        pro_base = os.path.join(base_dir, "pro")
        os.makedirs(pro_base, exist_ok=True)
        ProjectDB.base_dir = pro_base
        pipeline_pro = IndexerPipeline(project_path=PM_PRO, embedding_client=client)
        pipeline_pro.register()
        stats_pro = pipeline_pro.index_project_sync()
        index_time = time.perf_counter() - t0
        total_chunks = stats_core.chunks_embedded + stats_pro.chunks_embedded
        print(f"  Indexed: {total_chunks} chunks in {index_time:.0f}s")

        ProjectDB.base_dir = core_base
        db_core = ProjectDB(PM_CORE)
        ProjectDB.base_dir = pro_base
        db_pro = ProjectDB(PM_PRO)

        # VEC-only (no reranker)
        vec_results = run_queries(client, db_core, db_pro)
        vec_agg = aggregate(vec_results)
        print(f"  VEC-only:  MRR={vec_agg['mrr']:.3f}  Top1={vec_agg['top1']:.0f}%  Top3={vec_agg['top3']:.0f}%  Top10={vec_agg['top10']:.0f}%")

        matrix.append({
            "emb": emb_label, "emb_size": emb_size, "dim": actual_dim,
            "rr": "None", "rr_size": 0, "total_mb": emb_size,
            "mrr": vec_agg["mrr"], "top1": vec_agg["top1"],
            "top3": vec_agg["top3"], "top10": vec_agg["top10"],
            "index_s": index_time,
        })

        # Each reranker
        for rr_key, _rr_name, rr_label, rr_size in RERANKERS:
            if rr_key not in reranker_instances:
                continue
            reranker = reranker_instances[rr_key]
            rr_results = run_queries(client, db_core, db_pro, reranker)
            rr_agg = aggregate(rr_results)
            total = emb_size + rr_size
            print(f"  +{rr_label:<16} MRR={rr_agg['mrr']:.3f}  Top1={rr_agg['top1']:.0f}%  Top3={rr_agg['top3']:.0f}%  Top10={rr_agg['top10']:.0f}%  ({total}MB total)")

            matrix.append({
                "emb": emb_label, "emb_size": emb_size, "dim": actual_dim,
                "rr": rr_label, "rr_size": rr_size, "total_mb": total,
                "mrr": rr_agg["mrr"], "top1": rr_agg["top1"],
                "top3": rr_agg["top3"], "top10": rr_agg["top10"],
                "index_s": index_time,
            })

        db_core.close()
        db_pro.close()
        client.close()
        ProjectDB.base_dir = None
        shutil.rmtree(base_dir, ignore_errors=True)

    # Summary: sorted by MRR
    print(f"\n\n{'='*120}")
    print("CROSS-TEST MATRIX: Embedder × Reranker (sorted by MRR)")
    print(f"{'='*120}")
    print(f"{'Embedder':<12} {'Dim':>5} {'Reranker':<18} {'Total':>7} {'Index':>7} | {'MRR':>8} {'Top1':>5} {'Top3':>5} {'T10':>5}")
    print("-" * 120)

    for row in sorted(matrix, key=lambda r: r["mrr"], reverse=True):
        print(f"{row['emb']:<12} {row['dim']:>4}d {row['rr']:<18} {row['total_mb']:>5}MB {row['index_s']:>5.0f}s "
              f"| {row['mrr']:>8.3f} {row['top1']:>4.0f}% {row['top3']:>4.0f}% {row['top10']:>4.0f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
