"""Real-world benchmark: Popup Maker (core + Pro) — PHP codebase, ~580 files.

Indexes both popup-maker and popup-maker-pro, then runs 20 search queries with
known ground-truth file expectations. Tests multiple search modes with optional
PPR graph ranking, reranking, and multiple embedding models.

Usage:
    uv run python scripts/benchmark_pm.py                         # default: Nomic via gateway
    uv run python scripts/benchmark_pm.py --model qwen3           # Qwen3 via gateway
    uv run python scripts/benchmark_pm.py --all                   # all modes + PPR + rerank
    uv run python scripts/benchmark_pm.py --provider fastembed    # local bge-small + jina-tiny
    uv run python scripts/benchmark_pm.py --provider fastembed --model arctic  # ultra-compact
    uv run python scripts/benchmark_pm.py --provider fastembed --model bge-small --all
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tessera.db import ProjectDB
from tessera.embeddings import (
    EmbeddingClient,
    FastembedClient,
    FastembedReranker,
    HTTPReranker,
)
from tessera.graph import load_project_graph
from tessera.indexer import IndexerPipeline
from tessera.search import SearchType, hybrid_search

logger = logging.getLogger(__name__)

PM_CORE = os.path.expanduser(
    "~/Projects/ProContent/ProductCode/popup-maker"
)
PM_PRO = os.path.expanduser(
    "~/Projects/ProContent/ProductCode/popup-maker-pro"
)

# Gateway HTTP models
HTTP_MODELS = {
    "nomic": ("nomic-embed", "Nomic-768d"),
    "qwen3": ("qwen3-embed", "Qwen3-1024d"),
}

# Local fastembed models — ordered by size tier
FASTEMBED_MODELS = {
    # ~23-90MB tier (ultra-compact)
    "arctic-xs": ("Snowflake/snowflake-arctic-embed-xs", "Arctic-XS-384d", 90),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6-384d", 90),
    # ~67-130MB tier (small)
    "bge-small": ("BAAI/bge-small-en-v1.5", "BGE-small-384d", 67),
    "nomic-q": ("nomic-ai/nomic-embed-text-v1.5-Q", "Nomic-Q-768d", 130),
    "arctic-s": ("snowflake/snowflake-arctic-embed-s", "Arctic-S-384d", 130),
    "jina-small": ("jinaai/jina-embeddings-v2-small-en", "Jina-small-512d", 120),
    # ~210-440MB tier (base)
    "bge-base": ("BAAI/bge-base-en-v1.5", "BGE-base-768d", 210),
    "arctic-m": ("snowflake/snowflake-arctic-embed-m", "Arctic-M-768d", 430),
    "gte-base": ("thenlper/gte-base", "GTE-base-768d", 440),
    # ~520-640MB tier (large / code-specialized)
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", "Nomic-768d", 520),
    "jina-code": ("jinaai/jina-embeddings-v2-base-code", "Jina-Code-768d", 640),
    "mxbai-large": ("mixedbread-ai/mxbai-embed-large-v1", "MxBAI-large-1024d", 640),
}

FASTEMBED_RERANKERS = {
    "ms-marco": ("Xenova/ms-marco-MiniLM-L-6-v2", "MiniLM-L6-reranker", 80),
    "ms-marco-12": ("Xenova/ms-marco-MiniLM-L-12-v2", "MiniLM-L12-reranker", 120),
    "jina-tiny": ("jinaai/jina-reranker-v1-tiny-en", "Jina-tiny-reranker", 130),
    "jina-turbo": ("jinaai/jina-reranker-v1-turbo-en", "Jina-turbo-reranker", 150),
}

QUERIES = [
    # --- Core popup lifecycle ---
    ("popup rendering in WordPress footer hook", ["Popups.php"], "Frontend rendering"),
    ("register popup and popup_theme custom post types", ["PostTypes.php"], "Post type registration"),
    ("popup data model with triggers conditions and settings", ["Popup.php"], "Popup model"),
    # --- Trigger system ---
    ("singleton registry for popup trigger types", ["Triggers.php"], "Trigger registry"),
    ("exit intent mouse detection trigger", ["Triggers.php", "entry--plugin-init.php"], "Exit intent trigger"),
    # --- Conditions / targeting ---
    ("condition callback evaluation for page targeting", ["ConditionCallbacks.php", "Conditions.php"], "Condition callbacks"),
    ("check if popup is loadable on current page", ["conditionals.php"], "Popup conditionals"),
    # --- Cookies / display frequency ---
    ("popup cookie expiration session hours days", ["Cookies.php", "PopupCookie.php"], "Cookie system"),
    # --- Newsletter / forms ---
    ("newsletter subscription form AJAX submission handler", ["Newsletters.php", "Subscribe.php"], "Newsletter AJAX"),
    ("Gravity Forms integration for popup form submission", ["GravityForms.php"], "Gravity Forms integration"),
    # --- Shortcodes ---
    ("shortcode to trigger popup on click element", ["PopupTrigger.php"], "Trigger shortcode"),
    ("pum_subscribe email subscription shortcode", ["Subscribe.php"], "Subscribe shortcode"),
    # --- REST API / AJAX ---
    ("REST API endpoint for license validation", ["RestAPI.php", "License.php"], "License REST API"),
    ("AJAX analytics tracking popup open and conversion events", ["Analytics.php"], "Analytics tracking"),
    # --- Admin ---
    ("admin settings page for global plugin options", ["Settings.php"], "Admin settings"),
    ("Gutenberg block editor integration for popups", ["BlockEditor.php"], "Block editor"),
    # --- DI / architecture ---
    ("Pimple dependency injection container service registration", ["Container.php", "Core.php"], "DI container"),
    # --- Pro features ---
    ("FluentCRM add tag to contact on popup conversion", ["AddTag.php", "FluentCRM.php"], "FluentCRM tagging"),
    ("popup schedule date range recurring daily weekly", ["scheduling.php", "scheduled-actions.php"], "Scheduling"),
    ("attribution model calculation for conversion tracking", ["Attribution.php", "Conversions.php"], "Attribution service"),
]


def evaluate_hits(hit_files: list[str], expected_files: list[str]) -> dict:
    """Score a single query result against ground truth."""
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


def search_both_dbs(
    query: str,
    query_embedding: np.ndarray | None,
    db_core: ProjectDB,
    db_pro: ProjectDB,
    graph_core=None,
    graph_pro=None,
    search_types=None,
    src_filter=None,
    reranker=None,
) -> list[dict]:
    """Search both core and pro DBs, merge by score, optionally rerank."""
    hits_core = hybrid_search(
        query, query_embedding, db_core,
        graph=graph_core, limit=10,
        source_type=src_filter, search_types=search_types,
        advanced_fts=False, rrf_weights=None,
    )
    hits_pro = hybrid_search(
        query, query_embedding, db_pro,
        graph=graph_pro, limit=10,
        source_type=src_filter, search_types=search_types,
        advanced_fts=False, rrf_weights=None,
    )

    all_hits = []
    for h in hits_core:
        h["_source"] = "core"
        all_hits.append(h)
    for h in hits_pro:
        h["_source"] = "pro"
        all_hits.append(h)
    all_hits.sort(
        key=lambda x: x.get("rrf_score", x.get("score", 0)),
        reverse=True,
    )

    # Post-RRF reranking
    if reranker and all_hits:
        docs = [h.get("content", "")[:512] for h in all_hits[:20]]
        reranked = reranker.rerank(query, docs, top_k=10)
        if reranked and reranked[0][1] > 0:  # non-fallback scores
            reordered = []
            for orig_idx, score in reranked:
                if orig_idx < len(all_hits):
                    item = all_hits[orig_idx]
                    item["rerank_score"] = score
                    reordered.append(item)
            all_hits = reordered

    return all_hits


def run_mode(
    mode_label: str,
    search_types,
    src_filter,
    client,
    db_core: ProjectDB,
    db_pro: ProjectDB,
    graph_core=None,
    graph_pro=None,
    reranker=None,
) -> list[dict]:
    """Run all queries for a single mode, return results with timing."""
    results = []
    total_ms = 0.0

    for query, expected_files, desc in QUERIES:
        needs_embedding = (
            (search_types and SearchType.VEC in search_types)
            or search_types is None
        )
        query_embedding = None
        if needs_embedding:
            raw = client.embed_query(query)
            query_embedding = np.array(raw, dtype=np.float32)

        t0 = time.perf_counter()
        all_hits = search_both_dbs(
            query, query_embedding, db_core, db_pro,
            graph_core, graph_pro, search_types, src_filter, reranker,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        hit_files = [os.path.basename(h.get("file_path", "")) for h in all_hits]
        scores = evaluate_hits(hit_files, expected_files)
        scores["desc"] = desc
        scores["top_file"] = hit_files[0] if hit_files else "—"
        scores["top3_files"] = hit_files[:3]
        scores["latency_ms"] = elapsed_ms
        results.append(scores)

    avg_ms = total_ms / len(QUERIES)
    print(f"    {mode_label}: avg {avg_ms:.0f}ms/query")
    return results


def print_comparison(all_results: dict, labels: list[str]):
    n = len(QUERIES)

    print(f"\n{'=' * 100}")
    print("AGGREGATE METRICS")
    print(f"{'=' * 100}")

    header = f"{'Metric':<20}" + "".join(f"{l:>16}" for l in labels)
    print(header)
    print("-" * len(header))

    for metric_name, key in [
        ("Top-1 Accuracy", "top1"),
        ("Top-3 Accuracy", "top3"),
        ("Top-5 Accuracy", "top5"),
        ("Top-10 Accuracy", "top10"),
    ]:
        row = f"{metric_name:<20}"
        for label in labels:
            r = all_results[label]
            acc = sum(1 for x in r if x[key]) / n * 100
            row += f"{acc:>15.0f}%"
        print(row)

    row = f"{'MRR':<20}"
    for label in labels:
        r = all_results[label]
        mrr = sum(x["mrr"] for x in r) / n
        row += f"{mrr:>16.3f}"
    print(row)

    row = f"{'Avg Latency (ms)':<20}"
    for label in labels:
        r = all_results[label]
        avg = sum(x["latency_ms"] for x in r) / n
        row += f"{avg:>16.0f}"
    print(row)

    print(f"\n{'=' * 100}")
    print("PER-QUERY BREAKDOWN")
    print(f"{'=' * 100}")

    for i, (_, expected, desc) in enumerate(QUERIES):
        print(f"\n  Q{i+1}: {desc} — expected: {expected}")
        for label in labels:
            r = all_results[label][i]
            rank_str = "MISS" if r["mrr"] == 0 else f"rank {int(1/r['mrr'])}"
            marker = "+" if r["top3"] else ("-" if r["top10"] else "X")
            top3 = ", ".join(r["top3_files"])
            print(f"    [{marker}] {label:<16} {rank_str:<8} [{top3}]")


def try_load_graph(db: ProjectDB):
    """Attempt to load PPR graph for project_id=1 (standalone default)."""
    try:
        graph = load_project_graph(db, 1)
        print(f"    Graph loaded: {graph.n_symbols} symbols, {graph.edge_count} edges"
              f"{' (SPARSE)' if graph.is_sparse_fallback() else ''}")
        return graph
    except (ValueError, Exception) as e:
        print(f"    Graph load failed: {e}")
        return None


def create_client_and_reranker(args):
    """Create embedding client and reranker based on provider."""
    if args.provider == "fastembed":
        # Local fastembed models
        model_key = args.model or "bge-small"
        if model_key not in FASTEMBED_MODELS:
            print(f"ERROR: Unknown fastembed model '{model_key}'. "
                  f"Options: {', '.join(FASTEMBED_MODELS)}")
            return None, None, None, None

        model_name, model_label, embed_mb = FASTEMBED_MODELS[model_key]
        print(f"  Loading local embedding: {model_label} ({model_name}, ~{embed_mb}MB)")
        client = FastembedClient(model_name=model_name)

        reranker = None
        reranker_key = args.reranker or "jina-tiny"
        if (args.rerank or args.all) and reranker_key in FASTEMBED_RERANKERS:
            rr_name, rr_label, rr_mb = FASTEMBED_RERANKERS[reranker_key]
            print(f"  Loading local reranker: {rr_label} ({rr_name}, ~{rr_mb}MB)")
            reranker = FastembedReranker(model_name=rr_name)
            total_mb = embed_mb + rr_mb
            print(f"  Total model footprint: ~{total_mb}MB")

        return client, reranker, model_key, model_label

    else:
        # HTTP gateway models
        model_key = args.model or "nomic"
        if model_key not in HTTP_MODELS:
            print(f"ERROR: Unknown HTTP model '{model_key}'. "
                  f"Options: {', '.join(HTTP_MODELS)}")
            return None, None, None, None

        model_name, model_label = HTTP_MODELS[model_key]
        client = EmbeddingClient(
            endpoint="http://localhost:8800/v1/embeddings",
            model=model_name,
        )

        reranker = None
        if args.rerank or args.all:
            reranker = HTTPReranker(
                endpoint="http://localhost:8800/v1/rerank",
                model="jina-reranker",
            )
            try:
                test = reranker.rerank("test", ["test doc"], top_k=1)
                if test and test[0][1] > 0:
                    print(f"  Reranker: OK (jina-reranker via HTTP)")
                else:
                    print("  Reranker: FALLBACK (zero scores, disabling)")
                    reranker = None
            except Exception as e:
                print(f"  Reranker: UNAVAILABLE ({e}), disabling")
                reranker = None

        return client, reranker, model_key, model_label


def main():
    parser = argparse.ArgumentParser(description="Popup Maker search benchmark")
    parser.add_argument("--provider", choices=["http", "fastembed"], default="http",
                        help="Embedding provider (http=gateway, fastembed=local ONNX)")
    parser.add_argument("--model", default=None,
                        help="Model key (http: nomic/qwen3, fastembed: bge-small/arctic)")
    parser.add_argument("--reranker", default=None,
                        help="Reranker key for fastembed (jina-tiny/ms-marco)")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--all", action="store_true", help="Run all modes including PPR")
    parser.add_argument("--quick", action="store_true", help="Only VEC+code and HYBRID+code")
    args = parser.parse_args()

    print("=" * 100)
    print("Popup Maker Real-World Benchmark (Core + Pro)")
    print("=" * 100)

    for path, label in [(PM_CORE, "Core"), (PM_PRO, "Pro")]:
        if not os.path.isdir(path):
            print(f"ERROR: {label} not found at {path}")
            return

    client, reranker, model_key, model_label = create_client_and_reranker(args)
    if client is None:
        return

    # Test embedding
    try:
        test_vec = client.embed_single("test")
        dim = len(test_vec)
        provider_tag = "local" if args.provider == "fastembed" else "HTTP"
        print(f"  Embedding: {model_label} — {dim}d vectors ({provider_tag})")
    except Exception as e:
        print(f"  ERROR: Embedding not available ({e})")
        client.close()
        return

    base_dir = tempfile.mkdtemp(prefix=f"tessera_pm_{model_key}_")
    print(f"  Working dir: {base_dir}")

    # Index core
    core_base = os.path.join(base_dir, "core")
    os.makedirs(core_base, exist_ok=True)
    ProjectDB.base_dir = core_base

    pipeline_core = IndexerPipeline(project_path=PM_CORE, embedding_client=client)
    pipeline_core.register()
    t0 = time.perf_counter()
    stats_core = pipeline_core.index_project()
    core_time = time.perf_counter() - t0
    print(f"  Core: {stats_core.files_processed} files, {stats_core.chunks_created} chunks, "
          f"{stats_core.chunks_embedded} embedded in {core_time:.1f}s")

    # Index pro
    pro_base = os.path.join(base_dir, "pro")
    os.makedirs(pro_base, exist_ok=True)
    ProjectDB.base_dir = pro_base

    pipeline_pro = IndexerPipeline(project_path=PM_PRO, embedding_client=client)
    pipeline_pro.register()
    t0 = time.perf_counter()
    stats_pro = pipeline_pro.index_project()
    pro_time = time.perf_counter() - t0
    print(f"  Pro: {stats_pro.files_processed} files, {stats_pro.chunks_created} chunks, "
          f"{stats_pro.chunks_embedded} embedded in {pro_time:.1f}s")

    total = stats_core.files_processed + stats_pro.files_processed
    print(f"  Total: {total} files indexed in {core_time + pro_time:.1f}s")

    # Open DBs
    ProjectDB.base_dir = core_base
    db_core = ProjectDB(PM_CORE)
    ProjectDB.base_dir = pro_base
    db_pro = ProjectDB(PM_PRO)

    # Load PPR graphs
    graph_core = None
    graph_pro = None
    if args.all:
        print("\n  Loading PPR graphs...")
        ProjectDB.base_dir = core_base
        graph_core = try_load_graph(db_core)
        ProjectDB.base_dir = pro_base
        graph_pro = try_load_graph(db_pro)

    # Define modes
    all_results = {}
    labels = []

    if args.quick:
        modes = [
            ("VEC+code", [SearchType.VEC], ["code"], None, None),
            ("HYBRID+code", None, ["code"], None, None),
        ]
    elif args.all:
        modes = [
            ("VEC+code", [SearchType.VEC], ["code"], None, None),
            ("HYBRID+code", None, ["code"], None, None),
            ("VEC+PPR", [SearchType.VEC], ["code"], graph_core, graph_pro),
            ("HYB+PPR", None, ["code"], graph_core, graph_pro),
        ]
        if reranker:
            modes.extend([
                ("VEC+rerank", [SearchType.VEC], ["code"], None, None),
                ("HYB+rerank", None, ["code"], None, None),
                ("VEC+PPR+rr", [SearchType.VEC], ["code"], graph_core, graph_pro),
                ("FULL", None, ["code"], graph_core, graph_pro),
            ])
    else:
        modes = [
            ("VEC-only", [SearchType.VEC], None, None, None),
            ("LEX-only", [SearchType.LEX], None, None, None),
            ("HYBRID", None, None, None, None),
            ("VEC+code", [SearchType.VEC], ["code"], None, None),
            ("HYBRID+code", None, ["code"], None, None),
        ]

    print(f"\n  Running {len(modes)} search modes x {len(QUERIES)} queries...")

    for mode_label, search_types, src_filter, gc, gp in modes:
        rr = reranker if ("rerank" in mode_label or mode_label == "FULL") else None
        results = run_mode(
            mode_label, search_types, src_filter, client,
            db_core, db_pro, gc, gp, rr,
        )
        all_results[mode_label] = results
        labels.append(mode_label)

    print_comparison(all_results, labels)

    # Cleanup
    db_core.close()
    db_pro.close()
    client.close()
    ProjectDB.base_dir = None
    shutil.rmtree(base_dir, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
