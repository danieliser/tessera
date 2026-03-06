"""Mixed-media benchmark: Code + Documents + Cross-media search.

Indexes popup-maker code AND wppopupmaker website content (docs, guides,
tutorials, features, integrations), then runs queries across three categories:

  1. CODE — queries expecting a source file (same as PM20)
  2. DOCUMENT — queries expecting a content/docs markdown file
  3. CROSS-MEDIA — queries that span code and docs (either is correct)

Reports three MRR scores plus a blended headline MRR.

Usage:
    uv run python scripts/benchmark_mixed.py                    # default: bge-small + jina-tiny
    uv run python scripts/benchmark_mixed.py --all              # all modes + PPR + rerank
    uv run python scripts/benchmark_mixed.py --provider http    # gateway models
    uv run python scripts/benchmark_mixed.py --csv              # export CSV
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

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

# ---- Paths ----

PM_CORE = os.path.expanduser("~/Projects/ProContent/ProductCode/popup-maker")
PM_PRO = os.path.expanduser("~/Projects/ProContent/ProductCode/popup-maker-pro")
PM_DOCS = os.path.expanduser("~/Projects/ProContent/data/wppopupmaker/content")

# ---- Models (same as benchmark_pm.py) ----

HTTP_MODELS = {
    "nomic": ("nomic-embed", "Nomic-768d"),
    "qwen3": ("qwen3-embed", "Qwen3-1024d"),
    "coderank": ("code-rank-embed", "CodeRankEmbed-137M"),
}

FASTEMBED_MODELS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "BGE-small-384d", 67),
    "bge-base": ("BAAI/bge-base-en-v1.5", "BGE-base-768d", 210),
    "gte-base": ("thenlper/gte-base", "GTE-base-768d", 440),
}

FASTEMBED_RERANKERS = {
    "jina-tiny": ("jinaai/jina-reranker-v1-tiny-en", "Jina-tiny-reranker", 130),
    "jina-turbo": ("jinaai/jina-reranker-v1-turbo-en", "Jina-turbo-reranker", 220),
}

# ---- Queries ----
# Each: (query_text, expected_files, description, category)
# expected_files: partial filename matches (basename for code, slug/filename for docs)
# category: "code", "doc", or "cross"

QUERIES = [
    # ═══════════════════════════════════════════════════════════════
    # CODE QUERIES (10) — expecting source files
    # ═══════════════════════════════════════════════════════════════
    ("wp_footer action hook renders popup HTML output to page",
     ["Popups.php"], "Frontend rendering", "code"),

    ("register popup and popup_theme custom post types",
     ["PostTypes.php"], "Post type registration", "code"),

    ("exit intent mouse detection trigger",
     ["Triggers.php", "entry--plugin-init.php"], "Exit intent trigger", "code"),

    ("condition callback evaluation for page targeting",
     ["ConditionCallbacks.php", "Conditions.php"], "Condition callbacks", "code"),

    ("popup cookie expiration session hours days",
     ["Cookies.php", "PopupCookie.php"], "Cookie system", "code"),

    ("newsletter subscription form AJAX submission handler",
     ["Newsletters.php", "Subscribe.php"], "Newsletter AJAX", "code"),

    ("shortcode to trigger popup on click element",
     ["PopupTrigger.php"], "Trigger shortcode", "code"),

    ("REST API endpoint for license validation",
     ["RestAPI.php", "License.php"], "License REST API", "code"),

    ("Pimple dependency injection container service registration",
     ["Container.php", "Core.php"], "DI container", "code"),

    ("popup schedule date range recurring daily weekly",
     ["scheduling.php", "scheduled-actions.php"], "Scheduling", "code"),

    # ═══════════════════════════════════════════════════════════════
    # DOCUMENT QUERIES (10) — expecting markdown content files
    # ═══════════════════════════════════════════════════════════════
    ("how to set up geotargeting for popups by country",
     ["setting-up-geotargeting"], "Geotargeting setup", "doc"),

    ("disable popups on mobile devices responsive",
     ["can-popups-be-disabled-on-mobile-devices"], "Mobile disable", "doc"),

    ("close a popup when a link inside is clicked",
     ["close-a-popup-when-a-link-inside-the-popup-is-clicked"],
     "Close on link click", "doc"),

    ("create exit intent opt-in popup with gravity forms",
     ["exit-intent-opt-in-popup-using-gravity-forms"],
     "Exit intent opt-in guide", "doc"),

    ("WooCommerce cart abandonment popup setup",
     ["woocommerce-cart-abandonment-popup"], "Cart abandonment guide", "doc"),

    ("getting started adding a new popup trigger controls when popup comes into view",
     ["add-a-popup-trigger", "468173"], "Add trigger doc", "doc"),

    ("GDPR data privacy requirements for popups cookie consent",
     ["use-popup-maker-and-meet-the-gdpr-data-privacy-requirements"],
     "GDPR compliance", "doc"),

    ("Beaver Builder popup integration tutorial step by step",
     ["create-beaver-builder-popups"], "Beaver Builder tutorial", "doc"),

    ("popup maker form integration API developer documentation",
     ["popup-maker-form-integration-api"], "Form integration API", "doc"),

    ("export and import popups between WordPress sites",
     ["export-and-import-popups-between-sites"], "Export/import doc", "doc"),

    # ═══════════════════════════════════════════════════════════════
    # CROSS-MEDIA QUERIES (10) — code OR doc answer is correct
    # ═══════════════════════════════════════════════════════════════
    ("Ninja Forms integration popup form submission",
     ["GravityForms.php", "ninja-forms", "NinjaForms.php", "Subscribe.php",
      "form-submit-success"],
     "Ninja Forms integration", "cross"),

    ("how does the popup trigger system work click open",
     ["Triggers.php", "add-a-popup-trigger", "click-triggers",
      "PopupTrigger.php"],
     "Trigger system overview", "cross"),

    ("popup analytics tracking conversion events",
     ["Analytics.php", "Attribution.php", "Conversions.php",
      "popup-analytics", "by-the-numbers"],
     "Analytics & conversions", "cross"),

    ("popup cookie settings frequency control how often to show",
     ["Cookies.php", "PopupCookie.php", "cookie-notification",
      "popup-settings"],
     "Cookie frequency control", "cross"),

    ("popup theme customization CSS styling overlay",
     ["find-the-popup-maker-core-and-popup-theme-css",
      "popup-theme-settings", "Theme.php", "Themes.php"],
     "Theme & CSS styling", "cross"),

    ("FluentCRM tagging automation when popup converts",
     ["FluentCRM.php", "AddTag.php", "fluentcrm"],
     "FluentCRM integration", "cross"),

    ("Elementor button trigger popup on click",
     ["how-to-use-an-elementor-button-to-trigger-a-popup",
      "PopupTrigger.php", "Triggers.php"],
     "Elementor trigger", "cross"),

    ("popup scheduling show between dates recurring",
     ["scheduling.php", "scheduled-actions.php", "popup-schedule",
      "auto-opening-announcement"],
     "Scheduling (code + doc)", "cross"),

    ("subscribe to download using popup and ninja forms",
     ["Subscribe.php", "subscribe-to-download-file",
      "content-upgrade-popups"],
     "Subscribe-to-download", "cross"),

    ("popup maker glossary of terms definitions",
     ["popup-maker-glossary-of-terms", "Glossary.php"],
     "Glossary", "cross"),
]


def evaluate_hits(hit_files: list[str], expected_files: list[str]) -> dict:
    """Score a single query result against ground truth."""
    if not hit_files:
        return {"top1": False, "top3": False, "top5": False, "top10": False, "mrr": 0.0}

    def _matches(filepath: str, expected: list[str]) -> bool:
        # Check basename (for code) and full path components (for doc slugs)
        basename = os.path.basename(filepath)
        return any(exp in basename or exp in filepath for exp in expected)

    top1 = _matches(hit_files[0], expected_files)
    top3 = any(_matches(f, expected_files) for f in hit_files[:3])
    top5 = any(_matches(f, expected_files) for f in hit_files[:5])
    top10 = any(_matches(f, expected_files) for f in hit_files[:10])

    mrr = 0.0
    for rank, f in enumerate(hit_files, 1):
        if _matches(f, expected_files):
            mrr = 1.0 / rank
            break

    return {"top1": top1, "top3": top3, "top5": top5, "top10": top10, "mrr": mrr}


def search_all_dbs(
    query: str,
    query_embedding: np.ndarray | None,
    dbs: list[tuple[str, ProjectDB]],
    graphs: dict[str, object] | None = None,
    search_types=None,
    src_filter=None,
    reranker=None,
    keyword_weight: float | None = None,
) -> list[dict]:
    """Search all indexed DBs, merge by score, optionally rerank."""
    all_hits = []

    for label, db in dbs:
        graph = graphs.get(label) if graphs else None
        hits = hybrid_search(
            query, query_embedding, db,
            graph=graph, limit=10,
            source_type=src_filter, search_types=search_types,
            advanced_fts=False,
            keyword_weight=keyword_weight,
        )
        for h in hits:
            h["_source"] = label
            all_hits.append(h)

    all_hits.sort(
        key=lambda x: x.get("rrf_score", x.get("score", 0)),
        reverse=True,
    )

    # Post-RRF reranking
    if reranker and all_hits:
        docs = [h.get("content", "")[:512] for h in all_hits[:20]]
        reranked = reranker.rerank(query, docs, top_k=10)
        if reranked and reranked[0][1] > 0:
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
    dbs: list[tuple[str, ProjectDB]],
    graphs: dict[str, object] | None = None,
    reranker=None,
    keyword_weight: float | None = None,
    use_hyde: bool = False,
    queries: list | None = None,
) -> list[dict]:
    """Run queries for a single mode, return results with timing."""
    queries = queries or QUERIES
    results = []
    total_ms = 0.0

    for query, expected_files, desc, category in queries:
        needs_embedding = (
            (search_types and SearchType.VEC in search_types)
            or search_types is None
        )

        query_embedding = None
        if needs_embedding:
            raw = client.embed_single(query) if use_hyde else client.embed_query(query)
            query_embedding = np.array(raw, dtype=np.float32)

        t0 = time.perf_counter()
        all_hits = search_all_dbs(
            query, query_embedding, dbs,
            graphs, search_types, src_filter, reranker,
            keyword_weight=keyword_weight,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        hit_files = [h.get("file_path", "") for h in all_hits]
        scores = evaluate_hits(hit_files, expected_files)
        scores["desc"] = desc
        scores["category"] = category
        scores["top_file"] = os.path.basename(hit_files[0]) if hit_files else "—"
        scores["top3_files"] = [os.path.basename(f) for f in hit_files[:3]]
        scores["latency_ms"] = elapsed_ms
        results.append(scores)

    avg_ms = total_ms / len(queries)
    print(f"    {mode_label}: avg {avg_ms:.0f}ms/query")
    return results


def run_routed(
    client,
    dbs: list[tuple[str, ProjectDB]],
    graphs: dict[str, object] | None = None,
    reranker=None,
    keyword_weight: float | None = None,
) -> list[dict]:
    """Run each query with its optimal path based on category.

    Code   → HYDE (embed_single, source_type=code, no reranker)
    Doc    → HYDE+rerank (embed_single, source_type=markdown, reranker)
    Cross  → HYB+rerank (embed_query, no filter, reranker)
    """
    results = []
    total_ms = 0.0

    for query, expected_files, desc, category in QUERIES:
        if category == "code":
            use_hyde = True
            src_filter = ["code"]
            search_types = [SearchType.VEC]
            rr = None
        elif category == "doc":
            use_hyde = True
            src_filter = ["markdown", "json"]
            search_types = [SearchType.VEC]
            rr = reranker
        else:  # cross
            use_hyde = False
            src_filter = None
            search_types = None  # hybrid
            rr = reranker

        raw = client.embed_single(query) if use_hyde else client.embed_query(query)
        query_embedding = np.array(raw, dtype=np.float32)

        t0 = time.perf_counter()
        all_hits = search_all_dbs(
            query, query_embedding, dbs,
            graphs, search_types, src_filter, rr,
            keyword_weight=keyword_weight,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        hit_files = [h.get("file_path", "") for h in all_hits]
        scores = evaluate_hits(hit_files, expected_files)
        scores["desc"] = desc
        scores["category"] = category
        scores["top_file"] = os.path.basename(hit_files[0]) if hit_files else "—"
        scores["top3_files"] = [os.path.basename(f) for f in hit_files[:3]]
        scores["latency_ms"] = elapsed_ms
        results.append(scores)

    avg_ms = total_ms / len(QUERIES)
    print(f"    ROUTED: avg {avg_ms:.0f}ms/query")
    return results


def _category_mrr(results: list[dict], category: str | None) -> float:
    filtered = [r for r in results if category is None or r["category"] == category]
    if not filtered:
        return 0.0
    return sum(r["mrr"] for r in filtered) / len(filtered)


def print_comparison(all_results: dict, labels: list[str]):
    """Print aggregate metrics broken down by category."""
    categories = {"code": "CODE", "doc": "DOCUMENT", "cross": "CROSS-MEDIA"}

    for cat_key, cat_name in [("all", "ALL QUERIES"), *categories.items()]:
        if cat_key == "all":
            filter_fn = lambda _r: True
            n_queries = len(QUERIES)
        else:
            filter_fn = lambda r: r["category"] == cat_key
            n_queries = sum(1 for q in QUERIES if q[3] == cat_key)

        if n_queries == 0:
            continue

        print(f"\n{'=' * 100}")
        print(f"{cat_name} ({n_queries} queries)")
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
                filtered = [x for x in all_results[label] if filter_fn(x)]
                acc = sum(1 for x in filtered if x[key]) / len(filtered) * 100
                row += f"{acc:>15.0f}%"
            print(row)

        row = f"{'MRR':<20}"
        for label in labels:
            filtered = [x for x in all_results[label] if filter_fn(x)]
            mrr = sum(x["mrr"] for x in filtered) / len(filtered)
            row += f"{mrr:>16.3f}"
        print(row)

        row = f"{'Avg Latency (ms)':<20}"
        for label in labels:
            filtered = [x for x in all_results[label] if filter_fn(x)]
            avg = sum(x["latency_ms"] for x in filtered) / len(filtered)
            row += f"{avg:>16.0f}"
        print(row)

    # Routed headline: best-per-category from available results
    print(f"\n{'=' * 100}")
    print("ROUTED HEADLINE (best mode per category)")
    print(f"{'=' * 100}")
    all_labels = labels
    best_per_cat = []
    for ck, cn in [("code", "CODE"), ("doc", "DOC"), ("cross", "CROSS")]:
        best_label = max(
            all_labels,
            key=lambda l: sum(r["mrr"] for r in all_results[l] if r["category"] == ck) /
                          max(1, sum(1 for r in all_results[l] if r["category"] == ck)),
        )
        best_mrr = (sum(r["mrr"] for r in all_results[best_label] if r["category"] == ck) /
                    max(1, sum(1 for r in all_results[best_label] if r["category"] == ck)))
        best_per_cat.append(best_mrr)
        print(f"  {cn:<18}: {best_mrr:.3f}  (via {best_label})")
    blended = sum(best_per_cat) / 3
    print(f"  {'BLENDED MRR':<18}: {blended:.3f}")

    # Per-query breakdown
    print(f"\n{'=' * 100}")
    print("PER-QUERY BREAKDOWN")
    print(f"{'=' * 100}")

    for cat_key, cat_name in categories.items():
        cat_queries = [(i, q) for i, q in enumerate(QUERIES) if q[3] == cat_key]
        if not cat_queries:
            continue
        print(f"\n  --- {cat_name} ---")
        for i, (_, expected, desc, _cat) in cat_queries:
            print(f"\n  Q{i+1}: {desc} — expected: {expected[:3]}{'...' if len(expected) > 3 else ''}")
            for label in labels:
                r = all_results[label][i]
                rank_str = "MISS" if r["mrr"] == 0 else f"rank {int(1/r['mrr'])}"
                hit_marker = "+" if r["top3"] else ("-" if r["top10"] else "X")
                top3 = ", ".join(r["top3_files"])
                print(f"    [{hit_marker}] {label:<16} {rank_str:<8} [{top3}]")


def try_load_graph(db: ProjectDB):
    """Attempt to load PPR graph."""
    try:
        graph = load_project_graph(db, 1)
        print(f"    Graph loaded: {graph.n_symbols} symbols, {graph.edge_count} edges")
        return graph
    except (ValueError, Exception) as e:
        print(f"    Graph load failed: {e}")
        return None


def _needs_reindex(db_base: str, project_path: str, force: bool) -> bool:
    """Check if we need to re-index by comparing git HEAD or content hash."""
    if force:
        return True
    marker = os.path.join(db_base, ".indexed_commit")
    if not os.path.exists(marker):
        return True
    # For git repos, use HEAD. For content dirs, use mtime-based marker.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            head = result.stdout.strip()
            with open(marker) as f:
                return f.read().strip() != head
    except Exception:
        pass
    # Fallback: check if marker exists (for non-git content dirs)
    return not os.path.exists(marker)


def _mark_indexed(db_base: str, project_path: str) -> None:
    """Write git HEAD or timestamp to marker file."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            with open(os.path.join(db_base, ".indexed_commit"), "w") as f:
                f.write(result.stdout.strip())
            return
    except Exception:
        pass
    # Non-git fallback: use timestamp
    with open(os.path.join(db_base, ".indexed_commit"), "w") as f:
        f.write(f"ts:{int(time.time())}")


def create_client_and_reranker(args):
    """Create embedding client and reranker."""
    if args.provider == "fastembed":
        model_key = args.model or "bge-small"
        if model_key not in FASTEMBED_MODELS:
            print(f"ERROR: Unknown model '{model_key}'. Options: {', '.join(FASTEMBED_MODELS)}")
            return None, None, None, None

        model_name, model_label, embed_mb = FASTEMBED_MODELS[model_key]

        # Load embedding FIRST — must be fully warm before second ONNX model loads.
        # Two large ONNX runtimes initializing simultaneously deadlocks on macOS.
        print(f"  Loading embedding: {model_label} ({model_name}, ~{embed_mb}MB)")
        client = FastembedClient(model_name=model_name)
        _ = client.embed_single("warmup")  # force full ONNX graph init

        reranker = None
        if args.rerank or args.all:
            if args.reranker_model:
                rr_endpoint = args.reranker_endpoint or "http://localhost:8800/v1/rerank"
                reranker = HTTPReranker(endpoint=rr_endpoint, model=args.reranker_model)
                try:
                    test = reranker.rerank("test", ["test doc"], top_k=1)
                    if test and test[0][1] > 0:
                        print(f"  Reranker: OK ({args.reranker_model} via {rr_endpoint})")
                    else:
                        print("  Reranker: FALLBACK (zero scores, disabling)")
                        reranker = None
                except Exception as e:
                    print(f"  Reranker: UNAVAILABLE ({e}), disabling")
                    reranker = None
            else:
                reranker_key = args.reranker or "jina-tiny"
                if reranker_key in FASTEMBED_RERANKERS:
                    rr_name, rr_label, rr_mb = FASTEMBED_RERANKERS[reranker_key]
                    print(f"  Loading reranker: {rr_label} ({rr_name}, ~{rr_mb}MB)")
                    reranker = FastembedReranker(model_name=rr_name)

        return client, reranker, model_key, model_label

    else:
        model_key = args.model or "nomic"
        if model_key not in HTTP_MODELS:
            print(f"ERROR: Unknown model '{model_key}'. Options: {', '.join(HTTP_MODELS)}")
            return None, None, None, None

        model_name, model_label = HTTP_MODELS[model_key]
        embed_endpoint = args.embed_endpoint or "http://localhost:8800/v1/embeddings"
        client = EmbeddingClient(endpoint=embed_endpoint, model=model_name)

        reranker = None
        if args.rerank or args.all:
            rr_endpoint = args.reranker_endpoint or "http://localhost:8800/v1/rerank"
            rr_model = args.reranker_model or "jina-reranker"
            reranker = HTTPReranker(endpoint=rr_endpoint, model=rr_model)
            try:
                test = reranker.rerank("test", ["test doc"], top_k=1)
                if test and test[0][1] > 0:
                    print(f"  Reranker: OK ({rr_model} via {rr_endpoint})")
                else:
                    print("  Reranker: FALLBACK (zero scores, disabling)")
                    reranker = None
            except Exception as e:
                print(f"  Reranker: UNAVAILABLE ({e}), disabling")
                reranker = None

        return client, reranker, model_key, model_label


def _run_matrix(args) -> None:
    """Sweep all local models with jina-tiny, print summary table.

    Runs each model as a subprocess to keep ONNX runtimes isolated.
    Results land in benchmarks/mixed_{model}_*.csv — aggregate with
    scripts/aggregate_mixed.py after the run.
    """
    import subprocess

    models_to_run = list(FASTEMBED_MODELS.keys())
    print("=" * 100)
    print("Mixed-Media Matrix Sweep — all local models x jina-tiny")
    print("=" * 100)

    for model_key in models_to_run:
        print(f"\n--- {model_key} ---")
        cmd = [
            sys.executable, __file__,
            "--provider", "fastembed",
            "--model", model_key,
            "--reranker", "jina-tiny",
            "--all", "--csv",
        ]
        if args.reindex:
            cmd.append("--reindex")
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {model_key} exited {result.returncode}")

    print("\n" + "=" * 100)
    print("Matrix complete. CSVs in benchmarks/mixed_*.csv")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Mixed-media search benchmark")
    parser.add_argument("--provider", choices=["http", "fastembed"], default="fastembed")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reranker", default=None, help="Local reranker key")
    parser.add_argument("--reranker-endpoint", default=None)
    parser.add_argument("--reranker-model", default=None, help="HTTP reranker model name")
    parser.add_argument("--embed-endpoint", default=None)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--all", action="store_true", help="All modes including PPR + rerank")
    parser.add_argument("--quick", action="store_true", help="VEC-only and HYBRID-only")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--keyword-weight", type=float, default=None)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--matrix", action="store_true",
                        help="Sweep all local models, report summary table")
    args = parser.parse_args()

    if args.matrix:
        _run_matrix(args)
        return

    print("=" * 100)
    print("Mixed-Media Benchmark: Code + Documents + Cross-Media")
    print("=" * 100)

    # Validate paths
    for path, label in [(PM_CORE, "Core code"), (PM_PRO, "Pro code"), (PM_DOCS, "Docs content")]:
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

    # Index directories
    bench_root = os.path.expanduser(f"~/.tessera/benchmarks/mixed_{model_key}")
    force_reindex = args.reindex

    index_targets = [
        ("core", PM_CORE),
        ("pro", PM_PRO),
        ("docs", PM_DOCS),
    ]

    for idx_label, project_path in index_targets:
        idx_base = os.path.join(bench_root, idx_label)
        os.makedirs(idx_base, exist_ok=True)

        if _needs_reindex(idx_base, project_path, force_reindex):
            # Clear old data
            for f in os.listdir(idx_base):
                fp = os.path.join(idx_base, f)
                if os.path.isfile(fp):
                    os.remove(fp)

            ProjectDB.base_dir = idx_base
            pipeline = IndexerPipeline(
                project_path=project_path,
                embedding_client=client,
            )
            pipeline.register()
            t0 = time.perf_counter()
            stats = pipeline.index_project_sync()
            elapsed = time.perf_counter() - t0
            _mark_indexed(idx_base, project_path)
            print(f"  {idx_label}: {stats.files_processed} files, "
                  f"{stats.chunks_created} chunks, "
                  f"{stats.chunks_embedded} embedded in {elapsed:.1f}s")
        else:
            print(f"  {idx_label}: cached (unchanged)")

    print(f"  Index dir: {bench_root}")

    # Open DBs
    dbs = []
    graphs = {}
    for idx_label, project_path in index_targets:
        idx_base = os.path.join(bench_root, idx_label)
        ProjectDB.base_dir = idx_base
        db = ProjectDB(project_path)
        dbs.append((idx_label, db))

    # Load PPR graphs
    if args.all:
        print("\n  Loading PPR graphs...")
        for idx_label, project_path in index_targets:
            idx_base = os.path.join(bench_root, idx_label)
            ProjectDB.base_dir = idx_base
            db = [d for l, d in dbs if l == idx_label][0]
            g = try_load_graph(db)
            if g:
                graphs[idx_label] = g

    # Define modes
    all_results = {}
    labels = []

    if args.quick:
        modes = [
            ("VEC", [SearchType.VEC], None, False),
            ("HYBRID", None, None, False),
        ]
    elif args.all:
        modes = [
            ("VEC", [SearchType.VEC], None, False),
            ("HYDE", [SearchType.VEC], None, True),
            ("HYBRID", None, None, False),
            ("VEC+PPR", [SearchType.VEC], None, False),  # uses graphs
            ("HYB+PPR", None, None, False),
        ]
        if reranker:
            modes.extend([
                ("VEC+rerank", [SearchType.VEC], None, False),
                ("HYDE+rerank", [SearchType.VEC], None, True),
                ("HYB+rerank", None, None, False),
                ("FULL", None, None, False),  # HYB+PPR+rerank
            ])
    else:
        modes = [
            ("VEC", [SearchType.VEC], None, False),
            ("HYDE", [SearchType.VEC], None, True),
            ("HYBRID", None, None, False),
        ]
        if reranker:
            modes.extend([
                ("VEC+rerank", [SearchType.VEC], None, False),
                ("HYDE+rerank", [SearchType.VEC], None, True),
                ("HYB+rerank", None, None, False),
            ])

    n_queries = len(QUERIES)
    print(f"\n  Running {len(modes)} modes x {n_queries} queries...")

    kw_weight = args.keyword_weight

    for mode_label, search_types, _src_filter, hyde in modes:
        use_graphs = "PPR" in mode_label or mode_label == "FULL"
        use_rerank = "rerank" in mode_label or mode_label == "FULL"
        rr = reranker if use_rerank else None
        grs = graphs if use_graphs else None

        # No source_type filter — we search everything (code + docs)
        results = run_mode(
            mode_label, search_types, None, client,
            dbs, grs, rr,
            keyword_weight=kw_weight,
            use_hyde=hyde,
        )
        all_results[mode_label] = results
        labels.append(mode_label)

    # Always run the routed path (code->HYDE, doc->HYDE+rr, cross->HYB+rr)
    if reranker:
        print("    Running ROUTED mode...")
        routed_results = run_routed(client, dbs, keyword_weight=kw_weight, reranker=reranker)
        all_results["ROUTED"] = routed_results
        labels.append("ROUTED")

    print_comparison(all_results, labels)

    # CSV export
    if args.csv:
        os.makedirs("benchmarks", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"benchmarks/mixed_{model_key}_{ts}.csv"
        fieldnames = [
            "model", "mode", "query_id", "query_desc", "category",
            "expected", "top_file", "mrr", "top1", "top3", "top5", "top10",
            "latency_ms", "reranker",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for label in labels:
                for i, r in enumerate(all_results[label]):
                    writer.writerow({
                        "model": model_key,
                        "mode": label,
                        "query_id": i,
                        "query_desc": r["desc"],
                        "category": r["category"],
                        "expected": QUERIES[i][1][0] if QUERIES[i][1] else "",
                        "top_file": r["top_file"],
                        "mrr": f"{r['mrr']:.4f}",
                        "top1": int(r["top1"]),
                        "top3": int(r["top3"]),
                        "top5": int(r["top5"]),
                        "top10": int(r["top10"]),
                        "latency_ms": f"{r['latency_ms']:.1f}",
                        "reranker": "yes" if reranker else "no",
                    })
        print(f"\n  CSV exported: {csv_path}")

    # Cleanup
    for _, db in dbs:
        db.close()
    client.close()
    ProjectDB.base_dir = None

    print("\nDone.")


if __name__ == "__main__":
    main()
