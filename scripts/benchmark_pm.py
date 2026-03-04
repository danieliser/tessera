"""Real-world benchmark: Popup Maker (core + Pro) — PHP codebase, ~580 files.

Indexes both popup-maker and popup-maker-pro, then runs 20 search queries with
known ground-truth file expectations. Tests VEC-only, LEX-only, and HYBRID modes.
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

PM_CORE = os.path.expanduser(
    "~/Projects/ProContent/ProductCode/popup-maker"
)
PM_PRO = os.path.expanduser(
    "~/Projects/ProContent/ProductCode/popup-maker-pro"
)

# Ground truth: (query, expected_files, description)
# expected_files are basenames (or partial) that SHOULD appear in top results.
QUERIES = [
    # --- Core popup lifecycle ---
    (
        "popup rendering in WordPress footer hook",
        ["Popups.php"],  # Controllers/Frontend/Popups.php
        "Frontend rendering",
    ),
    (
        "register popup and popup_theme custom post types",
        ["PostTypes.php"],
        "Post type registration",
    ),
    (
        "popup data model with triggers conditions and settings",
        ["Popup.php"],  # Model/Popup.php
        "Popup model",
    ),
    # --- Trigger system ---
    (
        "singleton registry for popup trigger types",
        ["Triggers.php"],
        "Trigger registry",
    ),
    (
        "exit intent mouse detection trigger",
        ["Triggers.php", "entry--plugin-init.php"],
        "Exit intent trigger",
    ),
    # --- Conditions / targeting ---
    (
        "condition callback evaluation for page targeting",
        ["ConditionCallbacks.php", "Conditions.php"],
        "Condition callbacks",
    ),
    (
        "check if popup is loadable on current page",
        ["conditionals.php"],  # functions/popups/conditionals.php
        "Popup conditionals",
    ),
    # --- Cookies / display frequency ---
    (
        "popup cookie expiration session hours days",
        ["Cookies.php", "PopupCookie.php"],
        "Cookie system",
    ),
    # --- Newsletter / forms ---
    (
        "newsletter subscription form AJAX submission handler",
        ["Newsletters.php", "Subscribe.php"],
        "Newsletter AJAX",
    ),
    (
        "Gravity Forms integration for popup form submission",
        ["GravityForms.php"],
        "Gravity Forms integration",
    ),
    # --- Shortcodes ---
    (
        "shortcode to trigger popup on click element",
        ["PopupTrigger.php"],
        "Trigger shortcode",
    ),
    (
        "pum_subscribe email subscription shortcode",
        ["Subscribe.php"],
        "Subscribe shortcode",
    ),
    # --- REST API / AJAX ---
    (
        "REST API endpoint for license validation",
        ["RestAPI.php", "License.php"],
        "License REST API",
    ),
    (
        "AJAX analytics tracking popup open and conversion events",
        ["Analytics.php"],
        "Analytics tracking",
    ),
    # --- Admin ---
    (
        "admin settings page for global plugin options",
        ["Settings.php"],  # Admin/Settings.php
        "Admin settings",
    ),
    (
        "Gutenberg block editor integration for popups",
        ["BlockEditor.php"],
        "Block editor",
    ),
    # --- DI / architecture ---
    (
        "Pimple dependency injection container service registration",
        ["Container.php", "Core.php"],
        "DI container",
    ),
    # --- Pro features ---
    (
        "FluentCRM add tag to contact on popup conversion",
        ["AddTag.php", "FluentCRM.php"],
        "FluentCRM tagging",
    ),
    (
        "popup schedule date range recurring daily weekly",
        ["scheduling.php", "scheduled-actions.php"],
        "Scheduling",
    ),
    (
        "attribution model calculation for conversion tracking",
        ["Attribution.php", "Conversions.php"],
        "Attribution service",
    ),
]


def run_search(db, client, queries, search_types=None, label=""):
    results = []
    for query, expected_files, desc in queries:
        if search_types and SearchType.VEC in search_types:
            raw = client.embed_query(query)
            query_embedding = np.array(raw, dtype=np.float32)
        elif search_types is None:
            raw = client.embed_query(query)
            query_embedding = np.array(raw, dtype=np.float32)
        else:
            query_embedding = None

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


def print_comparison(all_results, labels):
    n = len(QUERIES)

    print(f"\n{'=' * 80}")
    print("AGGREGATE METRICS")
    print(f"{'=' * 80}")

    header = f"{'Metric':<20}" + "".join(f"{l:>20}" for l in labels)
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
            row += f"{acc:>19.0f}%"
        print(row)

    row = f"{'MRR':<20}"
    for label in labels:
        r = all_results[label]
        mrr = sum(x["mrr"] for x in r) / n
        row += f"{mrr:>20.3f}"
    print(row)

    print(f"\n{'=' * 80}")
    print("PER-QUERY BREAKDOWN")
    print(f"{'=' * 80}")

    for i, (_, expected, desc) in enumerate(QUERIES):
        print(f"\n  Q{i+1}: {desc} — expected: {expected}")
        for label in labels:
            r = all_results[label][i]
            rank_str = "MISS" if r["mrr"] == 0 else f"rank {int(1/r['mrr'])}"
            marker = "+" if r["top3"] else ("-" if r["top10"] else "X")
            top3 = ", ".join(r["top3_files"])
            print(f"    [{marker}] {label:<18} {rank_str:<8} [{top3}]")


def main():
    print("=" * 80)
    print("Popup Maker Real-World Benchmark (Core + Pro)")
    print("=" * 80)

    for path, label in [(PM_CORE, "Core"), (PM_PRO, "Pro")]:
        if not os.path.isdir(path):
            print(f"ERROR: {label} not found at {path}")
            return

    base_dir = tempfile.mkdtemp(prefix="tessera_pm_bench_")
    print(f"Working dir: {base_dir}")

    client = EmbeddingClient(
        endpoint="http://localhost:8800/v1/embeddings",
        model="nomic-embed",
    )

    # Index both core and pro into same DB
    all_results = {}
    labels = []

    for project_path, project_label in [(PM_CORE, "Core"), (PM_PRO, "Pro")]:
        model_base = os.path.join(base_dir, project_label.lower())
        os.makedirs(model_base, exist_ok=True)
        ProjectDB.base_dir = model_base

        pipeline = IndexerPipeline(
            project_path=project_path, embedding_client=client
        )
        pipeline.register()

        start = time.perf_counter()
        stats = pipeline.index_project()
        elapsed = time.perf_counter() - start

        print(
            f"\n  {project_label}: {stats.files_processed} files, "
            f"{stats.chunks_created} chunks, {stats.chunks_embedded} embedded "
            f"in {elapsed:.1f}s"
        )

    # Use core DB for queries (it has the core files)
    # Actually, we need to search both. Let's index both into the same project.
    print("\n--- Indexing combined (Core + Pro) ---")
    combined_base = os.path.join(base_dir, "combined")
    os.makedirs(combined_base, exist_ok=True)
    ProjectDB.base_dir = combined_base

    # Index core
    pipeline_core = IndexerPipeline(
        project_path=PM_CORE, embedding_client=client
    )
    pipeline_core.register()
    stats_core = pipeline_core.index_project()

    # Index pro into same DB location but different project
    pipeline_pro = IndexerPipeline(
        project_path=PM_PRO, embedding_client=client
    )
    pipeline_pro.register()
    stats_pro = pipeline_pro.index_project()

    total_files = stats_core.files_processed + stats_pro.files_processed
    total_chunks = stats_core.chunks_created + stats_pro.chunks_created
    total_embedded = stats_core.chunks_embedded + stats_pro.chunks_embedded
    print(f"  Combined: {total_files} files, {total_chunks} chunks, {total_embedded} embedded")

    # Open the core DB for searching (pro will be in its own DB)
    # For a proper search we need to search each DB and merge...
    # Actually the simpler approach: just search core DB and pro DB separately
    # and report which finds the expected file.

    # Search core
    db_core = ProjectDB(PM_CORE)
    # Search pro
    db_pro = ProjectDB(PM_PRO)

    modes = [
        ("VEC-only", [SearchType.VEC], None),
        ("LEX-only", [SearchType.LEX], None),
        ("HYBRID", None, None),
        ("VEC+code", [SearchType.VEC], ["code"]),
        ("LEX+code", [SearchType.LEX], ["code"]),
        ("HYBRID+code", None, ["code"]),
    ]

    for mode_label, search_types, src_filter in modes:
        print(f"\n  Running {mode_label} queries...")
        results = []
        for query, expected_files, desc in QUERIES:
            if search_types and SearchType.VEC in search_types:
                raw = client.embed_query(query)
                query_embedding = np.array(raw, dtype=np.float32)
            elif search_types is None:
                raw = client.embed_query(query)
                query_embedding = np.array(raw, dtype=np.float32)
            else:
                query_embedding = None

            # Search both DBs
            hits_core = hybrid_search(
                query, query_embedding, db_core,
                graph=None, limit=10,
                source_type=src_filter, search_types=search_types,
                advanced_fts=False, rrf_weights=None,
            )
            hits_pro = hybrid_search(
                query, query_embedding, db_pro,
                graph=None, limit=10,
                source_type=src_filter, search_types=search_types,
                advanced_fts=False, rrf_weights=None,
            )

            # Merge and interleave by score
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

            hit_files = [os.path.basename(h.get("file_path", "")) for h in all_hits]

            top1 = any(exp in hit_files[0] for exp in expected_files) if hit_files else False
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
                "mrr": mrr, "top_file": hit_files[0] if hit_files else "—",
                "top3_files": hit_files[:3],
            })

        all_results[mode_label] = results
        labels.append(mode_label)

    print_comparison(all_results, labels)

    db_core.close()
    db_pro.close()
    client.close()
    ProjectDB.base_dir = None
    shutil.rmtree(base_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
