"""Run the predeclared PR-04 file/path candidate-channel comparison.

The public development corpus is used only after the path representation,
ranking constants, and gates in PR-04.md have been frozen. Control and
treatment run on the same index; the sole retrieval difference is the explicit
``enable_path_search`` ablation.

Example:
    uv run python scripts/benchmark_path_channel.py --reindex \
      --output benchmarks/development-path-channel-2026-07-14.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_development import (  # noqa: E402
    BASELINE_MODEL,
    BASELINE_SEED,
    DEFAULT_INDEX_ROOT,
    TOP_K,
    _compare_baseline,
    _dedupe_paths,
    _git_revision,
    _index_repository,
    _latency_metrics,
    _source_filter,
    _tracked_worktree_changes,
    rank_expected,
)
from benchmark_governance import (  # noqa: E402
    DEFAULT_CHECKOUT_ROOT,
    DEFAULT_MANIFEST,
    build_run_metadata,
    checkout_repository,
    load_manifest,
    load_queries,
    macro_per_repository,
    manifest_digest,
    paired_bootstrap_interval,
    paired_candidate_recall_interval,
    protected_segment_metrics,
    repositories_for_split,
    retrieval_metrics,
    validate_labels_against_checkout,
)

from tessera.db import ProjectDB  # noqa: E402
from tessera.db._utils import PATH_QUERY_TOKEN_LIMIT  # noqa: E402
from tessera.embeddings import FastembedClient  # noqa: E402
from tessera.search import DEFAULT_RRF_WEIGHTS, hybrid_search  # noqa: E402
from tessera.search_trace import (  # noqa: E402
    SearchTrace,
    candidate_retrieval_diagnostics,
    stratified_candidate_diagnostics,
)

CONTROL_ENGINE = "content_semantic_control"
TREATMENT_ENGINE = "path_channel_treatment"
ENGINES = [CONTROL_ENGINE, TREATMENT_ENGINE]
DEFAULT_FROZEN_BASELINE = REPO_ROOT / "benchmarks" / "development-baseline-2026-07-14.json"
PATH_LATENCY_P95_BUDGET_MS = 10.0
END_TO_END_ADDED_P95_BUDGET_MS = 10.0
END_TO_END_P95_RATIO_BUDGET = 1.25
PATH_INDEX_ABSOLUTE_BUDGET_BYTES = 2 * 1024 * 1024
PATH_INDEX_RELATIVE_BUDGET = 0.10
PATH_BUILD_TIME_RATIO_BUDGET = 0.10


def _host_snapshot() -> dict[str, Any]:
    load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
    return {
        "load_average_1m": load[0],
        "load_average_5m": load[1],
        "load_average_15m": load[2],
        "logical_cpu_count": os.cpu_count(),
    }


def _database_storage(database: ProjectDB) -> dict[str, Any]:
    """Measure the FTS shadow tables and complete SQLite allocation."""
    database.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    page_size = int(database.conn.execute("PRAGMA page_size").fetchone()[0])
    page_count = int(database.conn.execute("PRAGMA page_count").fetchone()[0])
    path_bytes = int(
        database.conn.execute(
            "SELECT COALESCE(SUM(pgsize), 0) FROM dbstat WHERE name GLOB 'file_paths_fts*'"
        ).fetchone()[0]
    )
    return {
        "database_bytes": page_size * page_count,
        "path_index_bytes": path_bytes,
        "path_index_rows": int(
            database.conn.execute("SELECT COUNT(*) FROM file_paths_fts").fetchone()[0]
        ),
    }


def _measure_path_rebuild(database: ProjectDB) -> dict[str, Any]:
    started = time.perf_counter()
    indexed = database.rebuild_file_path_index()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "path_build_ms": round(elapsed_ms, 3),
        "paths_indexed": indexed,
        **_database_storage(database),
    }


def _path_latency_samples(
    case: dict[str, Any],
    database: ProjectDB,
    repetitions: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        database.path_search(
            case["query"],
            limit=TOP_K,
            source_type=_source_filter(case["category"]),
        )
        samples.append(round((time.perf_counter() - started) * 1000, 6))
    return samples


def _run_case(
    case: dict[str, Any],
    database: ProjectDB,
    embedder: FastembedClient,
    engine: str,
    *,
    path_latency_repetitions: int,
) -> dict[str, Any]:
    enable_path_search = engine == TREATMENT_ENGINE
    started = time.perf_counter()
    embedding_started = time.perf_counter()
    embedding = np.asarray(embedder.embed_query(case["query"]), dtype=np.float32)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    search_started = time.perf_counter()
    hits = hybrid_search(
        case["query"],
        embedding,
        database,
        graph=None,
        limit=TOP_K,
        source_type=_source_filter(case["category"]),
        file_dedup=True,
        enable_path_search=enable_path_search,
    )
    search_ms = (time.perf_counter() - search_started) * 1000
    latency_ms = (time.perf_counter() - started) * 1000

    trace = SearchTrace(project_name=case["repository"])
    traced_hits = hybrid_search(
        case["query"],
        embedding,
        database,
        graph=None,
        limit=TOP_K,
        source_type=_source_filter(case["category"]),
        file_dedup=True,
        trace=trace,
        enable_path_search=enable_path_search,
    )
    if [hit["id"] for hit in traced_hits] != [hit["id"] for hit in hits]:
        raise RuntimeError(
            f"Tracing changed result order for {case['repository']}/{case['case_id']}/{engine}"
        )

    paths = _dedupe_paths(hits)[:TOP_K]
    rank = rank_expected(paths, case["expected_files"])
    diagnostics = candidate_retrieval_diagnostics(trace, case["expected_files"])
    return {
        **case,
        "engine": engine,
        "supported": True,
        "rank": rank,
        "reciprocal_rank": round(1.0 / rank, 6) if rank else 0.0,
        "top_files": paths,
        "latency_ms": round(latency_ms, 3),
        "embedding_ms": round(embedding_ms, 3),
        "search_ms": round(search_ms, 3),
        "path_latency_samples_ms": _path_latency_samples(
            case,
            database,
            path_latency_repetitions,
        )
        if enable_path_search
        else [],
        "candidate_diagnostics": diagnostics,
        "trace_attribution_complete": diagnostics["attribution_complete"],
    }


def _protected_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dimension in ("language", "content_type"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get(dimension):
                grouped[str(row[dimension])].append(row)
        output[dimension] = {}
        for value, segment in sorted(grouped.items()):
            control = [row for row in segment if row["engine"] == CONTROL_ENGINE]
            treatment = [row for row in segment if row["engine"] == TREATMENT_ENGINE]
            if not control or len(control) != len(treatment):
                continue
            control_metrics = retrieval_metrics(control)
            treatment_metrics = retrieval_metrics(treatment)
            output[dimension][value] = {
                "paired_queries": len(control),
                "control_mrr_at_10": control_metrics["mrr_at_10"],
                "treatment_mrr_at_10": treatment_metrics["mrr_at_10"],
                "delta_mrr_at_10": round(
                    treatment_metrics["mrr_at_10"] - control_metrics["mrr_at_10"],
                    6,
                ),
            }
    return output


def _index_operations(repositories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [metadata["index"] for metadata in repositories.values()]
    uncached = all(not row.get("cached", False) and row.get("index_ms") is not None for row in rows)
    full_index_ms = sum(float(row.get("index_ms", 0.0)) for row in rows)
    path_build_ms = sum(float(row["path_build_ms"]) for row in rows)
    database_bytes = sum(int(row["database_bytes"]) for row in rows)
    path_index_bytes = sum(int(row["path_index_bytes"]) for row in rows)
    return {
        "repositories_measured": len(rows),
        "all_repositories_cold_indexed": uncached,
        "full_index_ms": round(full_index_ms, 3),
        "path_build_ms": round(path_build_ms, 3),
        "path_build_time_ratio": round(path_build_ms / full_index_ms, 6)
        if full_index_ms
        else None,
        "database_bytes": database_bytes,
        "path_index_bytes": path_index_bytes,
        "path_index_ratio": round(path_index_bytes / database_bytes, 6)
        if database_bytes
        else None,
        "path_index_budget_bytes": max(
            PATH_INDEX_ABSOLUTE_BUDGET_BYTES,
            int(database_bytes * PATH_INDEX_RELATIVE_BUDGET),
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repository", action="append", help="Development repository id; repeat as needed")
    parser.add_argument("--tier", choices=["quick", "standard", "full"], default="quick")
    parser.add_argument("--checkout-root", type=Path, default=DEFAULT_CHECKOUT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--path-latency-repetitions", type=int, default=3)
    parser.add_argument("--compare-baseline", type=Path, default=DEFAULT_FROZEN_BASELINE)
    parser.add_argument("--skip-baseline-comparison", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.path_latency_repetitions < 1:
        raise SystemExit("--path-latency-repetitions must be positive")
    manifest = load_manifest(args.manifest)
    available = {
        repository["id"]: repository
        for repository in repositories_for_split(manifest, "development")
    }
    selected_ids = args.repository or list(available)
    unknown = sorted(set(selected_ids) - available.keys())
    if unknown:
        raise SystemExit(f"Unknown development repositories: {', '.join(unknown)}")
    if _tracked_worktree_changes(REPO_ROOT):
        raise SystemExit(
            "Path-channel benchmarks require a clean tracked Tessera worktree so the revision identifies the code exactly"
        )

    tessera_revision = _git_revision(REPO_ROOT)
    corpus_digest = manifest_digest(args.manifest)
    host_start = _host_snapshot()
    embedder = FastembedClient(model_name=BASELINE_MODEL)
    _ = embedder.embed_query("path-channel benchmark warmup")

    revisions: dict[str, str] = {}
    repository_metadata: dict[str, dict[str, Any]] = {}
    databases: dict[str, ProjectDB] = {}
    cases: list[dict[str, Any]] = []
    try:
        for repository_id in selected_ids:
            repository = available[repository_id]
            checkout = checkout_repository(repository, args.checkout_root)
            repository_cases = load_queries(manifest, repository_id, args.tier)
            validate_labels_against_checkout(repository, repository_cases, checkout)
            revision = _git_revision(checkout)
            revisions[repository_id] = revision
            database, index_metadata = _index_repository(
                repository,
                checkout,
                embedder,
                args.index_root,
                tessera_revision,
                corpus_digest,
                reindex=args.reindex,
            )
            path_index_metadata = _measure_path_rebuild(database)
            database.path_search("path channel warmup", limit=TOP_K)
            databases[repository_id] = database
            repository_metadata[repository_id] = {
                "repository_url": repository["repository_url"],
                "revision": revision,
                "license": repository["license"],
                "languages": repository["languages"],
                "content_types": repository["content_types"],
                "index": {**index_metadata, **path_index_metadata},
            }
            cases.extend(repository_cases)

        rows: list[dict[str, Any]] = []
        total = len(cases) * len(ENGINES)
        completed = 0
        for case_index, case in enumerate(cases):
            offset = case_index % len(ENGINES)
            run_order = ENGINES[offset:] + ENGINES[:offset]
            by_engine: dict[str, dict[str, Any]] = {}
            for engine in run_order:
                result = _run_case(
                    case,
                    databases[case["repository"]],
                    embedder,
                    engine,
                    path_latency_repetitions=args.path_latency_repetitions,
                )
                by_engine[engine] = result
                completed += 1
                rank = result["rank"] if result["rank"] is not None else "MISS"
                print(f"[{completed}/{total}] {case['repository']} / {case['case_id']} / {engine}: {rank}")
            rows.extend(by_engine[engine] for engine in ENGINES)

        macro = macro_per_repository(rows)
        latency = {
            engine: _latency_metrics([row for row in rows if row["engine"] == engine])
            for engine in ENGINES
        }
        search_latency = {
            engine: _latency_metrics(
                [
                    {"latency_ms": row["search_ms"]}
                    for row in rows
                    if row["engine"] == engine
                ]
            )
            for engine in ENGINES
        }
        path_samples = [
            {"latency_ms": sample}
            for row in rows
            if row["engine"] == TREATMENT_ENGINE
            for sample in row["path_latency_samples_ms"]
        ]
        path_latency = _latency_metrics(path_samples)
        paired_mrr = paired_bootstrap_interval(
            rows,
            CONTROL_ENGINE,
            TREATMENT_ENGINE,
            seed=BASELINE_SEED,
        )
        paired_candidate = {
            str(cutoff): paired_candidate_recall_interval(
                rows,
                CONTROL_ENGINE,
                TREATMENT_ENGINE,
                cutoff=cutoff,
                seed=BASELINE_SEED,
            )
            for cutoff in (10, 20, 50)
        }
        protected_deltas = _protected_deltas(rows)
        index_operations = _index_operations(repository_metadata)
        control_p95 = latency[CONTROL_ENGINE]["p95_ms"]
        treatment_p95 = latency[TREATMENT_ENGINE]["p95_ms"]
        protected_regressions = [
            f"{dimension}/{value}"
            for dimension, segments in protected_deltas.items()
            for value, metrics in segments.items()
            if metrics["paired_queries"] >= 5 and metrics["delta_mrr_at_10"] < -0.05
        ]
        gates = {
            "candidate_recall_at_10_point_delta_positive": paired_candidate["10"]["delta"] > 0,
            "candidate_recall_at_10_ci_lower_positive": paired_candidate["10"]["ci_lower"] > 0,
            "candidate_recall_at_20_non_negative": paired_candidate["20"]["delta"] >= 0,
            "candidate_recall_at_50_non_negative": paired_candidate["50"]["delta"] >= 0,
            "paired_macro_mrr_ci_lower_at_least_minus_0_02": paired_mrr["ci_lower"] >= -0.02,
            "protected_segments_within_minus_0_05": not protected_regressions,
            "all_candidate_traces_complete": all(row["trace_attribution_complete"] for row in rows),
            "path_search_p95_at_most_10_ms": path_latency["p95_ms"] <= PATH_LATENCY_P95_BUDGET_MS,
            "end_to_end_added_p95_at_most_10_ms": (
                treatment_p95 - control_p95 <= END_TO_END_ADDED_P95_BUDGET_MS
            ),
            "end_to_end_p95_ratio_at_most_1_25": (
                treatment_p95 / control_p95 <= END_TO_END_P95_RATIO_BUDGET
                if control_p95
                else False
            ),
            "path_index_within_storage_budget": (
                index_operations["path_index_bytes"]
                <= index_operations["path_index_budget_bytes"]
            ),
            "path_build_time_at_most_10_percent": bool(
                index_operations["all_repositories_cold_indexed"]
                and index_operations["path_build_time_ratio"] is not None
                and index_operations["path_build_time_ratio"] <= PATH_BUILD_TIME_RATIO_BUDGET
            ),
        }
        gates["passed"] = all(gates.values())

        per_repository = {
            repository_id: {
                engine: retrieval_metrics(
                    [
                        row
                        for row in rows
                        if row["repository"] == repository_id and row["engine"] == engine
                    ]
                )
                for engine in ENGINES
            }
            for repository_id in selected_ids
        }
        control_rows = [row for row in rows if row["engine"] == CONTROL_ENGINE]
        baseline_comparison = (
            _compare_baseline(control_rows, args.compare_baseline)
            if args.compare_baseline and not args.skip_baseline_comparison
            else None
        )
        metadata = build_run_metadata(
            args.manifest,
            "development",
            revisions,
            command=sys.argv,
            seed=BASELINE_SEED,
            extra={
                "tier": args.tier,
                "query_cases": len(cases),
                "top_k": TOP_K,
                "tessera_revision": tessera_revision,
                "embedding_model": BASELINE_MODEL,
                "engines": ENGINES,
                "engine_order_rotated": True,
                "path_rrf_weight": DEFAULT_RRF_WEIGHTS["path"],
                "path_bm25_weights": ProjectDB.PATH_BM25_WEIGHTS,
                "path_query_token_limit": PATH_QUERY_TOKEN_LIMIT,
                "path_latency_repetitions": args.path_latency_repetitions,
            },
        )
        output = {
            "metadata": metadata,
            "host": {"start": host_start, "end": _host_snapshot()},
            "repositories": repository_metadata,
            "summary": {
                "primary_macro_per_repository": macro,
                "pooled": {
                    engine: retrieval_metrics([row for row in rows if row["engine"] == engine])
                    for engine in ENGINES
                },
                "per_repository": per_repository,
                "protected_segments": protected_segment_metrics(rows),
                "protected_segment_deltas": protected_deltas,
                "protected_regressions": protected_regressions,
                "candidate_diagnostics": {
                    engine: stratified_candidate_diagnostics(
                        [row for row in rows if row["engine"] == engine]
                    )
                    for engine in ENGINES
                },
                "paired_candidate_recall": paired_candidate,
                "paired_macro_mrr": paired_mrr,
                "latency": latency,
                "search_only_latency": search_latency,
                "path_search_latency": path_latency,
                "index_operations": index_operations,
                "frozen_baseline_control_comparison": baseline_comparison,
                "predeclared_gates": gates,
            },
            "queries": rows,
            "limitations": [
                "This is a public development treatment, not a sealed holdout result.",
                "The natural-query corpus has only 30 cases and ceilinged baseline candidate recall.",
                "File-level labels do not score the usefulness of the representative chunk.",
                "Path rebuild time measures the derived sparse view in isolation; full indexing includes embeddings.",
                "Host load is captured because end-to-end latency is sensitive to unrelated machine contention.",
            ],
        }
        rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered)
    finally:
        for database in databases.values():
            database.close()
        embedder.close()
        ProjectDB.base_dir = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
