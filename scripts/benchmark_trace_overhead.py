#!/usr/bin/env python3
"""Measure opt-in candidate-trace overhead on the development corpus.

The harness alternates traced and untraced warm searches with one precomputed
query embedding.  It asserts exact result equality for every pair and reports
both relative and absolute p95 overhead against PR-02's declared budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import fmean
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
    _git_revision,
    _index_repository,
    _source_filter,
    _tracked_worktree_changes,
)
from benchmark_governance import (  # noqa: E402
    DEFAULT_CHECKOUT_ROOT,
    DEFAULT_MANIFEST,
    build_run_metadata,
    checkout_repository,
    load_manifest,
    load_queries,
    manifest_digest,
    repositories_for_split,
    validate_labels_against_checkout,
)

from tessera.db import ProjectDB  # noqa: E402
from tessera.embeddings import FastembedClient  # noqa: E402
from tessera.search import hybrid_search  # noqa: E402
from tessera.search_trace import SearchTrace  # noqa: E402

DEFAULT_REPETITIONS = 10
DEFAULT_WARMUPS = 2
MAX_ADDED_P95_MS = 25.0
MAX_P95_RATIO = 1.5
MAX_TRACE_BYTES = 250 * 1024


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(fmean(values), 4) if values else 0.0,
        "p50_ms": round(_percentile(values, 0.5), 4),
        "p95_ms": round(_percentile(values, 0.95), 4),
    }


def _search(
    case: dict[str, Any],
    database: ProjectDB,
    embedding: np.ndarray,
    *,
    traced: bool,
) -> tuple[list[dict[str, Any]], float, SearchTrace | None]:
    trace = SearchTrace(project_name=case["repository"]) if traced else None
    started = time.perf_counter()
    results = hybrid_search(
        case["query"],
        embedding,
        database,
        graph=None,
        limit=TOP_K,
        source_type=_source_filter(case["category"]),
        file_dedup=True,
        trace=trace,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return results, elapsed_ms, trace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repository", action="append", help="Development repository id; repeat as needed")
    parser.add_argument("--tier", choices=["quick", "standard", "full"], default="quick")
    parser.add_argument("--checkout-root", type=Path, default=DEFAULT_CHECKOUT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        raise SystemExit("repetitions must be positive and warmups non-negative")
    if _tracked_worktree_changes(REPO_ROOT):
        raise SystemExit("Trace benchmarks require a clean tracked Tessera worktree")

    manifest = load_manifest(args.manifest)
    available = {
        repository["id"]: repository
        for repository in repositories_for_split(manifest, "development")
    }
    selected_ids = args.repository or list(available)
    unknown = sorted(set(selected_ids) - available.keys())
    if unknown:
        raise SystemExit(f"Unknown development repositories: {', '.join(unknown)}")

    tessera_revision = _git_revision(REPO_ROOT)
    corpus_digest = manifest_digest(args.manifest)
    embedder = FastembedClient(model_name=BASELINE_MODEL)
    _ = embedder.embed_query("candidate trace benchmark warmup")
    databases: dict[str, ProjectDB] = {}
    revisions: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
    try:
        for repository_id in selected_ids:
            repository = available[repository_id]
            checkout = checkout_repository(repository, args.checkout_root)
            repository_cases = load_queries(manifest, repository_id, args.tier)
            validate_labels_against_checkout(repository, repository_cases, checkout)
            revisions[repository_id] = _git_revision(checkout)
            database, _index_metadata = _index_repository(
                repository,
                checkout,
                embedder,
                args.index_root,
                tessera_revision,
                corpus_digest,
                reindex=args.reindex,
            )
            databases[repository_id] = database
            cases.extend(repository_cases)

        query_rows: list[dict[str, Any]] = []
        all_untraced: list[float] = []
        all_traced: list[float] = []
        all_paired_deltas: list[float] = []
        max_trace_bytes = 0

        for case_index, case in enumerate(cases, start=1):
            database = databases[case["repository"]]
            embedding = np.asarray(embedder.embed_query(case["query"]), dtype=np.float32)
            for _ in range(args.warmups):
                _search(case, database, embedding, traced=False)
                _search(case, database, embedding, traced=True)

            timings = {False: [], True: []}
            reference: list[dict[str, Any]] | None = None
            trace_sizes: list[int] = []
            for repetition in range(args.repetitions):
                order = (False, True) if repetition % 2 == 0 else (True, False)
                paired: dict[bool, float] = {}
                for traced in order:
                    results, elapsed_ms, trace = _search(
                        case,
                        database,
                        embedding,
                        traced=traced,
                    )
                    if reference is None:
                        reference = results
                    elif results != reference:
                        raise RuntimeError(
                            f"Trace changed results for {case['repository']}/{case['case_id']}"
                        )
                    timings[traced].append(elapsed_ms)
                    paired[traced] = elapsed_ms
                    if trace is not None:
                        problems = trace.validate_complete_attribution()
                        if problems:
                            raise RuntimeError(
                                f"Incomplete trace for {case['repository']}/{case['case_id']}: {problems}"
                            )
                        trace_sizes.append(len(trace.to_json().encode("utf-8")))
                all_paired_deltas.append(paired[True] - paired[False])

            all_untraced.extend(timings[False])
            all_traced.extend(timings[True])
            max_trace_bytes = max(max_trace_bytes, max(trace_sizes, default=0))
            query_rows.append({
                "repository": case["repository"],
                "case_id": case["case_id"],
                "language": case["language"],
                "content_type": case["content_type"],
                "untraced": _timing_summary(timings[False]),
                "traced": _timing_summary(timings[True]),
                "max_trace_bytes": max(trace_sizes, default=0),
                "results_equal": True,
                "attribution_complete": True,
            })
            print(f"[{case_index}/{len(cases)}] {case['repository']} / {case['case_id']}: equal")

        untraced_summary = _timing_summary(all_untraced)
        traced_summary = _timing_summary(all_traced)
        added_p95_ms = round(traced_summary["p95_ms"] - untraced_summary["p95_ms"], 4)
        p95_ratio = round(
            traced_summary["p95_ms"] / untraced_summary["p95_ms"],
            4,
        ) if untraced_summary["p95_ms"] else 0.0
        gates = {
            "all_results_equal": all(row["results_equal"] for row in query_rows),
            "all_attribution_complete": all(row["attribution_complete"] for row in query_rows),
            "added_p95_within_25ms": added_p95_ms <= MAX_ADDED_P95_MS,
            "p95_ratio_within_1_5x": p95_ratio <= MAX_P95_RATIO,
            "trace_size_within_250kib": max_trace_bytes <= MAX_TRACE_BYTES,
        }
        output = {
            "metadata": build_run_metadata(
                args.manifest,
                "development",
                revisions,
                command=sys.argv,
                seed=BASELINE_SEED,
                extra={
                    "tier": args.tier,
                    "query_cases": len(query_rows),
                    "repetitions": args.repetitions,
                    "warmups": args.warmups,
                    "tessera_revision": tessera_revision,
                    "embedding_model": BASELINE_MODEL,
                },
            ),
            "summary": {
                "untraced": untraced_summary,
                "traced": traced_summary,
                "paired_delta": _timing_summary(all_paired_deltas),
                "added_p95_ms": added_p95_ms,
                "p95_ratio": p95_ratio,
                "max_trace_bytes": max_trace_bytes,
                "gates": gates,
                "passed": all(gates.values()),
            },
            "queries": query_rows,
        }
        rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered)
        return 0 if output["summary"]["passed"] else 1
    finally:
        for database in databases.values():
            database.close()
        embedder.close()
        ProjectDB.base_dir = None


if __name__ == "__main__":
    raise SystemExit(main())
