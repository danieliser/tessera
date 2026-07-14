"""Run the predeclared PR-03 two-stage retrieval comparison.

This is a paired development-corpus experiment. It compares current
single-page reranking with the predeclared expanded, coverage-first, and
structured-input stages without selecting among pool sizes or reranker models.

Example:
    uv run python scripts/benchmark_two_stage.py \
      --tier quick \
      --output benchmarks/development-two-stage-2026-07-14.json
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
    protected_segment_metrics,
    repositories_for_split,
    retrieval_metrics,
    validate_labels_against_checkout,
)

from tessera.db import ProjectDB  # noqa: E402
from tessera.embeddings import FastembedClient, FastembedReranker  # noqa: E402
from tessera.rerank import (  # noqa: E402
    MAX_RERANK_DOCUMENT_CHARS,
    MAX_RERANK_FETCH_SIZE,
    MAX_RERANK_POOL_SIZE,
    RERANK_FETCH_MULTIPLIER,
    RERANK_POOL_MULTIPLIER,
    build_rerank_documents,
    rerank_candidate_budget,
    rerank_retrieval_budget,
    select_rerank_candidates,
)
from tessera.search import hybrid_search  # noqa: E402

CURRENT_ENGINE = "current_page_content"
EXPANDED_ENGINE = "expanded_score_order_content"
DIVERSE_ENGINE = "expanded_coverage_first_content"
TREATMENT_ENGINE = "two_stage_structured"
ENGINES = [CURRENT_ENGINE, EXPANDED_ENGINE, DIVERSE_ENGINE, TREATMENT_ENGINE]
DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


def _candidate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    known_paths = [str(result.get("file_path", "")) for result in results if result.get("file_path")]
    unique_files = len(set(known_paths))
    duplicates = len(known_paths) - unique_files
    return {
        "candidates": len(results),
        "known_files": len(known_paths),
        "unique_files": unique_files,
        "duplicate_file_rate": round(duplicates / len(known_paths), 6) if known_paths else 0.0,
    }


def _run_case(
    case: dict[str, Any],
    database: ProjectDB,
    embedder: FastembedClient,
    reranker: FastembedReranker,
    engine: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    embedding = np.asarray(embedder.embed_query(case["query"]), dtype=np.float32)
    candidate_budget = rerank_candidate_budget(TOP_K, reranker_active=True)
    retrieval_budget = rerank_retrieval_budget(candidate_budget, reranker_active=True)
    requested = TOP_K if engine == CURRENT_ENGINE else retrieval_budget
    raw = hybrid_search(
        case["query"],
        embedding,
        database,
        graph=None,
        limit=requested,
        source_type=_source_filter(case["category"]),
    )

    if engine == CURRENT_ENGINE:
        pool = raw[:TOP_K]
    elif engine == EXPANDED_ENGINE:
        pool = raw[:candidate_budget]
    else:
        pool = select_rerank_candidates(raw, candidate_budget)

    if engine == TREATMENT_ENGINE:
        documents = build_rerank_documents(pool, {None: database})
    else:
        documents = [str(candidate.get("content") or candidate.get("snippet") or "") for candidate in pool]

    reranked = reranker.rerank(case["query"], documents, top_k=TOP_K)
    hits = [pool[index] for index, _score in reranked if 0 <= index < len(pool)]
    paths = _dedupe_paths(hits)[:TOP_K]
    rank = rank_expected(paths, case["expected_files"])
    return {
        **case,
        "engine": engine,
        "supported": True,
        "rank": rank,
        "reciprocal_rank": round(1.0 / rank, 6) if rank else 0.0,
        "top_files": paths,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "requested_retrieval_budget": requested,
        "candidate_budget": len(pool),
        "raw_candidates": _candidate_metrics(raw),
        "rerank_pool": _candidate_metrics(pool),
        "max_document_chars": max((len(document) for document in documents), default=0),
    }


def _candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for engine in ENGINES:
        selected = [row for row in rows if row["engine"] == engine]
        result[engine] = {
            "mean_raw_candidates": round(fmean(row["raw_candidates"]["candidates"] for row in selected), 6),
            "mean_pool_candidates": round(fmean(row["rerank_pool"]["candidates"] for row in selected), 6),
            "mean_pool_unique_files": round(fmean(row["rerank_pool"]["unique_files"] for row in selected), 6),
            "mean_pool_duplicate_file_rate": round(
                fmean(row["rerank_pool"]["duplicate_file_rate"] for row in selected),
                6,
            ),
            "max_document_chars": max(row["max_document_chars"] for row in selected),
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repository", action="append", help="Development repository id; repeat as needed")
    parser.add_argument("--tier", choices=["quick", "standard", "full"], default="quick")
    parser.add_argument("--checkout-root", type=Path, default=DEFAULT_CHECKOUT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = load_manifest(args.manifest)
    available = {repository["id"]: repository for repository in repositories_for_split(manifest, "development")}
    selected_ids = args.repository or list(available)
    unknown = sorted(set(selected_ids) - available.keys())
    if unknown:
        raise SystemExit(f"Unknown development repositories: {', '.join(unknown)}")
    if _tracked_worktree_changes(REPO_ROOT):
        raise SystemExit(
            "Two-stage benchmarks require a clean tracked Tessera worktree so the revision identifies the code exactly"
        )

    tessera_revision = _git_revision(REPO_ROOT)
    corpus_digest = manifest_digest(args.manifest)
    embedder = FastembedClient(model_name=BASELINE_MODEL)
    reranker = FastembedReranker(model_name=args.reranker)
    _ = embedder.embed_query("two-stage benchmark warmup")
    _ = reranker.rerank("two-stage benchmark warmup", ["warmup document"], top_k=1)

    revisions: dict[str, str] = {}
    repository_metadata: dict[str, Any] = {}
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
            databases[repository_id] = database
            repository_metadata[repository_id] = {
                "repository_url": repository["repository_url"],
                "revision": revision,
                "license": repository["license"],
                "languages": repository["languages"],
                "content_types": repository["content_types"],
                "index": index_metadata,
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
                    reranker,
                    engine,
                )
                by_engine[engine] = result
                completed += 1
                rank = result["rank"] if result["rank"] is not None else "MISS"
                print(f"[{completed}/{total}] {case['repository']} / {case['case_id']} / {engine}: {rank}")
            rows.extend(by_engine[engine] for engine in ENGINES)

        per_repository: dict[str, dict[str, Any]] = {}
        for repository_id in selected_ids:
            per_repository[repository_id] = {
                engine: retrieval_metrics(
                    [row for row in rows if row["repository"] == repository_id and row["engine"] == engine]
                )
                for engine in ENGINES
            }
        latency = {engine: _latency_metrics([row for row in rows if row["engine"] == engine]) for engine in ENGINES}
        macro = macro_per_repository(rows)
        paired = {
            engine: paired_bootstrap_interval(
                rows,
                CURRENT_ENGINE,
                engine,
                seed=BASELINE_SEED,
            )
            for engine in ENGINES[1:]
        }
        baseline_mrr = macro[CURRENT_ENGINE]["mrr_at_10"]
        treatment_mrr = macro[TREATMENT_ENGINE]["mrr_at_10"]
        treatment_p95 = latency[TREATMENT_ENGINE]["p95_ms"]
        baseline_p95 = latency[CURRENT_ENGINE]["p95_ms"]
        gates = {
            "macro_mrr_non_negative": treatment_mrr >= baseline_mrr,
            "treatment_p95_at_most_500_ms": treatment_p95 <= 500.0,
            "added_p95_at_most_300_ms": treatment_p95 - baseline_p95 <= 300.0,
            "max_document_chars_enforced": all(
                row["max_document_chars"] <= MAX_RERANK_DOCUMENT_CHARS
                for row in rows
                if row["engine"] == TREATMENT_ENGINE
            ),
        }
        gates["passed"] = all(gates.values())

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
                "reranking_model": args.reranker,
                "engine_order_rotated": True,
                "pool_multiplier": RERANK_POOL_MULTIPLIER,
                "max_pool_size": MAX_RERANK_POOL_SIZE,
                "raw_fetch_multiplier": RERANK_FETCH_MULTIPLIER,
                "max_raw_fetch_size": MAX_RERANK_FETCH_SIZE,
                "max_document_chars": MAX_RERANK_DOCUMENT_CHARS,
            },
        )
        output = {
            "metadata": metadata,
            "repositories": repository_metadata,
            "summary": {
                "primary_macro_per_repository": macro,
                "pooled": {
                    engine: retrieval_metrics([row for row in rows if row["engine"] == engine]) for engine in ENGINES
                },
                "per_repository": per_repository,
                "protected_segments": protected_segment_metrics(rows),
                "latency": latency,
                "candidate_pool": _candidate_summary(rows),
                "paired_macro_mrr": paired,
                "predeclared_gates": gates,
            },
            "queries": rows,
            "limitations": [
                "This is a public development treatment, not a sealed holdout result.",
                "File-level labels do not score snippet usefulness, answer synthesis, or agent task completion.",
                "The four stages diagnose one predeclared mechanism; they are not a model or pool-size sweep.",
                "Segment samples are small and are reported as guardrails rather than tuning targets.",
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
        reranker.close()
        embedder.close()
        ProjectDB.base_dir = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
