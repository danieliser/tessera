"""Run the repository-separated Tessera development baseline.

The development corpus is public and may support predeclared experiments, but
this command itself runs one frozen baseline configuration. It emits the
manifest digest, exact repository revisions, macro-per-repository metrics, and
protected language/content segments required by the retrieval quality program.

Example:
    uv run python scripts/benchmark_development.py \
      --tier quick --output benchmarks/development-baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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

from benchmark_governance import (  # noqa: E402
    DEFAULT_CHECKOUT_ROOT,
    DEFAULT_MANIFEST,
    build_run_metadata,
    checkout_repository,
    load_manifest,
    load_queries,
    macro_per_repository,
    manifest_digest,
    protected_segment_metrics,
    repositories_for_split,
    retrieval_metrics,
    validate_labels_against_checkout,
)

from tessera.db import ProjectDB  # noqa: E402
from tessera.embeddings import FastembedClient  # noqa: E402
from tessera.indexer import IndexerPipeline  # noqa: E402
from tessera.search import hybrid_search  # noqa: E402

TOP_K = 10
BASELINE_ENGINE = "tessera_bge_small_hybrid"
BASELINE_MODEL = "BAAI/bge-small-en-v1.5"
BASELINE_SEED = 1729
DEFAULT_INDEX_ROOT = Path(os.path.expanduser("~/.tessera/benchmarks/development-indexes"))
DOCUMENT_SOURCE_TYPES = [
    "markdown",
    "pdf",
    "yaml",
    "json",
    "html",
    "xml",
    "text",
    "txt",
    "rst",
    "csv",
    "tsv",
    "log",
    "ini",
    "cfg",
    "toml",
    "conf",
]


def _source_filter(category: str) -> list[str] | None:
    if category == "code":
        return ["code"]
    if category == "document":
        return DOCUMENT_SOURCE_TYPES
    return None


def _matches_expected(path: str, expected_files: list[str]) -> bool:
    normalized = path.replace("\\", "/").casefold()
    return any(expected.replace("\\", "/").casefold() in normalized for expected in expected_files)


def rank_expected(paths: list[str], expected_files: list[str], k: int = TOP_K) -> int | None:
    """Return the first one-based expected-file rank."""
    for index, path in enumerate(paths[:k], start=1):
        if _matches_expected(path, expected_files):
            return index
    return None


def _dedupe_paths(hits: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for hit in hits:
        path = str(hit.get("file_path", ""))
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_worktree_changes(path: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _index_repository(
    repository: dict[str, Any],
    checkout: Path,
    embedder: FastembedClient,
    index_root: Path,
    tessera_revision: str,
    corpus_digest: str,
    *,
    reindex: bool,
) -> tuple[ProjectDB, dict[str, Any]]:
    index_dir = index_root / corpus_digest[:12] / tessera_revision[:12] / "bge-small" / repository["id"]
    if reindex and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    ProjectDB.base_dir = str(index_dir)
    database = ProjectDB(str(checkout))
    stamp_path = index_dir / ".indexed.json"
    expected_stamp = {
        "manifest_sha256": corpus_digest,
        "repository_revision": repository["revision"],
        "tessera_revision": tessera_revision,
        "embedding_model": BASELINE_MODEL,
    }
    if stamp_path.is_file() and json.loads(stamp_path.read_text(encoding="utf-8")) == expected_stamp:
        return database, {"cached": True}

    started = time.perf_counter()
    stats = IndexerPipeline(
        project_path=str(checkout),
        project_db=database,
        embedding_client=embedder,
        languages=repository["languages"],
    ).index_project_sync()
    stamp_path.write_text(json.dumps(expected_stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return database, {
        "cached": False,
        "index_ms": round((time.perf_counter() - started) * 1000, 2),
        "files_processed": stats.files_processed,
        "chunks_created": stats.chunks_created,
    }


def _run_case(
    case: dict[str, Any],
    database: ProjectDB,
    embedder: FastembedClient,
) -> dict[str, Any]:
    started = time.perf_counter()
    embedding = np.asarray(embedder.embed_query(case["query"]), dtype=np.float32)
    hits = hybrid_search(
        case["query"],
        embedding,
        database,
        graph=None,
        limit=TOP_K,
        source_type=_source_filter(case["category"]),
        file_dedup=True,
    )
    paths = _dedupe_paths(hits)[:TOP_K]
    rank = rank_expected(paths, case["expected_files"])
    return {
        **case,
        "engine": BASELINE_ENGINE,
        "supported": True,
        "rank": rank,
        "reciprocal_rank": round(1.0 / rank, 6) if rank else 0.0,
        "top_files": paths,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _latency_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = sorted(float(row["latency_ms"]) for row in rows)
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    def percentile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "mean_ms": round(sum(values) / len(values), 2),
        "p50_ms": round(percentile(0.5), 2),
        "p95_ms": round(percentile(0.95), 2),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repository", action="append", help="Development repository id; repeat as needed")
    parser.add_argument("--tier", choices=["quick", "standard", "full"], default="quick")
    parser.add_argument("--checkout-root", type=Path, default=DEFAULT_CHECKOUT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--reindex", action="store_true")
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
            "Development baselines require a clean tracked Tessera worktree so tessera_revision identifies the code exactly"
        )
    tessera_revision = _git_revision(REPO_ROOT)
    corpus_digest = manifest_digest(args.manifest)
    embedder = FastembedClient(model_name=BASELINE_MODEL)
    _ = embedder.embed_query("development benchmark warmup")

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
        for index, case in enumerate(cases, start=1):
            result = _run_case(case, databases[case["repository"]], embedder)
            rows.append(result)
            rank = result["rank"] if result["rank"] is not None else "MISS"
            print(f"[{index}/{len(cases)}] {case['repository']} / {case['case_id']}: {rank}")

        by_repository: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_repository[row["repository"]].append(row)
        metadata = build_run_metadata(
            args.manifest,
            "development",
            revisions,
            command=sys.argv,
            seed=BASELINE_SEED,
            extra={
                "tier": args.tier,
                "query_cases": len(rows),
                "top_k": TOP_K,
                "tessera_revision": tessera_revision,
                "embedding_model": BASELINE_MODEL,
                "engine": BASELINE_ENGINE,
            },
        )
        output = {
            "metadata": metadata,
            "repositories": repository_metadata,
            "summary": {
                "primary_macro_per_repository": macro_per_repository(rows),
                "pooled": {BASELINE_ENGINE: retrieval_metrics(rows)},
                "per_repository": {
                    repository: retrieval_metrics(repository_rows)
                    for repository, repository_rows in sorted(by_repository.items())
                },
                "protected_segments": protected_segment_metrics(rows),
                "latency": _latency_metrics(rows),
            },
            "queries": rows,
            "limitations": [
                "This is a public development baseline, not a sealed holdout result.",
                "File-level labels do not score snippet usefulness, answer synthesis, or agent task completion.",
                "Cold indexing and warm query latency are recorded separately and should not be combined.",
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
