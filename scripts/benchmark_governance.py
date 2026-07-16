"""Versioned corpus validation and repository-level benchmark statistics.

This module contains evaluation-only infrastructure. It deliberately has no
dependency on Tessera's product search path so governance changes cannot alter
ranking behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "corpora" / "v1" / "manifest.yaml"
DEFAULT_CHECKOUT_ROOT = Path(os.path.expanduser("~/.tessera/benchmarks/corpora"))

PUBLIC_SPLITS = {"legacy_regression", "development"}
QUERY_CATEGORIES = {"code", "document", "cross"}
TIER_ORDER = {"quick": 0, "standard": 1, "full": 2}
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".php", ".go", ".swift", ".rb"}
DOCUMENT_EXTENSIONS = {
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".pdf",
    ".yaml",
    ".yml",
    ".json",
    ".html",
    ".htm",
    ".xml",
    ".toml",
}
REQUIRED_DEVELOPMENT_LANGUAGES = {"python", "typescript", "javascript", "php", "go", "swift", "ruby"}


class CorpusValidationError(ValueError):
    """Raised when an evaluation manifest or label set violates policy."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "" or value == []:
        raise CorpusValidationError(f"{context} is missing required field {key!r}")
    return value


def _resolve_public_path(value: str, context: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise CorpusValidationError(f"{context} points outside the public repository: {value}") from exc
    if not path.is_file():
        raise CorpusValidationError(f"{context} does not exist: {value}")
    return path


def manifest_digest(path: Path = DEFAULT_MANIFEST) -> str:
    """Return the SHA-256 digest used to bind a run to its exact manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_by_id(manifest: dict[str, Any], repository_id: str) -> dict[str, Any]:
    for repository in manifest["repositories"]:
        if repository["id"] == repository_id:
            return repository
    raise CorpusValidationError(f"Unknown repository {repository_id!r}")


def _load_python_queries(location: Path, tier: str) -> list[dict[str, Any]]:
    module_name = f"_tessera_benchmark_labels_{location.stem}"
    spec = importlib.util.spec_from_file_location(module_name, location)
    if spec is None or spec.loader is None:
        raise CorpusValidationError(f"Cannot import benchmark labels from {location}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_queries"):
        raise CorpusValidationError(f"Python label source has no get_queries(): {location}")
    rows = []
    for index, item in enumerate(module.get_queries(tier), start=1):
        if not isinstance(item, (tuple, list)) or len(item) != 5:
            raise CorpusValidationError(f"Invalid query tuple #{index} in {location}")
        query, expected_files, description, category, query_tier = item
        rows.append(
            {
                "id": f"{location.stem}-{index:03d}",
                "query": query,
                "expected_files": expected_files,
                "description": description,
                "category": "document" if category == "doc" else category,
                "tier": query_tier,
            }
        )
    return rows


def _load_yaml_queries(location: Path, selector: str, tier: str) -> list[dict[str, Any]]:
    document = yaml.safe_load(location.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise CorpusValidationError(f"Unsupported query schema in {location}")
    repositories = document.get("repositories")
    if not isinstance(repositories, dict) or selector not in repositories:
        raise CorpusValidationError(f"Query selector {selector!r} not found in {location}")
    maximum_tier = TIER_ORDER[tier]
    return [dict(row) for row in repositories[selector] if TIER_ORDER.get(row.get("tier", ""), 99) <= maximum_tier]


def load_queries(
    manifest: dict[str, Any],
    repository_id: str,
    tier: str = "full",
) -> list[dict[str, Any]]:
    """Load and normalize public labels for one repository."""
    if tier not in TIER_ORDER:
        raise CorpusValidationError(f"Unknown query tier {tier!r}")
    repository = _repository_by_id(manifest, repository_id)
    labels = repository["labels"]
    location = _resolve_public_path(labels["location"], f"labels for {repository_id}")
    if labels["format"] == "python":
        rows = _load_python_queries(location, tier)
    elif labels["format"] == "yaml":
        rows = _load_yaml_queries(location, labels.get("selector", repository_id), tier)
    else:
        raise CorpusValidationError(f"Unsupported label format {labels['format']!r} for {repository_id}")

    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "case_id": row["id"],
                "repository": repository_id,
                "language": repository["languages"][0],
                "content_type": {
                    "code": "code",
                    "document": "document",
                    "cross": "mixed",
                }.get(row.get("category"), row.get("category")),
            }
        )
    return normalized


def _validate_query_set(manifest: dict[str, Any], repository: dict[str, Any]) -> None:
    rows = load_queries(manifest, repository["id"], tier="full")
    if not rows:
        raise CorpusValidationError(f"Repository {repository['id']} has no query labels")
    seen: set[str] = set()
    categories: set[str] = set()
    for row in rows:
        context = f"query {repository['id']}/{row.get('id', '<unknown>')}"
        query_id = str(_require(row, "id", context))
        if query_id in seen:
            raise CorpusValidationError(f"Duplicate query id {query_id!r} in {repository['id']}")
        seen.add(query_id)
        _require(row, "query", context)
        _require(row, "description", context)
        expected_files = _require(row, "expected_files", context)
        if not isinstance(expected_files, list) or not all(isinstance(item, str) and item for item in expected_files):
            raise CorpusValidationError(f"{context} expected_files must be a non-empty string list")
        category = _require(row, "category", context)
        if category not in QUERY_CATEGORIES:
            raise CorpusValidationError(f"{context} has unsupported category {category!r}")
        categories.add(category)
        if row.get("tier") not in TIER_ORDER:
            raise CorpusValidationError(f"{context} has unsupported tier {row.get('tier')!r}")
    if repository["split"] == "development" and categories != QUERY_CATEGORIES:
        missing = ", ".join(sorted(QUERY_CATEGORIES - categories))
        raise CorpusValidationError(f"Development repository {repository['id']} lacks query categories: {missing}")


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate corpus identity, split isolation, label provenance, and coverage."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise CorpusValidationError("Manifest schema_version must be 1")
    _require(manifest, "manifest_id", "manifest")
    repositories = _require(manifest, "repositories", "manifest")
    if not isinstance(repositories, list):
        raise CorpusValidationError("manifest repositories must be a list")

    ids: set[str] = set()
    urls_by_split: dict[str, set[str]] = defaultdict(set)
    development_languages: set[str] = set()
    development_content: set[str] = set()
    for repository in repositories:
        repository_id = str(_require(repository, "id", "repository"))
        context = f"repository {repository_id}"
        if repository_id in ids:
            raise CorpusValidationError(f"Duplicate repository id {repository_id!r}")
        ids.add(repository_id)
        split = _require(repository, "split", context)
        if split not in PUBLIC_SPLITS:
            raise CorpusValidationError(f"Public {context} cannot use split {split!r}")
        url = str(_require(repository, "repository_url", context))
        if url in urls_by_split["legacy_regression" if split == "development" else "development"]:
            raise CorpusValidationError(f"Repository URL crosses development/regression splits: {url}")
        urls_by_split[split].add(url)
        revision = str(_require(repository, "revision", context))
        if REVISION_PATTERN.fullmatch(revision) is None:
            raise CorpusValidationError(f"{context} revision must be a full lowercase commit SHA")
        _require(repository, "license", context)
        languages = _require(repository, "languages", context)
        content_types = _require(repository, "content_types", context)
        _require(repository, "paths", context)
        labels = _require(repository, "labels", context)
        for key in ("format", "location", "provenance", "public"):
            _require(labels, key, f"labels for {repository_id}")
        if labels["public"] is not True:
            raise CorpusValidationError(f"Public {context} must use public labels")
        if bool(repository.get("selection_allowed")) != (split == "development"):
            raise CorpusValidationError(f"{context} selection_allowed contradicts split {split}")
        _resolve_public_path(labels["location"], f"labels for {repository_id}")
        _validate_query_set(manifest, repository)
        if split == "development":
            development_languages.update(languages)
            development_content.update(content_types)

    missing_languages = REQUIRED_DEVELOPMENT_LANGUAGES - development_languages
    if missing_languages:
        raise CorpusValidationError(
            "Development corpus is missing supported language families: " + ", ".join(sorted(missing_languages))
        )
    if "markdown" not in development_content:
        raise CorpusValidationError("Development corpus must include Markdown content")

    for suite in manifest.get("external_suites", []):
        context = f"external suite {suite.get('id', '<unknown>')}"
        if suite.get("split") != "legacy_regression" or suite.get("selection_allowed") is not False:
            raise CorpusValidationError(f"{context} must be non-selecting legacy_regression")
        component_ids: set[str] = set()
        for component in _require(suite, "repositories", context):
            component_id = str(_require(component, "id", context))
            if component_id in component_ids:
                raise CorpusValidationError(f"{context} has duplicate component id {component_id!r}")
            component_ids.add(component_id)
            _require(component, "repository_url", context)
            revision = str(_require(component, "revision", context))
            if REVISION_PATTERN.fullmatch(revision) is None:
                raise CorpusValidationError(f"{context} revision must be a full lowercase commit SHA")
            _require(component, "license", context)

    holdout = _require(manifest, "sealed_holdout", "manifest")
    if holdout.get("split") != "sealed_holdout":
        raise CorpusValidationError("sealed_holdout split must be sealed_holdout")
    if holdout.get("public_labels") is not False or holdout.get("selection_allowed") is not False:
        raise CorpusValidationError("Sealed holdout labels and selection must remain inaccessible")
    if "repositories" in holdout or "labels" in holdout:
        raise CorpusValidationError("Public manifest must not expose sealed holdout repositories or labels")
    _require(holdout, "manifest_path_env", "sealed_holdout")
    _require(holdout, "rotation_policy", "sealed_holdout")

    return manifest


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Read and validate a public evaluation manifest."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate_manifest(document)


def repositories_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Return public repositories in a declared evaluation split."""
    if split not in PUBLIC_SPLITS:
        raise CorpusValidationError(f"Public runner cannot load split {split!r}")
    return [repository for repository in manifest["repositories"] if repository["split"] == split]


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.stdout.strip()


def checkout_repository(repository: dict[str, Any], checkout_root: Path = DEFAULT_CHECKOUT_ROOT) -> Path:
    """Create a sparse checkout at exactly the manifest revision."""
    checkout_root.mkdir(parents=True, exist_ok=True)
    target = checkout_root / repository["id"]
    if target.exists() and not (target / ".git").is_dir():
        raise CorpusValidationError(f"Checkout target exists but is not a git repository: {target}")
    if not target.exists():
        _run_git(["clone", "--filter=blob:none", "--no-checkout", repository["repository_url"], str(target)])
    remote_url = _run_git(["remote", "get-url", "origin"], cwd=target)
    normalized_remote = remote_url.removesuffix(".git")
    normalized_expected = repository["repository_url"].removesuffix(".git")
    if normalized_remote != normalized_expected:
        raise CorpusValidationError(f"Checkout {target} has unexpected origin {remote_url}")

    _run_git(["sparse-checkout", "init", "--cone"], cwd=target)
    # Cone mode normally accepts directories only. ``--skip-checks`` also lets
    # manifests retain root files such as README.md without switching to the
    # more error-prone gitignore-pattern mode.
    _run_git(["sparse-checkout", "set", "--skip-checks", *repository["paths"]], cwd=target)
    revision = repository["revision"]
    try:
        _run_git(["cat-file", "-e", f"{revision}^{{commit}}"], cwd=target)
    except subprocess.CalledProcessError:
        _run_git(["fetch", "--depth", "1", "origin", revision], cwd=target)
    _run_git(["checkout", "--detach", "--force", revision], cwd=target)
    actual = _run_git(["rev-parse", "HEAD"], cwd=target)
    if actual != revision:
        raise CorpusValidationError(f"Checkout {repository['id']} resolved {actual}, expected {revision}")
    _write_corpus_scope_ignore(repository, target)
    return target


def _write_corpus_scope_ignore(repository: dict[str, Any], checkout: Path) -> None:
    """Limit indexing to manifest-declared paths despite cone-mode root files."""
    ignore_path = checkout / ".tesseraignore"
    declared_paths = repository["paths"]
    if declared_paths == ["."]:
        if ignore_path.is_file() and ignore_path.read_text(encoding="utf-8").startswith(
            "# Generated by Tessera benchmark governance"
        ):
            ignore_path.unlink()
        return

    patterns = ["# Generated by Tessera benchmark governance; do not edit.", "*"]
    for value in declared_paths:
        normalized = value.strip("/")
        path = checkout / normalized
        if path.is_dir():
            patterns.extend((f"!{normalized}/", f"!{normalized}/**"))
        else:
            patterns.append(f"!{normalized}")
    ignore_path.write_text("\n".join(patterns) + "\n", encoding="utf-8")


def _matches_expected(path: str, expected_files: list[str]) -> bool:
    normalized = path.replace("\\", "/").casefold()
    return any(expected.replace("\\", "/").casefold() in normalized for expected in expected_files)


def validate_labels_against_checkout(
    repository: dict[str, Any],
    queries: list[dict[str, Any]],
    checkout: Path,
) -> None:
    """Verify that every expected label resolves to a compatible indexed file."""
    declared_paths = repository.get("paths", ["."])
    tracked = [
        path
        for path in _run_git(["ls-files"], cwd=checkout).splitlines()
        if any(
            declared == "." or path == declared.strip("/") or path.startswith(f"{declared.strip('/')}" + "/")
            for declared in declared_paths
        )
    ]
    invalid: list[str] = []
    for query in queries:
        candidates = tracked
        if query["category"] == "code":
            candidates = [path for path in tracked if Path(path).suffix.casefold() in CODE_EXTENSIONS]
        elif query["category"] == "document":
            candidates = [path for path in tracked if Path(path).suffix.casefold() in DOCUMENT_EXTENSIONS]
        if not any(_matches_expected(path, query["expected_files"]) for path in candidates):
            invalid.append(f"{query['case_id']}: {query['expected_files']}")
    if invalid:
        details = "\n".join(f"- {item}" for item in invalid)
        raise CorpusValidationError(
            f"Labels do not resolve in {repository['id']} at {repository['revision']}:\n{details}"
        )


def validate_repositories(
    manifest: dict[str, Any],
    checkout_root: Path = DEFAULT_CHECKOUT_ROOT,
) -> dict[str, str]:
    """Checkout all public repositories and validate their ground truth."""
    revisions: dict[str, str] = {}
    for repository in manifest["repositories"]:
        checkout = checkout_repository(repository, checkout_root)
        validate_labels_against_checkout(repository, load_queries(manifest, repository["id"]), checkout)
        revisions[repository["id"]] = _run_git(["rev-parse", "HEAD"], cwd=checkout)
    return revisions


def verify_external_suite_revisions(
    manifest: dict[str, Any],
    suite_id: str,
    paths: dict[str, Path],
) -> dict[str, str]:
    """Require an authorized legacy suite to match its pinned, clean snapshots."""
    suite = next((item for item in manifest.get("external_suites", []) if item["id"] == suite_id), None)
    if suite is None:
        raise CorpusValidationError(f"Unknown external suite {suite_id!r}")
    components = {component["id"]: component for component in suite["repositories"]}
    if paths.keys() != components.keys():
        raise CorpusValidationError(
            f"External suite {suite_id} paths must cover {sorted(components)}, got {sorted(paths)}"
        )
    revisions: dict[str, str] = {}
    for component_id, path in paths.items():
        if not path.is_dir():
            raise CorpusValidationError(f"External suite path does not exist: {path}")
        actual = _run_git(["rev-parse", "HEAD"], cwd=path)
        expected = components[component_id]["revision"]
        if actual != expected:
            raise CorpusValidationError(
                f"{suite_id}/{component_id} is at {actual}; manifest requires {expected}"
            )
        dirty = _run_git(["status", "--porcelain", "--", "."], cwd=path)
        if dirty:
            raise CorpusValidationError(
                f"{suite_id}/{component_id} has uncommitted corpus changes; use a clean pinned checkout"
            )
        revisions[component_id] = actual
    return revisions


def _rank(row: dict[str, Any]) -> int | None:
    value = row.get("rank")
    return int(value) if value is not None else None


def retrieval_metrics(rows: list[dict[str, Any]], k: int = 10) -> dict[str, Any]:
    """Compute file-level retrieval metrics for supported rows."""
    supported = [row for row in rows if row.get("supported", True)]
    count = len(supported)
    ranks = [_rank(row) for row in supported]
    return {
        "queries_supported": count,
        "mrr_at_10": round(sum(1.0 / rank for rank in ranks if rank is not None and rank <= k) / count, 6)
        if count
        else 0.0,
        "top_1": round(sum(rank == 1 for rank in ranks) / count, 6) if count else 0.0,
        "top_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / count, 6) if count else 0.0,
        "top_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / count, 6) if count else 0.0,
        "top_10": round(sum(rank is not None and rank <= k for rank in ranks) / count, 6) if count else 0.0,
    }


def macro_per_repository(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate each engine by repository first, then weight repositories equally."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["engine"], row["repository"])].append(row)
    by_engine: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (engine, repository), segment in sorted(grouped.items()):
        metrics = retrieval_metrics(segment)
        if metrics["queries_supported"]:
            by_engine[engine][repository] = metrics

    output: dict[str, Any] = {}
    metric_names = ("mrr_at_10", "top_1", "top_3", "top_5", "top_10")
    for engine, repositories in sorted(by_engine.items()):
        output[engine] = {
            "repositories": len(repositories),
            "queries_supported": sum(item["queries_supported"] for item in repositories.values()),
            **{
                metric: round(fmean(item[metric] for item in repositories.values()), 6)
                for metric in metric_names
            },
            "per_repository": repositories,
        }
    return output


def protected_segment_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report language and content-type guardrails independently per engine."""
    result: dict[str, Any] = {}
    for dimension in ("language", "content_type"):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(dimension)
            if dimension == "content_type" and not value:
                value = row.get("category")
            if value:
                grouped[(row["engine"], str(value))].append(row)
        result[dimension] = {
            f"{engine}/{value}": retrieval_metrics(segment)
            for (engine, value), segment in sorted(grouped.items())
        }
    return result


def _case_key(row: dict[str, Any]) -> tuple[str, str]:
    case_id = row.get("case_id") or row.get("id") or row.get("description") or row.get("query")
    if not case_id:
        raise CorpusValidationError("Paired rows require case_id, id, description, or query")
    return str(row["repository"]), str(case_id)


def _reciprocal_rank(row: dict[str, Any], k: int = 10) -> float:
    rank = _rank(row)
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def paired_bootstrap_interval(
    rows: list[dict[str, Any]],
    baseline_engine: str,
    treatment_engine: str,
    *,
    iterations: int = 10_000,
    seed: int = 1729,
) -> dict[str, Any]:
    """Return a paired 95% interval for the macro per-repository MRR delta.

    Query cases are paired first. Each repository contributes one mean delta,
    and the bootstrap resamples those repository deltas with replacement.
    """
    if iterations < 1:
        raise CorpusValidationError("Bootstrap iterations must be positive")
    baseline = {
        _case_key(row): row
        for row in rows
        if row.get("engine") == baseline_engine and row.get("supported", True)
    }
    treatment = {
        _case_key(row): row
        for row in rows
        if row.get("engine") == treatment_engine and row.get("supported", True)
    }
    if baseline.keys() != treatment.keys():
        missing_treatment = sorted(baseline.keys() - treatment.keys())
        missing_baseline = sorted(treatment.keys() - baseline.keys())
        raise CorpusValidationError(
            f"Paired engines cover different cases; missing treatment={missing_treatment}, "
            f"missing baseline={missing_baseline}"
        )
    if not baseline:
        raise CorpusValidationError("Paired comparison has no shared supported cases")

    by_repository: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key in sorted(baseline):
        repository = key[0]
        by_repository[repository].append(
            (_reciprocal_rank(baseline[key]), _reciprocal_rank(treatment[key]))
        )
    baseline_repo = {
        repository: fmean(pair[0] for pair in pairs) for repository, pairs in by_repository.items()
    }
    treatment_repo = {
        repository: fmean(pair[1] for pair in pairs) for repository, pairs in by_repository.items()
    }
    repository_deltas = [treatment_repo[name] - baseline_repo[name] for name in sorted(by_repository)]
    rng = random.Random(seed)
    sample_size = len(repository_deltas)
    samples = [
        fmean(repository_deltas[rng.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(iterations)
    ]
    return {
        "metric": "macro_mrr_at_10_delta",
        "baseline_engine": baseline_engine,
        "treatment_engine": treatment_engine,
        "repositories": sample_size,
        "paired_queries": len(baseline),
        "baseline": round(fmean(baseline_repo.values()), 6),
        "treatment": round(fmean(treatment_repo.values()), 6),
        "delta": round(fmean(repository_deltas), 6),
        "confidence": 0.95,
        "ci_lower": round(_percentile(samples, 0.025), 6),
        "ci_upper": round(_percentile(samples, 0.975), 6),
        "bootstrap_unit": "repository",
        "iterations": iterations,
        "seed": seed,
    }


def paired_candidate_recall_interval(
    rows: list[dict[str, Any]],
    baseline_engine: str,
    treatment_engine: str,
    *,
    cutoff: int,
    candidate_set: str = "union",
    iterations: int = 10_000,
    seed: int = 1729,
) -> dict[str, Any]:
    """Return a paired repository-bootstrap interval for candidate Recall@K."""
    if cutoff < 1:
        raise CorpusValidationError("Candidate recall cutoff must be positive")
    if iterations < 1:
        raise CorpusValidationError("Bootstrap iterations must be positive")

    baseline = {
        _case_key(row): row
        for row in rows
        if row.get("engine") == baseline_engine and row.get("supported", True)
    }
    treatment = {
        _case_key(row): row
        for row in rows
        if row.get("engine") == treatment_engine and row.get("supported", True)
    }
    if baseline.keys() != treatment.keys():
        missing_treatment = sorted(baseline.keys() - treatment.keys())
        missing_baseline = sorted(treatment.keys() - baseline.keys())
        raise CorpusValidationError(
            f"Paired engines cover different cases; missing treatment={missing_treatment}, "
            f"missing baseline={missing_baseline}"
        )
    if not baseline:
        raise CorpusValidationError("Paired comparison has no shared supported cases")

    def recall(row: dict[str, Any]) -> float:
        diagnostics = row.get("candidate_diagnostics") or {}
        if candidate_set == "union":
            candidate = diagnostics.get("union") or {}
        else:
            candidate = (diagnostics.get("channels") or {}).get(candidate_set) or {}
        value = (candidate.get("recall_at_k") or {}).get(str(cutoff))
        if value is None:
            raise CorpusValidationError(
                f"Missing {candidate_set} candidate Recall@{cutoff} for {_case_key(row)}"
            )
        return float(value)

    by_repository: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key in sorted(baseline):
        by_repository[key[0]].append((recall(baseline[key]), recall(treatment[key])))
    baseline_repo = {
        repository: fmean(pair[0] for pair in pairs)
        for repository, pairs in by_repository.items()
    }
    treatment_repo = {
        repository: fmean(pair[1] for pair in pairs)
        for repository, pairs in by_repository.items()
    }
    repository_deltas = [
        treatment_repo[name] - baseline_repo[name]
        for name in sorted(by_repository)
    ]
    rng = random.Random(seed)
    sample_size = len(repository_deltas)
    samples = [
        fmean(repository_deltas[rng.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(iterations)
    ]
    return {
        "metric": f"macro_{candidate_set}_candidate_recall_at_{cutoff}_delta",
        "baseline_engine": baseline_engine,
        "treatment_engine": treatment_engine,
        "repositories": sample_size,
        "paired_queries": len(baseline),
        "baseline": round(fmean(baseline_repo.values()), 6),
        "treatment": round(fmean(treatment_repo.values()), 6),
        "delta": round(fmean(repository_deltas), 6),
        "confidence": 0.95,
        "ci_lower": round(_percentile(samples, 0.025), 6),
        "ci_upper": round(_percentile(samples, 0.975), 6),
        "bootstrap_unit": "repository",
        "iterations": iterations,
        "seed": seed,
    }


def build_run_metadata(
    manifest_path: Path,
    split: str,
    repository_revisions: dict[str, str],
    *,
    command: list[str] | None = None,
    seed: int = 1729,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the required machine-readable provenance block for an artifact."""
    if split not in {"legacy_regression", "development", "sealed_holdout"}:
        raise CorpusValidationError(f"Unknown evaluation split {split!r}")
    resolved_manifest = manifest_path.resolve()
    try:
        recorded_manifest_path = resolved_manifest.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        recorded_manifest_path = str(resolved_manifest)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_split": split,
        "selection_allowed": split == "development",
        "manifest_schema_version": 1,
        "manifest_path": recorded_manifest_path,
        "manifest_sha256": manifest_digest(manifest_path),
        "repository_revisions": dict(sorted(repository_revisions.items())),
        "command": command if command is not None else sys.argv,
        "random_seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-repositories", action="store_true")
    parser.add_argument("--checkout-root", type=Path, default=DEFAULT_CHECKOUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = load_manifest(args.manifest)
    result: dict[str, Any] = {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_digest(args.manifest),
        "repositories": len(manifest["repositories"]),
        "legacy_regression": len(repositories_for_split(manifest, "legacy_regression")),
        "development": len(repositories_for_split(manifest, "development")),
        "sealed_holdout_exposed": False,
    }
    if args.verify_repositories:
        result["verified_revisions"] = validate_repositories(manifest, args.checkout_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
