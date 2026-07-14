"""Tests for corpus isolation, reproducibility metadata, and macro statistics."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
from scripts.benchmark_governance import (
    DEFAULT_MANIFEST,
    CorpusValidationError,
    build_run_metadata,
    load_manifest,
    load_queries,
    macro_per_repository,
    paired_bootstrap_interval,
    protected_segment_metrics,
    repositories_for_split,
    validate_labels_against_checkout,
    validate_manifest,
    verify_external_suite_revisions,
)


def _row(engine: str, repository: str, case_id: str, rank: int | None, **segments: str) -> dict:
    return {
        "engine": engine,
        "repository": repository,
        "case_id": case_id,
        "rank": rank,
        "supported": True,
        **segments,
    }


def test_public_manifest_separates_selection_and_holdout() -> None:
    manifest = load_manifest()
    legacy = repositories_for_split(manifest, "legacy_regression")
    development = repositories_for_split(manifest, "development")

    assert {repository["id"] for repository in legacy} == {"flask", "nextjs"}
    assert len(development) == 6
    assert all(repository["selection_allowed"] is False for repository in legacy)
    assert all(repository["selection_allowed"] is True for repository in development)
    assert manifest["sealed_holdout"]["public_labels"] is False
    assert "repositories" not in manifest["sealed_holdout"]
    assert "labels" not in manifest["sealed_holdout"]


def test_each_development_repository_has_code_document_and_cross_queries() -> None:
    manifest = load_manifest()
    for repository in repositories_for_split(manifest, "development"):
        queries = load_queries(manifest, repository["id"])
        assert len(queries) == 5
        assert {query["category"] for query in queries} == {"code", "document", "cross"}
        assert all(query["repository"] == repository["id"] for query in queries)


def test_manifest_rejects_repository_leakage_between_splits() -> None:
    manifest = copy.deepcopy(load_manifest())
    development = next(repository for repository in manifest["repositories"] if repository["split"] == "development")
    legacy = next(repository for repository in manifest["repositories"] if repository["split"] == "legacy_regression")
    development["repository_url"] = legacy["repository_url"]

    with pytest.raises(CorpusValidationError, match="crosses development/regression splits"):
        validate_manifest(manifest)


def test_manifest_rejects_public_holdout_identity_or_labels() -> None:
    manifest = copy.deepcopy(load_manifest())
    manifest["sealed_holdout"]["repositories"] = [{"id": "leaked"}]

    with pytest.raises(CorpusValidationError, match="must not expose"):
        validate_manifest(manifest)


def test_macro_metrics_weight_repositories_equally() -> None:
    rows = [
        _row("baseline", "small", "s1", 1),
        _row("baseline", "large", "l1", None),
        _row("baseline", "large", "l2", None),
        _row("baseline", "large", "l3", None),
    ]

    metrics = macro_per_repository(rows)["baseline"]
    assert metrics["repositories"] == 2
    assert metrics["queries_supported"] == 4
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["per_repository"]["small"]["mrr_at_10"] == 1.0
    assert metrics["per_repository"]["large"]["mrr_at_10"] == 0.0


def test_protected_segments_are_reported_per_engine() -> None:
    rows = [
        _row("candidate", "python-repo", "p1", 1, language="python", content_type="code"),
        _row("candidate", "docs-repo", "d1", 2, language="ruby", content_type="document"),
    ]

    segments = protected_segment_metrics(rows)
    assert segments["language"]["candidate/python"]["mrr_at_10"] == 1.0
    assert segments["language"]["candidate/ruby"]["mrr_at_10"] == 0.5
    assert segments["content_type"]["candidate/document"]["top_3"] == 1.0


def test_paired_bootstrap_is_deterministic_and_repository_level() -> None:
    rows = [
        _row("baseline", "repo-a", "a1", 4),
        _row("candidate", "repo-a", "a1", 2),
        _row("baseline", "repo-a", "a2", None),
        _row("candidate", "repo-a", "a2", 4),
        _row("baseline", "repo-b", "b1", 2),
        _row("candidate", "repo-b", "b1", 1),
    ]

    first = paired_bootstrap_interval(rows, "baseline", "candidate", iterations=500, seed=7)
    second = paired_bootstrap_interval(rows, "baseline", "candidate", iterations=500, seed=7)

    assert first == second
    assert first["repositories"] == 2
    assert first["paired_queries"] == 3
    assert first["delta"] > 0
    assert first["ci_lower"] > 0
    assert first["bootstrap_unit"] == "repository"


def test_paired_bootstrap_rejects_unpaired_cases() -> None:
    rows = [
        _row("baseline", "repo", "one", 1),
        _row("candidate", "repo", "different", 1),
    ]
    with pytest.raises(CorpusValidationError, match="different cases"):
        paired_bootstrap_interval(rows, "baseline", "candidate", iterations=10)


def test_run_metadata_binds_split_manifest_and_revisions() -> None:
    metadata = build_run_metadata(
        DEFAULT_MANIFEST,
        "development",
        {"repo-b": "b" * 40, "repo-a": "a" * 40},
        command=["benchmark", "--quick"],
        seed=42,
    )

    assert metadata["evaluation_split"] == "development"
    assert metadata["selection_allowed"] is True
    assert metadata["manifest_path"] == "benchmarks/corpora/v1/manifest.yaml"
    assert len(metadata["manifest_sha256"]) == 64
    assert list(metadata["repository_revisions"]) == ["repo-a", "repo-b"]
    assert metadata["command"] == ["benchmark", "--quick"]
    assert metadata["random_seed"] == 42


def test_label_validation_requires_category_compatible_tracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "implementation.py").write_text("def work(): pass\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\n", encoding="utf-8")
    subprocess.run(["git", "add", "implementation.py", "guide.md"], cwd=tmp_path, check=True)
    repository = {"id": "fixture", "revision": "f" * 40}

    validate_labels_against_checkout(
        repository,
        [
            {"case_id": "code", "category": "code", "expected_files": ["implementation.py"]},
            {"case_id": "doc", "category": "document", "expected_files": ["guide.md"]},
        ],
        tmp_path,
    )
    with pytest.raises(CorpusValidationError, match="wrong-category"):
        validate_labels_against_checkout(
            repository,
            [{"case_id": "wrong-category", "category": "code", "expected_files": ["guide.md"]}],
            tmp_path,
        )


def test_external_suite_requires_exact_clean_revision(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "source.txt"
    source.write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "external_suites": [
            {
                "id": "private-fixture",
                "repositories": [{"id": "source", "revision": revision}],
            }
        ]
    }

    assert verify_external_suite_revisions(manifest, "private-fixture", {"source": tmp_path}) == {
        "source": revision
    }
    source.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="uncommitted corpus changes"):
        verify_external_suite_revisions(manifest, "private-fixture", {"source": tmp_path})
