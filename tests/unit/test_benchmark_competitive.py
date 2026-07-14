"""Tests for competitive benchmark scoring and output adapters."""

from pathlib import Path

import pytest
from scripts.benchmark_competitive import (
    parse_codegraph_files,
    rank_expected,
    ranker_routing_analysis,
    summarize,
    validate_ground_truth,
)


def test_parse_codegraph_files_preserves_unique_source_order() -> None:
    output = """
**Source Code**
**`src/first.py`** — symbols
**`src/second.ts`** — symbols
**`src/first.py`** — duplicate
"""
    assert parse_codegraph_files(output) == ["src/first.py", "src/second.ts"]


def test_rank_expected_accepts_full_or_partial_expected_paths() -> None:
    paths = ["src/noise.py", "packages/core/target.ts"]
    assert rank_expected(paths, ["core/target.ts"]) == 2
    assert rank_expected(paths, ["missing.ts"]) is None


def test_summary_excludes_unsupported_rows_from_denominator() -> None:
    rows = [
        {"supported": True, "rank": 1, "latency_ms": 10.0},
        {"supported": True, "rank": None, "latency_ms": 20.0},
        {"supported": False, "rank": None, "latency_ms": None},
    ]
    result = summarize(rows)
    assert result["queries_total"] == 3
    assert result["queries_supported"] == 2
    assert result["queries_unsupported"] == 1
    assert result["mrr_at_10"] == 0.5
    assert result["top_10"] == 0.5


def test_ground_truth_requires_category_compatible_file(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("guide", encoding="utf-8")
    case = {
        "repository": "fixture",
        "category": "code",
        "description": "Mislabeled code query",
        "expected_files": ["guide.md"],
    }
    with pytest.raises(ValueError, match="Mislabeled code query"):
        validate_ground_truth([case], {"fixture": tmp_path})


def test_ground_truth_accepts_markdown_document(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("guide", encoding="utf-8")
    case = {
        "repository": "fixture",
        "category": "document",
        "description": "Markdown query",
        "expected_files": ["README.md"],
    }
    validate_ground_truth([case], {"fixture": tmp_path})


def test_ranker_routing_uses_jina_only_for_code() -> None:
    rows = [
        {
            "engine": engine,
            "category": category,
            "supported": True,
            "rank": rank,
            "latency_ms": 1.0,
        }
        for category, base_rank, reranked_rank in (
            ("code", 4, 1),
            ("document", 1, 4),
            ("cross", 2, 5),
        )
        for engine, rank in (
            ("tessera_bge_small", base_rank),
            ("tessera_bge_small_jina_tiny", reranked_rank),
        )
    ]
    result = ranker_routing_analysis(rows)
    assert result["post_hoc"] is True
    assert result["overall"]["queries_supported"] == 3
    assert result["categories"]["code"]["mrr_at_10"] == 1.0
    assert result["categories"]["document"]["mrr_at_10"] == 1.0
    assert result["categories"]["cross"]["mrr_at_10"] == 0.5
