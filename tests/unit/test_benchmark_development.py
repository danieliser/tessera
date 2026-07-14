"""Tests for development benchmark trace comparisons."""

from __future__ import annotations

import json

import pytest
from scripts.benchmark_development import _compare_baseline


def test_compare_baseline_requires_case_coverage_and_reports_order(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({
        "queries": [
            {
                "repository": "fixture",
                "case_id": "one",
                "rank": 1,
                "top_files": ["a.py", "b.py"],
            },
        ],
    }), encoding="utf-8")

    equal = _compare_baseline([
        {
            "repository": "fixture",
            "case_id": "one",
            "rank": 1,
            "top_files": ["a.py", "b.py"],
        },
    ], baseline_path)
    assert equal["exact_rank_and_order_match"] is True

    changed = _compare_baseline([
        {
            "repository": "fixture",
            "case_id": "one",
            "rank": 2,
            "top_files": ["b.py", "a.py"],
        },
    ], baseline_path)
    assert changed["exact_rank_and_order_match"] is False
    assert len(changed["rank_changes"]) == 1
    assert len(changed["ordered_top_file_changes"]) == 1

    with pytest.raises(ValueError, match="coverage differs"):
        _compare_baseline([], baseline_path)
