"""Aggregate mixed-media benchmark CSVs into a summary leaderboard.

Usage:
    uv run python scripts/aggregate_mixed.py                  # latest CSV per model
    uv run python scripts/aggregate_mixed.py --all            # all CSVs (shows history)
    uv run python scripts/aggregate_mixed.py --routed         # routed MRR only
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict


def load_csvs(pattern: str = "benchmarks/mixed_*.csv") -> dict[str, list[dict]]:
    """Load CSVs, keyed by (model, mode). Returns latest per model+mode by default."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No CSVs found matching {pattern}")
        return {}

    # Group by (model, mode) — take latest file per model
    latest_per_model: dict[str, str] = {}
    for fpath in files:
        basename = os.path.basename(fpath)
        # mixed_bge-small_20260305_133337.csv → bge-small
        parts = basename.replace("mixed_", "").split("_")
        model = parts[0]
        latest_per_model[model] = fpath  # later files overwrite

    all_rows: dict[str, list[dict]] = {}
    for model, fpath in sorted(latest_per_model.items()):
        rows = list(csv.DictReader(open(fpath)))
        all_rows[model] = rows

    return all_rows


def compute_mrr(rows: list[dict], category: str | None = None, mode: str | None = None) -> float:
    filtered = rows
    if category:
        filtered = [r for r in filtered if r.get("category") == category]
    if mode:
        filtered = [r for r in filtered if r.get("mode") == mode]
    if not filtered:
        return 0.0
    return sum(float(r["mrr"]) for r in filtered) / len(filtered)


def best_mode(rows: list[dict], category: str) -> tuple[str, float]:
    """Return (mode_label, mrr) for the best mode in a category."""
    modes = {r["mode"] for r in rows}
    best_m, best_mrr = "—", 0.0
    for m in modes:
        mrr = compute_mrr(rows, category=category, mode=m)
        if mrr > best_mrr:
            best_mrr = mrr
            best_m = m
    return best_m, best_mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Show all CSVs, not just latest")
    parser.add_argument("--routed", action="store_true", help="Show routed breakdown only")
    args = parser.parse_args()

    data = load_csvs()
    if not data:
        return

    models = sorted(data.keys())

    print("\n" + "=" * 110)
    print("MIXED-MEDIA BENCHMARK LEADERBOARD")
    print("=" * 110)
    print(f"{'Model':<16} {'Code MRR':>10} {'Code Mode':>14} {'Doc MRR':>9} {'Doc Mode':>16} {'Cross MRR':>10} {'Cross Mode':>14} {'Blended':>9}")
    print("-" * 110)

    summary = []
    for model in models:
        rows = data[model]
        code_mode, code_mrr = best_mode(rows, "code")
        doc_mode, doc_mrr = best_mode(rows, "doc")
        cross_mode, cross_mrr = best_mode(rows, "cross")
        blended = (code_mrr + doc_mrr + cross_mrr) / 3

        summary.append((blended, model, code_mrr, code_mode, doc_mrr, doc_mode, cross_mrr, cross_mode))
        print(f"{model:<16} {code_mrr:>10.3f} {code_mode:>14} {doc_mrr:>9.3f} {doc_mode:>16} {cross_mrr:>10.3f} {cross_mode:>14} {blended:>9.3f}")

    summary.sort(reverse=True)
    print("\n" + "=" * 110)
    print("RANKED BY BLENDED MRR")
    print("=" * 110)
    for rank, (blended, model, code_mrr, code_mode, doc_mrr, doc_mode, cross_mrr, cross_mode) in enumerate(summary, 1):
        print(f"  #{rank}  {model:<16}  Blended: {blended:.3f}  "
              f"(code: {code_mrr:.3f} via {code_mode}, "
              f"doc: {doc_mrr:.3f} via {doc_mode}, "
              f"cross: {cross_mrr:.3f} via {cross_mode})")

    # ROUTED mode comparison
    routed_models = [m for m in models if any(r["mode"] == "ROUTED" for r in data[m])]
    if routed_models:
        print("\n" + "=" * 110)
        print("ROUTED MODE COMPARISON (source-type aware routing)")
        print("=" * 110)
        print(f"{'Model':<16} {'Code':>8} {'Doc':>8} {'Cross':>8} {'Blended':>9} {'Avg ms/q':>10}")
        print("-" * 65)
        routed_summary = []
        for model in routed_models:
            rows = data[model]
            code_mrr = compute_mrr(rows, category="code", mode="ROUTED")
            doc_mrr = compute_mrr(rows, category="doc", mode="ROUTED")
            cross_mrr = compute_mrr(rows, category="cross", mode="ROUTED")
            blended = (code_mrr + doc_mrr + cross_mrr) / 3
            routed_rows = [r for r in rows if r["mode"] == "ROUTED"]
            avg_ms = sum(float(r["latency_ms"]) for r in routed_rows) / len(routed_rows) if routed_rows else 0
            routed_summary.append((blended, model, code_mrr, doc_mrr, cross_mrr, avg_ms))

        routed_summary.sort(reverse=True)
        for blended, model, code_mrr, doc_mrr, cross_mrr, avg_ms in routed_summary:
            print(f"{model:<16} {code_mrr:>8.3f} {doc_mrr:>8.3f} {cross_mrr:>8.3f} {blended:>9.3f} {avg_ms:>10.0f}")


if __name__ == "__main__":
    main()
