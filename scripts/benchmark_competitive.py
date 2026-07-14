"""Run a reproducible Tessera vs CodeGraph retrieval benchmark.

The benchmark uses pinned Next.js and Flask checkouts to cover TypeScript,
Python, Markdown/MDX, reStructuredText, and mixed code/document queries. Both
engines are reduced to an ordered list of file paths before scoring. CodeGraph
is marked unsupported for document and mixed-content queries because it does
not index those sources; unsupported rows never count as misses.

Example:
    CODEGRAPH_NODE=~/.nvm/versions/node/v22.19.0/bin/node \
      uv run python scripts/benchmark_competitive.py \
      --codegraph-cli /tmp/codegraph-review/dist/bin/codegraph.js \
      --output benchmarks/competitive-2026-07-14.json \
      --report docs/benchmark-competitive-2026-07-14.md
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = Path(__file__).resolve().parent / "benchmark_validation"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_validation.run import (  # noqa: E402
    CODEBASES,
    ensure_codebase,
    index_codebase,
)

from tessera.embeddings import FastembedClient, FastembedReranker  # noqa: E402
from tessera.model_profiles import RERANKER_JINA_TINY  # noqa: E402
from tessera.search import hybrid_search  # noqa: E402

TOP_K = 10
RERANK_POOL = 40
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
CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}
DOCUMENT_EXTENSIONS = {
    ".md",
    ".mdx",
    ".pdf",
    ".yaml",
    ".yml",
    ".json",
    ".html",
    ".htm",
    ".xml",
    ".xsl",
    ".xslt",
    ".svg",
    ".txt",
    ".rst",
    ".csv",
    ".tsv",
    ".log",
    ".ini",
    ".cfg",
    ".toml",
    ".conf",
}
CODEGRAPH_FILE = re.compile(r"^\*\*`([^`]+)`\*\*", re.MULTILINE)


def _matches_expected(path: str, expected: list[str]) -> bool:
    normalized = path.replace("\\", "/").casefold()
    return any(item.replace("\\", "/").casefold() in normalized for item in expected)


def rank_expected(paths: list[str], expected: list[str], k: int = TOP_K) -> int | None:
    """Return the one-based rank of the first expected file."""
    for index, path in enumerate(paths[:k], start=1):
        if _matches_expected(path, expected):
            return index
    return None


def parse_codegraph_files(output: str) -> list[str]:
    """Extract ordered source-file blocks from CodeGraph explore output."""
    seen: set[str] = set()
    paths: list[str] = []
    for path in CODEGRAPH_FILE.findall(output):
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate supported query rows without treating unsupported as misses."""
    supported = [row for row in rows if row["supported"]]
    latencies = [float(row["latency_ms"]) for row in supported]
    ranks = [row["rank"] for row in supported]
    count = len(supported)
    return {
        "queries_total": len(rows),
        "queries_supported": count,
        "queries_unsupported": len(rows) - count,
        "mrr_at_10": (round(sum(1.0 / rank for rank in ranks if rank is not None) / count, 4) if count else 0.0),
        "top_1": round(sum(rank == 1 for rank in ranks) / count, 4) if count else 0.0,
        "top_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / count, 4) if count else 0.0,
        "top_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / count, 4) if count else 0.0,
        "top_10": round(sum(rank is not None for rank in ranks) / count, 4) if count else 0.0,
        "latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "latency_ms_p50": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_ms_p95": round(_percentile(latencies, 0.95), 2) if latencies else 0.0,
    }


def _query_module(codebase: str):
    return importlib.import_module(f"benchmark_validation.{CODEBASES[codebase]['queries']}")


def load_cases(codebase: str, tier: str) -> list[dict[str, Any]]:
    """Load the versioned ground-truth cases for one repository."""
    language = "typescript" if codebase == "nextjs" else "python"
    cases = []
    for query, expected, description, category, query_tier in _query_module(codebase).get_queries(tier):
        cases.append(
            {
                "repository": codebase,
                "language": language,
                "category": "document" if category == "doc" else category,
                "tier": query_tier,
                "query": query,
                "description": description,
                "expected_files": expected,
            }
        )
    return cases


def _source_filter(category: str) -> list[str] | None:
    if category == "code":
        return ["code"]
    if category == "document":
        return DOCUMENT_SOURCE_TYPES
    return None


def validate_ground_truth(cases: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    """Fail when a label cannot resolve to a category-compatible corpus file."""
    corpus: dict[str, list[str]] = {}
    for repository, root in paths.items():
        corpus[repository] = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts and ".codegraph" not in path.parts
        ]

    invalid: list[str] = []
    for case in cases:
        candidates = corpus[case["repository"]]
        if case["category"] == "code":
            candidates = [path for path in candidates if Path(path).suffix in CODE_EXTENSIONS]
        elif case["category"] == "document":
            candidates = [path for path in candidates if Path(path).suffix in DOCUMENT_EXTENSIONS]
        if not any(_matches_expected(path, case["expected_files"]) for path in candidates):
            invalid.append(f"{case['repository']} / {case['description']}: {case['expected_files']}")

    if invalid:
        details = "\n".join(f"- {item}" for item in invalid)
        raise ValueError(f"Ground-truth labels do not resolve in the pinned corpus:\n{details}")


def _dedupe_paths(hits: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for hit in hits:
        path = str(hit.get("file_path", ""))
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def run_tessera_case(
    case: dict[str, Any],
    db: Any,
    embedder: FastembedClient,
    reranker: FastembedReranker | None,
    engine: str,
) -> dict[str, Any]:
    """Run one Tessera query and normalize it to ranked file paths."""
    started = time.perf_counter()
    embedding = np.array(embedder.embed_query(case["query"]), dtype=np.float32)
    hits = hybrid_search(
        case["query"],
        embedding,
        db,
        graph=None,
        limit=RERANK_POOL,
        source_type=_source_filter(case["category"]),
        file_dedup=True,
    )
    if reranker and hits:
        documents = [hit.get("content", "")[:1024] for hit in hits]
        reranked = reranker.rerank(case["query"], documents, top_k=TOP_K)
        hits = [hits[index] for index, _score in reranked if index < len(hits)]
    paths = _dedupe_paths(hits)[:TOP_K]
    elapsed_ms = (time.perf_counter() - started) * 1000
    rank = rank_expected(paths, case["expected_files"])
    return {
        **case,
        "engine": engine,
        "supported": True,
        "rank": rank,
        "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
        "top_files": paths,
        "latency_ms": round(elapsed_ms, 2),
    }


def run_codegraph_case(
    case: dict[str, Any],
    project_path: Path,
    codegraph_cli: Path,
    node_binary: str,
) -> dict[str, Any]:
    """Run one CodeGraph query or mark unsupported source categories."""
    if case["category"] != "code":
        return {
            **case,
            "engine": "codegraph",
            "supported": False,
            "unsupported_reason": "CodeGraph indexes source code, not Markdown/document content",
            "rank": None,
            "reciprocal_rank": None,
            "top_files": [],
            "latency_ms": None,
        }
    started = time.perf_counter()
    completed = subprocess.run(
        [
            node_binary,
            str(codegraph_cli),
            "explore",
            "--path",
            str(project_path),
            "--max-files",
            str(TOP_K),
            case["query"],
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    paths = parse_codegraph_files(completed.stdout)[:TOP_K]
    elapsed_ms = (time.perf_counter() - started) * 1000
    rank = rank_expected(paths, case["expected_files"])
    return {
        **case,
        "engine": "codegraph",
        "supported": True,
        "rank": rank,
        "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
        "top_files": paths,
        "latency_ms": round(elapsed_ms, 2),
    }


def ensure_codegraph_index(project_path: Path, codegraph_cli: Path, node_binary: str) -> dict[str, Any]:
    """Initialize CodeGraph when needed and return index metadata."""
    database = project_path / ".codegraph" / "codegraph.db"
    cached = database.exists()
    started = time.perf_counter()
    if not cached:
        subprocess.run(
            [node_binary, str(codegraph_cli), "init", str(project_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
    return {
        "cached": cached,
        "setup_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build overall and per-repository/category summaries."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comparable_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["engine"], row["repository"], row["category"])].append(row)
        by_engine[row["engine"]].append(row)
        if row["category"] == "code":
            comparable_code[row["engine"]].append(row)
    return {
        "overall": {engine: summarize(engine_rows) for engine, engine_rows in by_engine.items()},
        "comparable_code": {engine: summarize(engine_rows) for engine, engine_rows in comparable_code.items()},
        "segments": {
            f"{engine}/{repository}/{category}": summarize(segment_rows)
            for (engine, repository, category), segment_rows in sorted(grouped.items())
        },
    }


def run_cross_file_suite(codegraph_cli: Path, node_binary: str) -> dict[str, Any]:
    """Run the exact cross-file target suite and parse its JSON output."""
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark_cross_file_resolution.py"),
            "--codegraph-cli",
            str(codegraph_cli),
            "--node-binary",
            node_binary,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )
    return json.loads(completed.stdout)


def _format_pct(value: float) -> str:
    return f"{value:.0%}"


def render_report(report: dict[str, Any]) -> str:
    """Render an answer-first Markdown benchmark report."""
    comparable = report["summary"]["comparable_code"]
    segments = report["summary"]["segments"]
    graph = report["graph_resolution"]
    reranked = comparable["tessera_bge_small_jina_tiny"]
    codegraph = comparable["codegraph"]
    winner = "Tessera" if reranked["mrr_at_10"] >= codegraph["mrr_at_10"] else "CodeGraph"
    lines = [
        "# Tessera vs CodeGraph: Python, TypeScript, and Documentation Benchmark",
        "",
        f"*Generated {report['metadata']['generated_at']} from pinned Flask and Next.js revisions.*",
        "",
        "## Technical Summary",
        "",
        f"**{winner} leads the normalized source-code comparison on MRR@10.** "
        f"Tessera with BGE-small + Jina-tiny scored {reranked['mrr_at_10']:.3f}; "
        f"CodeGraph scored {codegraph['mrr_at_10']:.3f} across the same Python and "
        "TypeScript query set. Tessera also covers document and mixed-source queries, "
        "which CodeGraph does not index and which are therefore reported separately.",
        "",
        f"The exact-edge guardrail remains {graph['tessera']['passed']}/{graph['tessera']['total']} "
        f"for Tessera versus {graph['codegraph']['passed']}/{graph['codegraph']['total']} "
        "for CodeGraph. These graph cases measure exact target correctness, not retrieval ranking.",
        "",
        "## Normalized Code Retrieval",
        "",
        "Both engines are reduced to ordered source-file paths and scored against the same expected files.",
        "",
        "| Engine | Queries | MRR@10 | Top-1 | Top-3 | Top-10 | Mean latency | P95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for engine in ("tessera_bge_small", "tessera_bge_small_jina_tiny", "codegraph"):
        metric = comparable[engine]
        lines.append(
            f"| {engine} | {metric['queries_supported']} | {metric['mrr_at_10']:.3f} | "
            f"{_format_pct(metric['top_1'])} | {_format_pct(metric['top_3'])} | "
            f"{_format_pct(metric['top_10'])} | {metric['latency_ms_mean']:.0f} ms | "
            f"{metric['latency_ms_p95']:.0f} ms |"
        )
    lines += [
        "",
        "Latency is directional: Tessera uses an in-process cached index, while CodeGraph is invoked as a fresh CLI process for each query.",
        "",
        "## Language and Content Segments",
        "",
        "| Engine | Repository / segment | Queries | MRR@10 | Top-3 | Top-10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, metric in segments.items():
        engine, repository, category = key.split("/")
        if metric["queries_supported"] == 0:
            continue
        lines.append(
            f"| {engine} | {repository} / {category} | {metric['queries_supported']} | "
            f"{metric['mrr_at_10']:.3f} | {_format_pct(metric['top_3'])} | "
            f"{_format_pct(metric['top_10'])} |"
        )
    lines += [
        "",
        "## Method and Definitions",
        "",
        f"- **Corpus:** Next.js `{report['repositories']['nextjs']['revision'][:12]}` and Flask "
        f"`{report['repositories']['flask']['revision'][:12]}`.",
        f"- **Queries:** {report['metadata']['query_cases']} unique ground-truth questions; each Tessera "
        "configuration runs all questions, while CodeGraph runs the source-code subset.",
        "- **MRR@10:** reciprocal rank of the first expected file, averaged across supported queries; a miss contributes zero.",
        "- **Top-k:** fraction of supported queries with any expected file in the first k unique file results.",
        "- **Documents:** Markdown, MDX, and reStructuredText are grouped as document retrieval. Unsupported CodeGraph rows are excluded rather than scored as failures.",
        "- **Tessera configuration:** BGE-small embeddings, hybrid retrieval, file deduplication; one run without reranking and one with Jina-tiny reranking.",
        "- **CodeGraph adapter:** ordered source-code file blocks emitted by `codegraph explore --max-files 10`.",
        "",
        "## Validation Assessment",
        "",
        "**Share with caveats.** File-level relevance labels and calculations are deterministic and machine-readable. "
        "The sample is broad enough to guide engineering priorities, but it is curated rather than independently blinded, "
        "and latency is not a process-normalized performance comparison.",
        "",
        "Recommended next step: inspect the per-query misses in the JSON artifact, then add the highest-impact failures as permanent regression cases before changing models or ranker weights.",
        "",
    ]
    return "\n".join(lines)


def _version(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, cwd=cwd, check=True)
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codegraph-cli", type=Path, required=True)
    parser.add_argument(
        "--node-binary",
        default=os.environ.get("CODEGRAPH_NODE") or shutil.which("node") or "node",
    )
    parser.add_argument("--tier", choices=["quick", "standard", "full"], default="standard")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    repositories = {}
    cases = []
    paths: dict[str, Path] = {}
    for name in ("nextjs", "flask"):
        path = Path(ensure_codebase(name))
        paths[name] = path
        revision = _version(["git", "rev-parse", "HEAD"], cwd=path)
        repositories[name] = {
            "path": str(path),
            "revision": revision,
            "configured_ref": CODEBASES[name]["ref"],
            "languages": CODEBASES[name]["languages"],
            "codegraph_index": ensure_codegraph_index(path, args.codegraph_cli, args.node_binary),
        }
        cases.extend(load_cases(name, args.tier))

    validate_ground_truth(cases, paths)

    embedder = FastembedClient(model_name="BAAI/bge-small-en-v1.5")
    _ = embedder.embed_query("benchmark warmup")
    reranker = FastembedReranker(model_name=RERANKER_JINA_TINY)
    _ = reranker.rerank("benchmark", ["benchmark warmup"], top_k=1)

    databases = {}
    for name, path in paths.items():
        databases[name] = index_codebase(name, str(path), embedder, model_key="bge-small", reindex=args.reindex)

    rows: list[dict[str, Any]] = []
    total_runs = len(cases) * 2 + sum(case["category"] == "code" for case in cases)
    completed_runs = 0
    for case in cases:
        for engine, selected_reranker in (
            ("tessera_bge_small", None),
            ("tessera_bge_small_jina_tiny", reranker),
        ):
            rows.append(run_tessera_case(case, databases[case["repository"]], embedder, selected_reranker, engine))
            completed_runs += 1
            print(f"[{completed_runs}/{total_runs}] {engine}: {case['repository']} / {case['description']}")

        codegraph_row = run_codegraph_case(case, paths[case["repository"]], args.codegraph_cli, args.node_binary)
        rows.append(codegraph_row)
        if codegraph_row["supported"]:
            completed_runs += 1
            print(f"[{completed_runs}/{total_runs}] codegraph: {case['repository']} / {case['description']}")

    output = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "tier": args.tier,
            "query_cases": len(cases),
            "measured_query_runs": total_runs,
            "top_k": TOP_K,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tessera_revision": _version(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
            "codegraph_version": _version([args.node_binary, str(args.codegraph_cli), "--version"]),
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "reranking_model": RERANKER_JINA_TINY,
        },
        "repositories": repositories,
        "summary": aggregate(rows),
        "graph_resolution": run_cross_file_suite(args.codegraph_cli, args.node_binary),
        "queries": rows,
        "limitations": [
            "CodeGraph does not index Markdown/document content; unsupported rows are excluded.",
            "Tessera runs in-process while CodeGraph launches a CLI process per query; latency is directional.",
            "Ground truth is curated and file-level; it does not score answer synthesis or snippet quality.",
            "Cached indexes are used unless --reindex is supplied, so setup_ms is not a cold-index benchmark.",
        ],
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(output), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
