"""Compare Tessera and CodeGraph on exact cross-file call resolution.

The fixture deliberately includes same-named unrelated functions. A benchmark
counts a case as correct only when the call edge reaches the expected file and
symbol, never merely when a same-named target exists somewhere in the graph.

Usage:
    uv run python scripts/benchmark_cross_file_resolution.py \
        --codegraph-cli /path/to/codegraph/dist/bin/codegraph.js

Set CODEGRAPH_NODE when the CodeGraph checkout requires a particular Node.js
binary (for example, Node 22 LTS).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tessera.db import ProjectDB
from tessera.indexer import IndexerPipeline


@dataclass(frozen=True)
class Case:
    """One exact call-edge expectation in the generated fixture."""

    name: str
    caller_file: str
    caller_symbol: str
    target_file: str
    target_symbol: str


CASES = (
    Case("python_named_alias", "caller_alias.py", "caller_alias", "service.py", "run"),
    Case("python_module_call", "caller_module.py", "caller_module", "service.py", "run"),
    Case("python_reexport", "caller_reexport.py", "caller_reexport", "service.py", "run"),
    Case("typescript_named_alias", "caller_alias.ts", "callerAlias", "service.ts", "run"),
)


FIXTURE_FILES = {
    "service.py": 'def run() -> str:\n    return "service"\n',
    "facade.py": "from service import run\n\n__all__ = [\"run\"]\n",
    "unrelated.py": 'def execute() -> str:\n    return "unrelated"\n\n\ndef run() -> str:\n    return "other"\n',
    "caller_alias.py": "from service import run as execute\n\n\ndef caller_alias() -> str:\n    return execute()\n",
    "caller_module.py": "import service\n\n\ndef caller_module() -> str:\n    return service.run()\n",
    "caller_reexport.py": "from facade import run\n\n\ndef caller_reexport() -> str:\n    return run()\n",
    "service.ts": "export function run(): string { return 'service'; }\n",
    "unrelated.ts": "export function execute(): string { return 'unrelated'; }\n",
    "caller_alias.ts": (
        "import { run as execute } from './service';\n\n"
        "export function callerAlias(): string { return execute(); }\n"
    ),
}


def create_fixture(root: Path) -> None:
    """Write the small, collision-heavy project used by both engines."""
    for relative_path, content in FIXTURE_FILES.items():
        (root / relative_path).write_text(content, encoding="utf-8")


def score_targets(targets: dict[tuple[str, str], tuple[str, str] | None]) -> dict[str, bool]:
    """Return one exact-match result per case from a caller-to-target mapping."""
    return {
        case.name: targets.get((case.caller_file, case.caller_symbol))
        == (case.target_file, case.target_symbol)
        for case in CASES
    }


def format_targets(
    targets: dict[tuple[str, str], tuple[str, str] | None],
) -> dict[str, tuple[str, str] | None]:
    """Convert internal tuple keys into stable JSON object keys."""
    return {
        f"{source_file}:{caller_name}": target
        for (source_file, caller_name), target in sorted(targets.items())
    }


def tessera_targets(project_root: Path, data_root: Path) -> tuple[dict[tuple[str, str], tuple[str, str] | None], float]:
    """Index the fixture and return Tessera's resolved call targets."""
    ProjectDB.base_dir = str(data_root)
    pipeline = IndexerPipeline(str(project_root), languages=["python", "typescript"])
    started = time.perf_counter()
    pipeline.index_project_sync()
    elapsed_ms = (time.perf_counter() - started) * 1000

    rows = pipeline.project_db.conn.execute(
        """
        SELECT source.path, caller.name, destination.path, target.name
        FROM refs
        JOIN symbols AS caller ON caller.id = refs.from_symbol_id
        JOIN files AS source ON source.id = caller.file_id
        LEFT JOIN symbols AS target ON target.id = refs.to_symbol_id
        LEFT JOIN files AS destination ON destination.id = target.file_id
        WHERE refs.kind = 'calls'
        """
    ).fetchall()
    targets: dict[tuple[str, str], tuple[str, str] | None] = {}
    for source_file, caller_name, target_file, target_name in rows:
        if not isinstance(source_file, str) or not isinstance(caller_name, str):
            continue
        if isinstance(target_file, str) and isinstance(target_name, str):
            targets[(source_file, caller_name)] = (target_file, target_name)
        else:
            targets[(source_file, caller_name)] = None
    return targets, elapsed_ms


def codegraph_targets(
    project_root: Path, codegraph_cli: Path, node_binary: str
) -> tuple[dict[tuple[str, str], tuple[str, str] | None], float]:
    """Index with CodeGraph's CLI and query exact calls from its SQLite graph."""
    started = time.perf_counter()
    subprocess.run(
        [node_binary, str(codegraph_cli), "init", str(project_root)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    database_path = project_root / ".codegraph" / "codegraph.db"
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT source.file_path, source.name, target.file_path, target.name
            FROM edges AS edge
            JOIN nodes AS source ON source.id = edge.source
            JOIN nodes AS target ON target.id = edge.target
            WHERE edge.kind = 'calls'
            """
        ).fetchall()
    targets: dict[tuple[str, str], tuple[str, str] | None] = {}
    for source_file, caller_name, target_file, target_name in rows:
        if all(isinstance(value, str) for value in (source_file, caller_name, target_file, target_name)):
            targets[(source_file, caller_name)] = (target_file, target_name)
    return targets, elapsed_ms


def main() -> int:
    """Run the suite and print a machine-readable comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codegraph-cli",
        type=Path,
        required=True,
        help="Path to CodeGraph's dist/bin/codegraph.js",
    )
    parser.add_argument(
        "--node-binary",
        default=os.environ.get("CODEGRAPH_NODE") or shutil.which("node") or "node",
        help="Node.js binary used to run CodeGraph (default: CODEGRAPH_NODE or PATH)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="tessera-codegraph-") as temporary_directory:
        fixture_root = Path(temporary_directory) / "fixture"
        fixture_root.mkdir()
        create_fixture(fixture_root)

        tessera, tessera_ms = tessera_targets(fixture_root, Path(temporary_directory) / "tessera")
        codegraph, codegraph_ms = codegraph_targets(
            fixture_root, args.codegraph_cli, args.node_binary
        )

        report = {
            "cases": [asdict(case) for case in CASES],
            "tessera": {
                "cold_index_ms": round(tessera_ms, 2),
                "resolved_targets": format_targets(tessera),
                "passes": score_targets(tessera),
            },
            "codegraph": {
                "cold_index_ms": round(codegraph_ms, 2),
                "resolved_targets": format_targets(codegraph),
                "passes": score_targets(codegraph),
            },
        }
        for engine in ("tessera", "codegraph"):
            report[engine]["passed"] = sum(report[engine]["passes"].values())
            report[engine]["total"] = len(CASES)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
