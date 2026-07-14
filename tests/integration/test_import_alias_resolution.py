"""End-to-end coverage for import-aware cross-file call resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.indexer import IndexerPipeline


@pytest.mark.parametrize(
    ("language", "service_name", "caller_name", "service_source", "caller_source"),
    [
        (
            "python",
            "service.py",
            "caller.py",
            "def run() -> str:\n    return 'service'\n",
            "from service import run as execute\n\ndef caller() -> str:\n    return execute()\n",
        ),
        (
            "typescript",
            "service.ts",
            "caller.ts",
            "export function run(): string { return 'service'; }\n",
            "import { run as execute } from './service';\n\nexport function caller(): string { return execute(); }\n",
        ),
    ],
)
def test_import_alias_call_resolves_to_symbol_in_imported_file(
    tmp_path: Path,
    language: str,
    service_name: str,
    caller_name: str,
    service_source: str,
    caller_source: str,
) -> None:
    """An import alias must beat an unrelated same-named local candidate."""
    (tmp_path / service_name).write_text(service_source)
    (tmp_path / caller_name).write_text(caller_source)
    (tmp_path / f"unrelated.{service_name.rsplit('.', 1)[1]}").write_text(
        service_source.replace("run", "execute")
    )

    pipeline = IndexerPipeline(str(tmp_path), languages=[language])
    pipeline.index_project_sync()

    db = pipeline.project_db
    caller = db.lookup_symbols("caller", kind="function")[0]
    service_file = db.get_file(path=service_name)
    assert service_file is not None
    service_run = db.conn.execute(
        "SELECT id FROM symbols WHERE file_id = ? AND name = 'run'",
        (service_file["id"],),
    ).fetchone()
    assert service_run is not None

    refs = db.get_refs(symbol_id=caller["id"])
    execute_ref = next(ref for ref in refs if ref["to_symbol_name"] == "execute")
    assert execute_ref["to_symbol_id"] == service_run[0]


def test_python_module_import_call_resolves_to_symbol_in_imported_file(
    tmp_path: Path,
) -> None:
    """A module-qualified call must not resolve to an unrelated global symbol."""
    (tmp_path / "service.py").write_text("def run() -> str:\n    return 'service'\n")
    (tmp_path / "caller.py").write_text(
        "import service\n\ndef caller() -> str:\n    return service.run()\n"
    )
    (tmp_path / "unrelated.py").write_text("def run() -> str:\n    return 'unrelated'\n")

    pipeline = IndexerPipeline(str(tmp_path), languages=["python"])
    pipeline.index_project_sync()

    db = pipeline.project_db
    caller = db.lookup_symbols("caller", kind="function")[0]
    service_file = db.get_file(path="service.py")
    assert service_file is not None
    service_run = db.conn.execute(
        "SELECT id FROM symbols WHERE file_id = ? AND name = 'run'",
        (service_file["id"],),
    ).fetchone()
    assert service_run is not None

    run_ref = next(ref for ref in db.get_refs(symbol_id=caller["id"]) if ref["to_symbol_name"] == "run")
    assert run_ref["to_symbol_id"] == service_run[0]


def test_python_one_hop_reexport_resolves_to_original_symbol(tmp_path: Path) -> None:
    """A named import forwarded once through an in-project facade stays precise."""
    (tmp_path / "service.py").write_text("def run() -> str:\n    return 'service'\n")
    (tmp_path / "facade.py").write_text("from service import run\n")
    (tmp_path / "caller.py").write_text(
        "from facade import run\n\ndef caller() -> str:\n    return run()\n"
    )
    (tmp_path / "unrelated.py").write_text("def run() -> str:\n    return 'unrelated'\n")

    pipeline = IndexerPipeline(str(tmp_path), languages=["python"])
    pipeline.index_project_sync()

    db = pipeline.project_db
    caller = db.lookup_symbols("caller", kind="function")[0]
    service_file = db.get_file(path="service.py")
    assert service_file is not None
    service_run = db.conn.execute(
        "SELECT id FROM symbols WHERE file_id = ? AND name = 'run'",
        (service_file["id"],),
    ).fetchone()
    assert service_run is not None

    run_ref = next(ref for ref in db.get_refs(symbol_id=caller["id"]) if ref["to_symbol_name"] == "run")
    assert run_ref["to_symbol_id"] == service_run[0]
