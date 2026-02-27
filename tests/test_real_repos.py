"""Smoke tests: index real local repositories and validate results.

Runs the full indexing pipeline against actual repos on disk to verify
Tessera handles real-world code. Skips repos that aren't present.
No embedding endpoint needed — tests structural indexing only.
"""

import os
import tempfile
import pytest
from tessera.db import ProjectDB, GlobalDB
from tessera.indexer import IndexerPipeline

# Repos to test: (name, path, languages, min_expected_files)
REAL_REPOS = [
    (
        "popup-maker-php",
        "/Users/danieliser/SVN/popup-maker/trunk",
        ["php", "javascript"],
        50,
    ),
    (
        "popup-maker-ts",
        "/Users/danieliser/Local Sites/popup-maker/app/public/wp-content/plugins/popup-maker/packages",
        ["typescript", "javascript"],
        100,
    ),
    (
        "popup-maker-local",
        "/Users/danieliser/Local Sites/popup-maker/app/public/wp-content/plugins/popup-maker",
        ["php", "typescript", "javascript"],
        50,
    ),
    (
        "popup-maker-pro",
        "/Users/danieliser/Local Sites/popup-maker/app/public/wp-content/plugins/popup-maker-pro",
        ["php", "typescript", "javascript"],
        50,
    ),
    (
        "popup-maker-ecommerce",
        "/Users/danieliser/Local Sites/popup-maker/app/public/wp-content/plugins/popup-maker-ecommerce-popups",
        ["php", "typescript"],
        10,
    ),
    (
        "popup-maker-lms",
        "/Users/danieliser/Local Sites/popup-maker/app/public/wp-content/plugins/popup-maker-lms-popups",
        ["php", "typescript"],
        10,
    ),
    (
        "ignition-stack",
        "/Users/danieliser/Projects/ignition-stack",
        ["typescript"],
        100,
    ),
    (
        "tessera-self",
        "/Users/danieliser/Toolkit/codemem",
        ["python"],
        10,
    ),
    (
        "persistence",
        "/Users/danieliser/Toolkit/persistence",
        ["python"],
        20,
    ),
]


def _skip_if_missing(path: str):
    if not os.path.isdir(path):
        pytest.skip(f"Repo not found: {path}")


class TestRealRepoIndexing:
    """Index real repos and verify symbols, refs, and edges are extracted."""

    @pytest.fixture(params=REAL_REPOS, ids=[r[0] for r in REAL_REPOS])
    def repo_index(self, request, tmp_path):
        """Index a real repo into a temp DB and return (pipeline, stats, db)."""
        name, path, languages, min_files = request.param
        _skip_if_missing(path)

        db_path = str(tmp_path / f"{name}.db")
        project_db = ProjectDB(db_path)
        global_db = GlobalDB(str(tmp_path / "global.db"))

        pipeline = IndexerPipeline(
            project_path=path,
            project_db=project_db,
            global_db=global_db,
            embedding_client=None,
            languages=languages,
        )
        pipeline.register(name)
        stats = pipeline.index_project()

        return name, pipeline, stats, project_db, min_files

    def test_files_indexed(self, repo_index):
        name, pipeline, stats, db, min_files = repo_index
        total = stats.files_processed + stats.files_skipped
        assert total >= min_files, (
            f"{name}: expected >= {min_files} files discovered, got {total} "
            f"(processed={stats.files_processed}, skipped={stats.files_skipped})"
        )

    def test_no_crashes(self, repo_index):
        """All files processed or skipped, none failed."""
        name, pipeline, stats, db, _ = repo_index
        assert stats.files_failed == 0, (
            f"{name}: {stats.files_failed} files failed indexing"
        )

    def test_symbols_extracted(self, repo_index):
        name, pipeline, stats, db, _ = repo_index
        assert stats.symbols_extracted > 0, (
            f"{name}: no symbols extracted from {stats.files_processed} files"
        )

    def test_symbols_have_kinds(self, repo_index):
        """All symbols have a valid kind field."""
        name, pipeline, stats, db, _ = repo_index
        rows = db.conn.execute(
            "SELECT DISTINCT kind FROM symbols WHERE project_id = ?",
            (pipeline.project_id,)
        ).fetchall()
        kinds = {r[0] for r in rows}
        # Should have at least functions + module
        assert "module" in kinds, f"{name}: no module symbols found"
        assert len(kinds) >= 2, f"{name}: only found kinds: {kinds}"

    def test_references_extracted(self, repo_index):
        name, pipeline, stats, db, _ = repo_index
        count = db.conn.execute(
            "SELECT COUNT(*) FROM refs WHERE project_id = ?",
            (pipeline.project_id,)
        ).fetchone()[0]
        assert count > 0, f"{name}: no references extracted"

    def test_edges_extracted(self, repo_index):
        name, pipeline, stats, db, _ = repo_index
        count = db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE project_id = ?",
            (pipeline.project_id,)
        ).fetchone()[0]
        assert count > 0, f"{name}: no edges extracted"

    def test_chunks_created(self, repo_index):
        name, pipeline, stats, db, _ = repo_index
        assert stats.chunks_created > 0, (
            f"{name}: no chunks created from {stats.files_processed} files"
        )

    def test_module_symbols_have_file_paths(self, repo_index):
        """Module symbols should use file paths, not '<module>'."""
        name, pipeline, stats, db, _ = repo_index
        modules = db.conn.execute(
            "SELECT name FROM symbols WHERE project_id = ? AND kind = 'module'",
            (pipeline.project_id,)
        ).fetchall()
        for (mod_name,) in modules:
            assert mod_name != "<module>", (
                f"{name}: found old-style '<module>' pseudo-symbol"
            )
            # Should look like a file path
            assert "." in mod_name, (
                f"{name}: module name '{mod_name}' doesn't look like a file path"
            )


class TestRealRepoStats:
    """Print detailed stats for each repo (useful for manual review)."""

    @pytest.fixture(params=REAL_REPOS, ids=[r[0] for r in REAL_REPOS])
    def indexed_repo(self, request, tmp_path):
        name, path, languages, _ = request.param
        _skip_if_missing(path)

        db_path = str(tmp_path / f"{name}.db")
        project_db = ProjectDB(db_path)
        global_db = GlobalDB(str(tmp_path / "global.db"))

        pipeline = IndexerPipeline(
            project_path=path,
            project_db=project_db,
            global_db=global_db,
            embedding_client=None,
            languages=languages,
        )
        pipeline.register(name)
        stats = pipeline.index_project()
        return name, stats, project_db, pipeline

    def test_print_summary(self, indexed_repo, capsys):
        name, stats, db, pipeline = indexed_repo

        # Symbol breakdown
        kind_rows = db.conn.execute(
            "SELECT kind, COUNT(*) FROM symbols WHERE project_id = ? GROUP BY kind ORDER BY COUNT(*) DESC",
            (pipeline.project_id,)
        ).fetchall()

        # Reference breakdown
        ref_rows = db.conn.execute(
            "SELECT kind, COUNT(*) FROM refs WHERE project_id = ? GROUP BY kind ORDER BY COUNT(*) DESC",
            (pipeline.project_id,)
        ).fetchall()

        # Edge breakdown
        edge_rows = db.conn.execute(
            "SELECT type, COUNT(*) FROM edges WHERE project_id = ? GROUP BY type ORDER BY COUNT(*) DESC",
            (pipeline.project_id,)
        ).fetchall()

        print(f"\n{'='*60}")
        print(f"  {name} — Real Repo Indexing Stats")
        print(f"{'='*60}")
        print(f"  Files: {stats.files_processed} indexed, {stats.files_skipped} skipped, {stats.files_failed} failed")
        print(f"  Symbols: {stats.symbols_extracted}")
        print(f"  Chunks: {stats.chunks_created}")
        print(f"  Time: {stats.time_elapsed:.2f}s")
        print()
        print("  Symbol Kinds:")
        for kind, count in kind_rows:
            print(f"    {kind:20s} {count:>6}")
        print()
        print("  Reference Kinds:")
        for kind, count in ref_rows:
            print(f"    {kind:20s} {count:>6}")
        print()
        print("  Edge Types:")
        for etype, count in edge_rows:
            print(f"    {etype:20s} {count:>6}")
        print(f"{'='*60}")
