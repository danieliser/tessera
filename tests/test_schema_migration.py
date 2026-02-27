"""Tests for schema migration v1 → v2."""
import tempfile
import sqlite3
import pytest
from tessera.db import ProjectDB


class TestSchemaMigration:
    def test_fresh_db_has_v2_schema(self):
        db = ProjectDB(tempfile.mkdtemp())
        cur = db.conn.execute("SELECT value FROM _meta WHERE key='schema_version'")
        assert cur.fetchone()[0] == "2"

    def test_new_columns_exist(self):
        db = ProjectDB(tempfile.mkdtemp())
        cols = [r[1] for r in db.conn.execute("PRAGMA table_info(chunk_meta)").fetchall()]
        for col in ["source_type", "section_heading", "key_path", "page_number", "parent_section"]:
            assert col in cols

    def test_migration_idempotent(self):
        db = ProjectDB(tempfile.mkdtemp())
        db._run_migrations()  # Should not raise
        cur = db.conn.execute("SELECT value FROM _meta WHERE key='schema_version'")
        assert cur.fetchone()[0] == "2"

    def test_source_type_defaults_to_code(self):
        db = ProjectDB(tempfile.mkdtemp())
        fid = db.upsert_file(1, "test.py", "python", "abc123")
        ids = db.insert_chunks([{
            "project_id": 1, "file_id": fid, "start_line": 1, "end_line": 5,
            "symbol_ids": [], "ast_type": "function", "chunk_type": "code",
            "content": "def foo(): pass", "file_path": "test.py",
        }])
        chunk = db.get_chunk(ids[0])
        assert chunk["source_type"] == "code"

    def test_document_chunk_stored_with_metadata(self):
        db = ProjectDB(tempfile.mkdtemp())
        fid = db.upsert_file(1, "doc.md", "markdown", "abc123")
        ids = db.insert_chunks([{
            "project_id": 1, "file_id": fid, "start_line": 1, "end_line": 10,
            "symbol_ids": [], "ast_type": "document", "chunk_type": "document",
            "content": "# Hello", "source_type": "markdown",
            "section_heading": "# Hello", "parent_section": "",
            "file_path": "doc.md",
        }])
        chunk = db.get_chunk(ids[0])
        assert chunk["source_type"] == "markdown"
        assert chunk["section_heading"] == "# Hello"
