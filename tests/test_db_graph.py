"""Tests for graph query methods in ProjectDB."""

import tempfile
import json
import pytest
from tessera.db import ProjectDB


class TestGetAllEdges:
    """Test get_all_edges method."""

    def test_get_all_edges_empty(self):
        """Test get_all_edges returns empty list for project with no edges."""
        db = ProjectDB(tempfile.mkdtemp())
        result = db.get_all_edges(project_id=1)
        assert result == []

    def test_get_all_edges_with_data(self):
        """Test get_all_edges returns all edges for a project."""
        db = ProjectDB(tempfile.mkdtemp())

        # Insert test file
        file_id = db.upsert_file(
            project_id=1,
            path="test.py",
            language="python",
            file_hash="abc123"
        )

        # Insert test symbols
        symbol_ids = db.insert_symbols([
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "func_a",
                "kind": "function",
                "line": 10,
                "col": 0,
                "scope": "module",
                "signature": "def func_a()"
            },
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "func_b",
                "kind": "function",
                "line": 20,
                "col": 0,
                "scope": "module",
                "signature": "def func_b()"
            },
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "func_c",
                "kind": "function",
                "line": 30,
                "col": 0,
                "scope": "module",
                "signature": "def func_c()"
            }
        ])

        # Insert edges
        edge_data = [
            {
                "project_id": 1,
                "from_id": symbol_ids[0],
                "to_id": symbol_ids[1],
                "type": "calls",
                "weight": 1.0
            },
            {
                "project_id": 1,
                "from_id": symbol_ids[1],
                "to_id": symbol_ids[2],
                "type": "calls",
                "weight": 0.8
            }
        ]
        db.insert_edges(edge_data)

        # Test get_all_edges
        result = db.get_all_edges(project_id=1)
        assert len(result) == 2

        # Verify structure and content
        assert all(key in result[0] for key in ["from_id", "to_id", "weight"])

        # Verify edges are ordered by from_id, to_id
        assert result[0]["from_id"] == symbol_ids[0]
        assert result[0]["to_id"] == symbol_ids[1]
        assert result[0]["weight"] == 1.0

        assert result[1]["from_id"] == symbol_ids[1]
        assert result[1]["to_id"] == symbol_ids[2]
        assert result[1]["weight"] == 0.8

    def test_get_all_edges_multiple_projects(self):
        """Test get_all_edges only returns edges for specified project."""
        db = ProjectDB(tempfile.mkdtemp())

        # Create files and symbols for project 1
        file_id_1 = db.upsert_file(1, "file1.py", "python", "hash1")
        sym_ids_1 = db.insert_symbols([
            {
                "project_id": 1,
                "file_id": file_id_1,
                "name": "func_1",
                "kind": "function",
                "line": 1,
                "col": 0,
                "scope": "module",
                "signature": "def func_1()"
            },
            {
                "project_id": 1,
                "file_id": file_id_1,
                "name": "func_2",
                "kind": "function",
                "line": 5,
                "col": 0,
                "scope": "module",
                "signature": "def func_2()"
            }
        ])

        # Create files and symbols for project 2
        file_id_2 = db.upsert_file(2, "file2.py", "python", "hash2")
        sym_ids_2 = db.insert_symbols([
            {
                "project_id": 2,
                "file_id": file_id_2,
                "name": "func_3",
                "kind": "function",
                "line": 1,
                "col": 0,
                "scope": "module",
                "signature": "def func_3()"
            },
            {
                "project_id": 2,
                "file_id": file_id_2,
                "name": "func_4",
                "kind": "function",
                "line": 5,
                "col": 0,
                "scope": "module",
                "signature": "def func_4()"
            }
        ])

        # Insert edges for both projects
        db.insert_edges([
            {"project_id": 1, "from_id": sym_ids_1[0], "to_id": sym_ids_1[1], "type": "calls", "weight": 1.0},
            {"project_id": 2, "from_id": sym_ids_2[0], "to_id": sym_ids_2[1], "type": "calls", "weight": 1.0}
        ])

        # Get edges for project 1 only
        result_1 = db.get_all_edges(project_id=1)
        assert len(result_1) == 1
        assert result_1[0]["from_id"] == sym_ids_1[0]

        # Get edges for project 2 only
        result_2 = db.get_all_edges(project_id=2)
        assert len(result_2) == 1
        assert result_2[0]["from_id"] == sym_ids_2[0]


class TestGetAllSymbols:
    """Test get_all_symbols method."""

    def test_get_all_symbols_empty(self):
        """Test get_all_symbols returns empty list for project with no symbols."""
        db = ProjectDB(tempfile.mkdtemp())
        result = db.get_all_symbols(project_id=1)
        assert result == []

    def test_get_all_symbols_with_data(self):
        """Test get_all_symbols returns all symbols for a project."""
        db = ProjectDB(tempfile.mkdtemp())

        # Insert test file
        file_id = db.upsert_file(
            project_id=1,
            path="test.py",
            language="python",
            file_hash="abc123"
        )

        # Insert test symbols
        symbol_data = [
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "MyClass",
                "kind": "class",
                "line": 5,
                "col": 0,
                "scope": "module",
                "signature": "class MyClass"
            },
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "my_function",
                "kind": "function",
                "line": 15,
                "col": 4,
                "scope": "MyClass",
                "signature": "def my_function(self, x)"
            },
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "module_level",
                "kind": "function",
                "line": 25,
                "col": 0,
                "scope": "module",
                "signature": "def module_level()"
            }
        ]
        db.insert_symbols(symbol_data)

        # Test get_all_symbols
        result = db.get_all_symbols(project_id=1)
        assert len(result) == 3

        # Verify all required fields are present
        for symbol in result:
            assert "id" in symbol
            assert "name" in symbol
            assert "kind" in symbol
            assert "file_id" in symbol
            assert "line" in symbol
            assert "col" in symbol
            assert "scope" in symbol

        # Verify ordering by id and content
        names = [sym["name"] for sym in result]
        assert names == ["MyClass", "my_function", "module_level"]

    def test_get_all_symbols_multiple_projects(self):
        """Test get_all_symbols only returns symbols for specified project."""
        db = ProjectDB(tempfile.mkdtemp())

        # Project 1 symbols
        file_id_1 = db.upsert_file(1, "file1.py", "python", "hash1")
        db.insert_symbols([
            {
                "project_id": 1,
                "file_id": file_id_1,
                "name": "symbol_1",
                "kind": "function",
                "line": 1,
                "col": 0,
                "scope": "module",
                "signature": "def symbol_1()"
            }
        ])

        # Project 2 symbols
        file_id_2 = db.upsert_file(2, "file2.py", "python", "hash2")
        db.insert_symbols([
            {
                "project_id": 2,
                "file_id": file_id_2,
                "name": "symbol_2",
                "kind": "function",
                "line": 1,
                "col": 0,
                "scope": "module",
                "signature": "def symbol_2()"
            },
            {
                "project_id": 2,
                "file_id": file_id_2,
                "name": "symbol_3",
                "kind": "class",
                "line": 5,
                "col": 0,
                "scope": "module",
                "signature": "class symbol_3"
            }
        ])

        # Get symbols for project 1
        result_1 = db.get_all_symbols(project_id=1)
        assert len(result_1) == 1
        assert result_1[0]["name"] == "symbol_1"

        # Get symbols for project 2
        result_2 = db.get_all_symbols(project_id=2)
        assert len(result_2) == 2
        names = {sym["name"] for sym in result_2}
        assert names == {"symbol_2", "symbol_3"}


class TestGetSymbolToChunksMapping:
    """Test get_symbol_to_chunks_mapping method."""

    def test_get_symbol_to_chunks_mapping_empty(self):
        """Test mapping returns empty dict when no chunks have symbols."""
        db = ProjectDB(tempfile.mkdtemp())
        result = db.get_symbol_to_chunks_mapping()
        assert result == {}

    def test_get_symbol_to_chunks_mapping_with_data(self):
        """Test mapping correctly maps symbol IDs to chunk IDs."""
        db = ProjectDB(tempfile.mkdtemp())

        # Insert test file
        file_id = db.upsert_file(
            project_id=1,
            path="test.py",
            language="python",
            file_hash="abc123"
        )

        # Insert symbols
        symbol_ids = db.insert_symbols([
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "func_a",
                "kind": "function",
                "line": 1,
                "col": 0,
                "scope": "module",
                "signature": "def func_a()"
            },
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "func_b",
                "kind": "function",
                "line": 10,
                "col": 0,
                "scope": "module",
                "signature": "def func_b()"
            }
        ])

        # Insert chunks with symbol_ids
        chunk_ids = db.insert_chunks([
            {
                "project_id": 1,
                "file_id": file_id,
                "start_line": 1,
                "end_line": 5,
                "symbol_ids": [symbol_ids[0]],
                "ast_type": "function",
                "chunk_type": "code",
                "content": "def func_a():\n    pass",
                "file_path": "test.py"
            },
            {
                "project_id": 1,
                "file_id": file_id,
                "start_line": 10,
                "end_line": 15,
                "symbol_ids": [symbol_ids[0], symbol_ids[1]],
                "ast_type": "function",
                "chunk_type": "code",
                "content": "def func_b():\n    pass",
                "file_path": "test.py"
            },
            {
                "project_id": 1,
                "file_id": file_id,
                "start_line": 20,
                "end_line": 25,
                "symbol_ids": [symbol_ids[1]],
                "ast_type": "function",
                "chunk_type": "code",
                "content": "def func_c():\n    pass",
                "file_path": "test.py"
            }
        ])

        # Test mapping
        result = db.get_symbol_to_chunks_mapping()

        # Verify structure
        assert symbol_ids[0] in result
        assert symbol_ids[1] in result

        # Symbol 0 appears in chunks 0 and 1
        assert set(result[symbol_ids[0]]) == {chunk_ids[0], chunk_ids[1]}

        # Symbol 1 appears in chunks 1 and 2
        assert set(result[symbol_ids[1]]) == {chunk_ids[1], chunk_ids[2]}

    def test_get_symbol_to_chunks_mapping_invalid_json(self):
        """Test that chunks with invalid symbol_ids JSON are skipped gracefully."""
        db = ProjectDB(tempfile.mkdtemp())

        # Insert test file
        file_id = db.upsert_file(
            project_id=1,
            path="test.py",
            language="python",
            file_hash="abc123"
        )

        # Insert a symbol
        symbol_id = db.insert_symbols([
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "func_a",
                "kind": "function",
                "line": 1,
                "col": 0,
                "scope": "module",
                "signature": "def func_a()"
            }
        ])[0]

        # Insert a chunk with valid symbol_ids
        chunk_id_1 = db.insert_chunks([
            {
                "project_id": 1,
                "file_id": file_id,
                "start_line": 1,
                "end_line": 5,
                "symbol_ids": [symbol_id],
                "ast_type": "function",
                "chunk_type": "code",
                "content": "def func_a():\n    pass",
                "file_path": "test.py"
            }
        ])[0]

        # Directly insert a chunk with invalid JSON symbol_ids (simulate corruption)
        db.conn.execute("""
            INSERT INTO chunk_meta (
                project_id, file_id, start_line, end_line, symbol_ids,
                ast_type, chunk_type, content, length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, file_id, 10, 15, "{invalid json", "function", "code", "bad", 4))
        db.conn.commit()

        # Test that mapping still works and skips invalid entry
        result = db.get_symbol_to_chunks_mapping()

        # Should have mapping for valid chunk only
        assert symbol_id in result
        assert chunk_id_1 in result[symbol_id]

    def test_get_symbol_to_chunks_mapping_none_symbol_ids(self):
        """Test that chunks with NULL symbol_ids are skipped."""
        db = ProjectDB(tempfile.mkdtemp())

        # Insert test file
        file_id = db.upsert_file(
            project_id=1,
            path="test.py",
            language="python",
            file_hash="abc123"
        )

        # Insert a symbol
        symbol_id = db.insert_symbols([
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "func_a",
                "kind": "function",
                "line": 1,
                "col": 0,
                "scope": "module",
                "signature": "def func_a()"
            }
        ])[0]

        # Insert a chunk with None symbol_ids
        db.insert_chunks([
            {
                "project_id": 1,
                "file_id": file_id,
                "start_line": 1,
                "end_line": 5,
                "symbol_ids": None,
                "ast_type": "function",
                "chunk_type": "code",
                "content": "def func_a():\n    pass",
                "file_path": "test.py"
            }
        ])

        # Insert a chunk with valid symbol_ids
        chunk_id_2 = db.insert_chunks([
            {
                "project_id": 1,
                "file_id": file_id,
                "start_line": 10,
                "end_line": 15,
                "symbol_ids": [symbol_id],
                "ast_type": "function",
                "chunk_type": "code",
                "content": "def func_b():\n    pass",
                "file_path": "test.py"
            }
        ])[0]

        # Test that mapping only includes chunks with actual symbol_ids
        result = db.get_symbol_to_chunks_mapping()

        # Should have mapping for chunk with symbols
        assert symbol_id in result
        assert chunk_id_2 in result[symbol_id]
