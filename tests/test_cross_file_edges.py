"""Tests for cross-file edge resolution."""

import pytest
import tempfile
from pathlib import Path

from tessera.db import ProjectDB
from tessera.indexer import IndexerPipeline


@pytest.fixture
def temp_project():
    """Create a temporary project with database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_db = ProjectDB(tmpdir)
        yield {
            'path': tmpdir,
            'db': project_db,
            'project_id': 1,  # Fixed project_id for testing
        }


class TestCrossFileEdgeResolution:
    """Test cross-file edge resolution."""

    def test_resolve_cross_file_edges_simple(self, temp_project):
        """Test resolving simple cross-file references."""
        db = temp_project['db']
        project_id = temp_project['project_id']

        # Create 3 files
        # File A defines helper_func
        file_a_id = db.upsert_file(project_id, 'file_a.py', 'python', 'hash_a')
        # File B defines main_func
        file_b_id = db.upsert_file(project_id, 'file_b.py', 'python', 'hash_b')
        # File C defines util_func
        file_c_id = db.upsert_file(project_id, 'file_c.py', 'python', 'hash_c')

        # Insert symbols
        # File A: helper_func
        helper_func_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_a_id,
            'name': 'helper_func',
            'kind': 'function',
            'line': 1,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # File B: main_func
        main_func_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_b_id,
            'name': 'main_func',
            'kind': 'function',
            'line': 5,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # File C: util_func
        util_func_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_c_id,
            'name': 'util_func',
            'kind': 'function',
            'line': 10,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Create unresolved refs (cross-file calls)
        # main_func calls helper_func (unresolved, to_symbol_id is None)
        ref1_id = db.insert_refs([{
            'project_id': project_id,
            'from_symbol_id': main_func_id,
            'to_symbol_id': None,
            'to_symbol_name': 'helper_func',
            'kind': 'calls',
            'context': '',
            'line': 6,
        }])[0]

        # util_func calls helper_func (unresolved)
        ref2_id = db.insert_refs([{
            'project_id': project_id,
            'from_symbol_id': util_func_id,
            'to_symbol_id': None,
            'to_symbol_name': 'helper_func',
            'kind': 'calls',
            'context': '',
            'line': 11,
        }])[0]

        # util_func calls main_func (unresolved)
        ref3_id = db.insert_refs([{
            'project_id': project_id,
            'from_symbol_id': util_func_id,
            'to_symbol_id': None,
            'to_symbol_name': 'main_func',
            'kind': 'calls',
            'context': '',
            'line': 12,
        }])[0]

        # Now resolve cross-file edges
        indexer = IndexerPipeline(temp_project['path'], project_db=db)
        indexer.project_id = project_id
        edges_created = indexer._resolve_cross_file_edges()

        # Should have created 3 edges
        assert edges_created == 3

        # Verify refs were updated
        main_func_refs = db.get_refs(symbol_id=main_func_id)
        assert len(main_func_refs) == 1
        assert main_func_refs[0]['to_symbol_id'] == helper_func_id

        util_func_refs = db.get_refs(symbol_id=util_func_id)
        assert len(util_func_refs) == 2
        ref_names = [r['to_symbol_name'] for r in util_func_refs]
        assert 'helper_func' in ref_names
        assert 'main_func' in ref_names

        # Verify edges were created
        edges = db.conn.execute(
            "SELECT * FROM edges WHERE project_id = ? ORDER BY from_id",
            (project_id,)
        ).fetchall()
        assert len(edges) >= 3

        # Check specific edges exist
        edges_dict = [dict(row) for row in edges]
        # main_func -> helper_func
        assert any(e['from_id'] == main_func_id and e['to_id'] == helper_func_id for e in edges_dict)
        # util_func -> helper_func
        assert any(e['from_id'] == util_func_id and e['to_id'] == helper_func_id for e in edges_dict)
        # util_func -> main_func
        assert any(e['from_id'] == util_func_id and e['to_id'] == main_func_id for e in edges_dict)

    def test_resolve_cross_file_edges_no_duplicates(self, temp_project):
        """Test that duplicate edges are not created."""
        db = temp_project['db']
        project_id = temp_project['project_id']

        # Create 2 files
        file_a_id = db.upsert_file(project_id, 'file_a.py', 'python', 'hash_a')
        file_b_id = db.upsert_file(project_id, 'file_b.py', 'python', 'hash_b')

        # Insert symbols
        func_a_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_a_id,
            'name': 'func_a',
            'kind': 'function',
            'line': 1,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        func_b_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_b_id,
            'name': 'func_b',
            'kind': 'function',
            'line': 5,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Create an edge manually (as if it was created during intra-file parsing)
        db.insert_edges([{
            'project_id': project_id,
            'from_id': func_a_id,
            'to_id': func_b_id,
            'type': 'calls',
            'weight': 1.0,
        }])

        # Create unresolved ref (should also be resolved)
        db.insert_refs([{
            'project_id': project_id,
            'from_symbol_id': func_a_id,
            'to_symbol_id': None,
            'to_symbol_name': 'func_b',
            'kind': 'calls',
            'context': '',
            'line': 6,
        }])

        # Count edges before resolution
        edges_before = db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE project_id = ?",
            (project_id,)
        ).fetchone()[0]

        # Resolve cross-file edges
        indexer = IndexerPipeline(temp_project['path'], project_db=db)
        indexer.project_id = project_id
        edges_created = indexer._resolve_cross_file_edges()

        # Should not create a new edge since it already exists
        assert edges_created == 0

        # Verify no duplicate edges were created
        edges_after = db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE project_id = ? AND from_id = ? AND to_id = ?",
            (project_id, func_a_id, func_b_id)
        ).fetchone()[0]
        # Should still be 1 (not 2)
        assert edges_after == 1

        # But verify the ref was still updated with the symbol_id
        ref_rows = db.conn.execute(
            "SELECT to_symbol_id FROM refs WHERE project_id = ? AND from_symbol_id = ?",
            (project_id, func_a_id)
        ).fetchall()
        assert len(ref_rows) == 1
        assert ref_rows[0][0] == func_b_id

    def test_resolve_cross_file_edges_namespaced_symbols(self, temp_project):
        """Test resolving namespaced PHP symbols."""
        db = temp_project['db']
        project_id = temp_project['project_id']

        # Create 2 files
        file_utils_id = db.upsert_file(project_id, 'utils.php', 'php', 'hash_utils')
        file_main_id = db.upsert_file(project_id, 'main.php', 'php', 'hash_main')

        # Insert PHP symbols with namespaces
        # Utility class with full namespace
        util_class_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_utils_id,
            'name': 'App\\Utils\\helper',
            'kind': 'function',
            'line': 5,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Main function that calls it
        main_func_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_main_id,
            'name': 'main',
            'kind': 'function',
            'line': 10,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Unresolved ref using short name (as PHP would parse it)
        db.insert_refs([{
            'project_id': project_id,
            'from_symbol_id': main_func_id,
            'to_symbol_id': None,
            'to_symbol_name': 'helper',  # Short name, not full namespace
            'kind': 'calls',
            'context': '',
            'line': 11,
        }])

        # Resolve cross-file edges
        indexer = IndexerPipeline(temp_project['path'], project_db=db)
        indexer.project_id = project_id
        edges_created = indexer._resolve_cross_file_edges()

        # Should resolve the short name to the full namespace symbol
        assert edges_created == 1

        # Verify ref was updated with correct symbol_id
        main_refs = db.get_refs(symbol_id=main_func_id)
        assert len(main_refs) == 1
        assert main_refs[0]['to_symbol_id'] == util_class_id

    def test_resolve_cross_file_edges_ambiguous_names(self, temp_project):
        """Test that ambiguous symbol names are skipped."""
        db = temp_project['db']
        project_id = temp_project['project_id']

        # Create 3 files
        file_a_id = db.upsert_file(project_id, 'file_a.py', 'python', 'hash_a')
        file_b_id = db.upsert_file(project_id, 'file_b.py', 'python', 'hash_b')
        file_c_id = db.upsert_file(project_id, 'file_c.py', 'python', 'hash_c')

        # Insert symbols with same name (ambiguous)
        helper_v1_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_a_id,
            'name': 'helper',
            'kind': 'function',
            'line': 1,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        helper_v2_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_b_id,
            'name': 'helper',
            'kind': 'function',
            'line': 5,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Insert a caller
        caller_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_c_id,
            'name': 'caller',
            'kind': 'function',
            'line': 10,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Unresolved ref to ambiguous name
        db.insert_refs([{
            'project_id': project_id,
            'from_symbol_id': caller_id,
            'to_symbol_id': None,
            'to_symbol_name': 'helper',
            'kind': 'calls',
            'context': '',
            'line': 11,
        }])

        # Resolve cross-file edges
        indexer = IndexerPipeline(temp_project['path'], project_db=db)
        indexer.project_id = project_id
        edges_created = indexer._resolve_cross_file_edges()

        # Should skip ambiguous reference (not create edge)
        # Implementation prefers functions > classes > methods, so with both functions,
        # they're still ambiguous
        assert edges_created == 0

    def test_resolve_cross_file_edges_preference_function_over_class(self, temp_project):
        """Test that function symbols are preferred over classes when ambiguous."""
        db = temp_project['db']
        project_id = temp_project['project_id']

        # Create 3 files
        file_a_id = db.upsert_file(project_id, 'file_a.py', 'python', 'hash_a')
        file_b_id = db.upsert_file(project_id, 'file_b.py', 'python', 'hash_b')
        file_c_id = db.upsert_file(project_id, 'file_c.py', 'python', 'hash_c')

        # Insert a class
        helper_class_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_a_id,
            'name': 'helper',
            'kind': 'class',
            'line': 1,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Insert a function with same name
        helper_func_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_b_id,
            'name': 'helper',
            'kind': 'function',
            'line': 5,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Insert a caller
        caller_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_c_id,
            'name': 'caller',
            'kind': 'function',
            'line': 10,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Unresolved ref
        db.insert_refs([{
            'project_id': project_id,
            'from_symbol_id': caller_id,
            'to_symbol_id': None,
            'to_symbol_name': 'helper',
            'kind': 'calls',
            'context': '',
            'line': 11,
        }])

        # Resolve cross-file edges
        indexer = IndexerPipeline(temp_project['path'], project_db=db)
        indexer.project_id = project_id
        edges_created = indexer._resolve_cross_file_edges()

        # Should prefer function over class
        assert edges_created == 1

        # Verify the edge points to the function
        edges = db.conn.execute(
            "SELECT * FROM edges WHERE from_id = ? AND to_id = ?",
            (caller_id, helper_func_id)
        ).fetchall()
        assert len(edges) == 1

    def test_resolve_cross_file_edges_unresolved_remains_unresolved(self, temp_project):
        """Test that unresolvable refs remain unresolved."""
        db = temp_project['db']
        project_id = temp_project['project_id']

        # Create 2 files
        file_a_id = db.upsert_file(project_id, 'file_a.py', 'python', 'hash_a')
        file_b_id = db.upsert_file(project_id, 'file_b.py', 'python', 'hash_b')

        # Insert a caller
        caller_id = db.insert_symbols([{
            'project_id': project_id,
            'file_id': file_a_id,
            'name': 'caller',
            'kind': 'function',
            'line': 1,
            'col': 0,
            'scope': '',
            'signature': '',
        }])[0]

        # Unresolved ref to non-existent symbol
        db.insert_refs([{
            'project_id': project_id,
            'from_symbol_id': caller_id,
            'to_symbol_id': None,
            'to_symbol_name': 'nonexistent_func',
            'kind': 'calls',
            'context': '',
            'line': 2,
        }])

        # Resolve cross-file edges
        indexer = IndexerPipeline(temp_project['path'], project_db=db)
        indexer.project_id = project_id
        edges_created = indexer._resolve_cross_file_edges()

        # Should not create any edges
        assert edges_created == 0

        # Verify ref is still unresolved
        caller_refs = db.get_refs(symbol_id=caller_id)
        assert len(caller_refs) == 1
        assert caller_refs[0]['to_symbol_id'] is None
