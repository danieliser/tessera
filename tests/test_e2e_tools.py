"""End-to-end tests: index real code, then verify each MCP tool returns correct data.

These tests catch wiring bugs between server tool handlers and DB queries
by asserting on actual content, not just response shape.
"""

import json
import os
import tempfile

import pytest

from tessera.server import create_server
from tessera.indexer import IndexerPipeline


# Sample Python file with known symbols and relationships
SAMPLE_CODE = '''\
"""Sample module for testing."""

import os
from pathlib import Path


class Animal:
    """Base class."""

    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return ""


class Dog(Animal):
    """A dog that speaks."""

    def speak(self) -> str:
        return f"{self.name} says woof"

    def fetch(self, item: str) -> str:
        return f"{self.name} fetches {item}"


def make_dog(name: str) -> Dog:
    """Factory function that creates a Dog."""
    dog = Dog(name)
    dog.speak()
    return dog
'''

SAMPLE_CODE_2 = '''\
"""Helper module."""

from pathlib import Path


def read_file(path: str) -> str:
    """Read a file and return contents."""
    p = Path(path)
    return p.read_text()


def list_files(directory: str) -> list:
    """List files in a directory."""
    p = Path(directory)
    return [str(f) for f in p.iterdir()]
'''

# Cross-file: imports from animals.py and calls its symbols
SAMPLE_CODE_3 = '''\
"""App module that uses animals."""

from animals import Dog, make_dog


def run_app():
    """Create a dog and make it speak."""
    dog = make_dog("Rex")
    result = dog.fetch("ball")
    return result
'''

# Nested functions: outer_func contains inner_func
SAMPLE_CODE_4 = '''\
"""Module with nested function definitions."""


def outer_func():
    """Outer function that contains a nested function."""

    def inner_func():
        """Inner function defined inside outer_func."""
        return "inner"

    return inner_func()
'''


@pytest.fixture
def indexed_server():
    """Create a server with indexed sample code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sample files
        src_dir = os.path.join(tmpdir, "src")
        os.makedirs(src_dir)

        with open(os.path.join(src_dir, "animals.py"), "w") as f:
            f.write(SAMPLE_CODE)

        with open(os.path.join(src_dir, "helpers.py"), "w") as f:
            f.write(SAMPLE_CODE_2)

        with open(os.path.join(src_dir, "app.py"), "w") as f:
            f.write(SAMPLE_CODE_3)

        with open(os.path.join(src_dir, "nested.py"), "w") as f:
            f.write(SAMPLE_CODE_4)

        # Index
        global_db_path = os.path.join(tmpdir, "global.db")
        pipeline = IndexerPipeline(tmpdir)
        pipeline.index_project()

        # Create server pointing at indexed data
        server = create_server(tmpdir, global_db_path)
        yield server


async def get_json(server, tool_name, args=None):
    """Call a tool and parse the JSON result."""
    content, _ = await server.call_tool(tool_name, args or {})
    return json.loads(content[0].text)


class TestServerStartup:
    """Smoke tests: server must start without errors."""

    def test_fresh_db_startup(self):
        """Server starts cleanly with a fresh (empty) database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            global_db_path = os.path.join(tmpdir, "global.db")
            server = create_server(tmpdir, global_db_path)
            assert server is not None

    def test_existing_db_migration(self):
        """Server starts against a pre-existing DB missing new columns."""
        import sqlite3
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an old-schema DB (refs without to_symbol_name)
            tessera_dir = os.path.join(tmpdir, ".tessera")
            os.makedirs(tessera_dir)
            db_path = os.path.join(tessera_dir, "index.db")
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE files (
                    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL,
                    path TEXT NOT NULL, language TEXT, hash TEXT,
                    index_status TEXT DEFAULT 'pending', indexed_at TIMESTAMP,
                    UNIQUE(project_id, path)
                )
            """)
            conn.execute("""
                CREATE TABLE symbols (
                    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL,
                    file_id INTEGER NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL,
                    line INTEGER, col INTEGER, scope TEXT, signature TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE refs (
                    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL,
                    from_symbol_id INTEGER NOT NULL, to_symbol_id INTEGER,
                    kind TEXT NOT NULL, context TEXT, line INTEGER
                )
            """)
            conn.commit()
            conn.close()

            # Server should start and migrate without error
            global_db_path = os.path.join(tmpdir, "global.db")
            server = create_server(tmpdir, global_db_path)
            assert server is not None


class TestSearchE2E:
    """Search returns relevant indexed content."""

    async def test_search_finds_code(self, indexed_server):
        results = await get_json(indexed_server, "search", {"query": "dog speak woof"})
        assert len(results) > 0
        # Should find the Dog class or make_dog function
        texts = " ".join(r.get("content", "") for r in results)
        assert "woof" in texts

    async def test_search_finds_different_file(self, indexed_server):
        results = await get_json(indexed_server, "search", {"query": "read file contents"})
        assert len(results) > 0
        texts = " ".join(r.get("content", "") for r in results)
        assert "read_text" in texts or "read_file" in texts

    async def test_search_empty_query_returns_something(self, indexed_server):
        """Even empty query should not crash."""
        results = await get_json(indexed_server, "search", {"query": ""})
        assert isinstance(results, list)


class TestSymbolsE2E:
    """Symbols tool returns indexed symbols with correct metadata."""

    async def test_symbols_finds_class(self, indexed_server):
        results = await get_json(indexed_server, "symbols", {"query": "Dog"})
        assert len(results) > 0
        names = [s["name"] for s in results]
        assert "Dog" in names

    async def test_symbols_finds_function(self, indexed_server):
        results = await get_json(indexed_server, "symbols", {"query": "make_dog"})
        assert len(results) > 0
        assert results[0]["name"] == "make_dog"
        assert results[0]["kind"] == "function"

    async def test_symbols_filter_by_kind(self, indexed_server):
        results = await get_json(indexed_server, "symbols", {"query": "*", "kind": "class"})
        assert len(results) > 0
        kinds = {s["kind"] for s in results}
        assert kinds == {"class"}

    async def test_symbols_returns_signature(self, indexed_server):
        results = await get_json(indexed_server, "symbols", {"query": "make_dog"})
        assert len(results) > 0
        sig = results[0].get("signature", "")
        assert "name" in sig  # should contain parameter name

    async def test_symbols_substring_fallback(self, indexed_server):
        """Query 'dog' (no wildcards, no exact match) should fall back to substring and find make_dog, Dog."""
        results = await get_json(indexed_server, "symbols", {"query": "dog"})
        assert len(results) > 0
        names = [s["name"].lower() for s in results]
        assert any("dog" in n for n in names)

    async def test_symbols_exact_match_preferred(self, indexed_server):
        """Exact match 'Dog' should return Dog directly (not fall back)."""
        results = await get_json(indexed_server, "symbols", {"query": "Dog"})
        assert len(results) > 0
        assert results[0]["name"] == "Dog"


class TestReferencesE2E:
    """References tool returns outgoing refs and callers."""

    async def test_references_outgoing_calls(self, indexed_server):
        results = await get_json(indexed_server, "references", {"symbol_name": "make_dog"})
        assert "outgoing" in results
        assert len(results["outgoing"]) > 0
        # make_dog calls Dog() and dog.speak()
        kinds = [r["kind"] for r in results["outgoing"]]
        assert "calls" in kinds

    async def test_references_outgoing_extends(self, indexed_server):
        results = await get_json(indexed_server, "references", {"symbol_name": "Dog"})
        assert "outgoing" in results
        assert len(results["outgoing"]) > 0
        kinds = [r["kind"] for r in results["outgoing"]]
        assert "extends" in kinds

    async def test_references_callers_cross_file(self, indexed_server):
        """run_app calls make_dog — make_dog should show run_app as a caller."""
        results = await get_json(indexed_server, "references", {"symbol_name": "make_dog"})
        assert "callers" in results
        caller_names = [c["name"] for c in results["callers"]]
        assert "run_app" in caller_names

    async def test_references_callers_same_file(self, indexed_server):
        """Dog is called by make_dog — should appear as caller."""
        results = await get_json(indexed_server, "references", {"symbol_name": "Dog"})
        assert "callers" in results
        caller_names = [c["name"] for c in results["callers"]]
        assert "make_dog" in caller_names

    async def test_references_nonexistent_symbol(self, indexed_server):
        results = await get_json(indexed_server, "references", {"symbol_name": "NoSuchThing"})
        assert results["outgoing"] == []
        assert results["callers"] == []


class TestFileContextE2E:
    """File context returns structural summary with symbols and refs."""

    async def test_file_context_has_symbols(self, indexed_server):
        result = await get_json(indexed_server, "file_context", {"file_path": "src/animals.py"})
        assert "symbols" in result
        assert len(result["symbols"]) > 0
        names = [s["name"] for s in result["symbols"]]
        assert "Animal" in names
        assert "Dog" in names
        assert "make_dog" in names

    async def test_file_context_has_refs(self, indexed_server):
        result = await get_json(indexed_server, "file_context", {"file_path": "src/animals.py"})
        assert "references" in result
        assert len(result["references"]) > 0

    async def test_file_context_has_file_metadata(self, indexed_server):
        result = await get_json(indexed_server, "file_context", {"file_path": "src/animals.py"})
        assert "file" in result
        assert result["file"]["language"] == "python"
        assert result["file"]["path"] == "src/animals.py"

    async def test_file_context_nonexistent_file(self, indexed_server):
        result = await get_json(indexed_server, "file_context", {"file_path": "src/nope.py"})
        assert result is None

    async def test_file_context_path_traversal_blocked(self, indexed_server):
        content, _ = await indexed_server.call_tool("file_context", {"file_path": "../../etc/passwd"})
        assert "Error" in content[0].text


class TestImpactE2E:
    """Impact tool traces the dependency graph."""

    async def test_impact_backward_callers(self, indexed_server):
        """Impact of make_dog = what depends on it. run_app calls make_dog."""
        results = await get_json(indexed_server, "impact", {"symbol_name": "make_dog"})
        names = [r["name"] for r in results]
        assert len(names) > 0
        # run_app calls make_dog, so run_app is affected if make_dog changes
        assert "run_app" in names

    async def test_impact_transitive(self, indexed_server):
        """Impact of Dog: make_dog calls Dog(), run_app calls make_dog — transitive."""
        results = await get_json(indexed_server, "impact", {"symbol_name": "Dog", "depth": 2})
        names = [r["name"] for r in results]
        # make_dog directly calls Dog
        assert "make_dog" in names
        # run_app transitively depends via make_dog
        assert "run_app" in names

    async def test_impact_leaf_symbol(self, indexed_server):
        """run_app is a leaf — nothing calls it, so impact is empty or module-only."""
        results = await get_json(indexed_server, "impact", {"symbol_name": "run_app"})
        names = [r["name"] for r in results]
        # run_app is not called by other functions, so no function-level callers
        assert "make_dog" not in names

    async def test_impact_nested_scope(self, indexed_server):
        """outer_func contains inner_func — impact should find inner_func via containment edge."""
        results = await get_json(indexed_server, "impact", {"symbol_name": "outer_func"})
        names = [r["name"] for r in results]
        assert "inner_func" in names

    async def test_impact_nonexistent_symbol(self, indexed_server):
        results = await get_json(indexed_server, "impact", {"symbol_name": "NoSuchThing"})
        assert results == []

    async def test_impact_respects_depth(self, indexed_server):
        """Depth 1 should return fewer or equal results than depth 3."""
        r1 = await get_json(indexed_server, "impact", {"symbol_name": "Dog", "depth": 1})
        r3 = await get_json(indexed_server, "impact", {"symbol_name": "Dog", "depth": 3})
        assert len(r1) <= len(r3)


class TestAdminTools:
    """Tests for admin/global-scope tools."""

    @pytest.mark.asyncio
    async def test_status_returns_projects(self, indexed_server):
        """Status tool should return project info."""
        result = await get_json(indexed_server, "status", {})
        assert "projects" in result or "error" not in str(result).lower()

    @pytest.mark.asyncio
    async def test_create_scope_valid(self, indexed_server):
        """Create a project-scope session token."""
        result = await get_json(indexed_server, "create_scope_tool", {
            "agent_id": "test-agent",
            "scope_level": "project",
        })
        assert "session_id" in result
        assert result["agent_id"] == "test-agent"
        assert result["scope_level"] == "project"

    @pytest.mark.asyncio
    async def test_create_scope_invalid_level(self, indexed_server):
        """Invalid scope level should return error."""
        content, _ = await indexed_server.call_tool("create_scope_tool", {
            "agent_id": "test-agent",
            "scope_level": "invalid",
        })
        assert "Error" in content[0].text

    @pytest.mark.asyncio
    async def test_revoke_scope(self, indexed_server):
        """Revoke sessions for an agent."""
        # First create a session
        result = await get_json(indexed_server, "create_scope_tool", {
            "agent_id": "test-revoke",
            "scope_level": "project",
        })
        assert "session_id" in result

        # Then revoke
        revoke_result = await get_json(indexed_server, "revoke_scope_tool", {
            "agent_id": "test-revoke",
        })
        assert "revoked" in str(revoke_result).lower() or "agent_id" in revoke_result
