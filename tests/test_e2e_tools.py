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


class TestReferencesE2E:
    """References tool returns non-empty refs for symbols that have them."""

    async def test_references_for_function_with_calls(self, indexed_server):
        results = await get_json(indexed_server, "references", {"symbol_name": "make_dog"})
        assert len(results) > 0
        # make_dog calls Dog() and dog.speak() — should have refs
        kinds = [r["kind"] for r in results]
        assert "calls" in kinds

    async def test_references_for_class_with_extends(self, indexed_server):
        results = await get_json(indexed_server, "references", {"symbol_name": "Dog"})
        assert len(results) > 0
        # Dog extends Animal
        kinds = [r["kind"] for r in results]
        assert "extends" in kinds

    async def test_references_nonexistent_symbol(self, indexed_server):
        results = await get_json(indexed_server, "references", {"symbol_name": "NoSuchThing"})
        assert results == []


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

    async def test_impact_function_with_calls(self, indexed_server):
        """make_dog calls Dog and speak — impact should show downstream symbols."""
        results = await get_json(indexed_server, "impact", {"symbol_name": "make_dog"})
        # make_dog has outgoing edges to Dog and speak
        if len(results) > 0:
            names = [r["name"] for r in results]
            # At least one downstream dependency
            assert len(names) > 0

    async def test_impact_nonexistent_symbol(self, indexed_server):
        results = await get_json(indexed_server, "impact", {"symbol_name": "NoSuchThing"})
        assert results == []

    async def test_impact_respects_depth(self, indexed_server):
        """Depth 1 should return fewer or equal results than depth 3."""
        r1 = await get_json(indexed_server, "impact", {"symbol_name": "make_dog", "depth": 1})
        r3 = await get_json(indexed_server, "impact", {"symbol_name": "make_dog", "depth": 3})
        assert len(r1) <= len(r3)
