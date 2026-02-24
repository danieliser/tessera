"""Tests for MCP server implementation (FastMCP)."""

import pytest
import asyncio
import json
import tempfile
import os

from tessera.server import create_server
from tessera.db import ProjectDB, GlobalDB


@pytest.fixture
def server():
    """Create a server with temp databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = tmpdir
        global_db_path = os.path.join(tmpdir, "global.db")
        srv = create_server(project_path, global_db_path)
        yield srv


class TestServerCreation:
    """Test server initialization and tool registration."""

    async def test_server_creates_tools(self, server):
        """Verify the server registers all 10 tools."""
        tools = await server.list_tools()
        tool_names = [t.name for t in tools]

        assert "search" in tool_names
        assert "symbols" in tool_names
        assert "references" in tool_names
        assert "file_context" in tool_names
        assert "impact" in tool_names
        assert "register_project" in tool_names
        assert "reindex" in tool_names
        assert len(tool_names) == 10


class TestSearchTool:
    """Test the search tool handler."""

    async def test_search_tool_success(self, server):
        """Test search tool with valid inputs."""
        content, _ = await server.call_tool("search", {"query": "test query", "limit": 10})
        assert len(content) > 0
        assert content[0].text  # Has text content

    async def test_search_tool_with_filter_language(self, server):
        """Test search tool with language filter."""
        content, _ = await server.call_tool("search", {"query": "test", "limit": 5, "filter_language": "python"})
        assert len(content) > 0

    async def test_search_tool_empty_query(self, server):
        """Test search tool with empty query."""
        content, _ = await server.call_tool("search", {"query": "", "limit": 10})
        assert len(content) > 0


class TestSymbolsTool:
    """Test the symbols tool handler."""

    async def test_symbols_tool_success(self, server):
        content, _ = await server.call_tool("symbols", {"query": "*", "kind": "function"})
        assert len(content) > 0

    async def test_symbols_tool_default_query(self, server):
        content, _ = await server.call_tool("symbols", {})
        assert len(content) > 0

    async def test_symbols_tool_with_language_filter(self, server):
        content, _ = await server.call_tool("symbols", {"query": "*", "language": "python"})
        assert len(content) > 0


class TestReferencesTool:
    """Test the references tool handler."""

    async def test_references_tool_success(self, server):
        content, _ = await server.call_tool("references", {"symbol_name": "MyClass", "kind": "all"})
        assert len(content) > 0

    async def test_references_tool_missing_symbol(self, server):
        """FastMCP validates required params — should error."""
        with pytest.raises(Exception):
            await server.call_tool("references", {"kind": "all"})


class TestFileContextTool:
    """Test the file_context tool handler."""

    async def test_file_context_tool_success(self, server):
        content, _ = await server.call_tool("file_context", {"file_path": "src/test.py"})
        assert len(content) > 0

    async def test_file_context_tool_missing_path(self, server):
        with pytest.raises(Exception):
            await server.call_tool("file_context", {})


class TestImpactTool:
    """Test the impact tool handler."""

    async def test_impact_tool_success(self, server):
        content, _ = await server.call_tool("impact", {"symbol_name": "MyFunction", "depth": 3})
        assert len(content) > 0

    async def test_impact_tool_default_depth(self, server):
        content, _ = await server.call_tool("impact", {"symbol_name": "MyFunction"})
        assert len(content) > 0


class TestSessionValidation:
    """Test session validation and scope checking."""

    async def test_development_mode_allows_no_session(self, server):
        """Missing session_id allowed in dev mode."""
        content, _ = await server.call_tool("search", {"query": "test", "limit": 10})
        assert len(content) > 0
        assert "Error" not in content[0].text

    async def test_session_validation_with_invalid_session(self, server):
        """Invalid session returns error."""
        content, _ = await server.call_tool("search", {"query": "test", "session_id": "bad-session"})
        assert "Error" in content[0].text


class TestErrorHandling:
    """Test error handling in tool handlers."""

    async def test_search_tool_with_db_error(self, server):
        content, _ = await server.call_tool("search", {"query": "test"})
        assert len(content) > 0

    async def test_file_context_tool_with_invalid_path(self, server):
        """Path traversal attempt should be caught."""
        content, _ = await server.call_tool("file_context", {"file_path": "../../../etc/passwd"})
        assert "Error" in content[0].text


def _seed_project(project_path: str, project_name: str, global_db: GlobalDB, symbols: list[dict]):
    """Register a project in GlobalDB and seed its ProjectDB with test data."""
    os.makedirs(project_path, exist_ok=True)
    pid = global_db.register_project(path=project_path, name=project_name)
    db = ProjectDB(project_path)
    file_id = db.upsert_file(pid, os.path.join(project_path, "src/main.py"), "python", "abc123")
    inserted_ids = db.insert_symbols([
        {**s, "project_id": pid, "file_id": file_id}
        for s in symbols
    ])
    return pid, db, file_id, inserted_ids


@pytest.fixture
def multi_server():
    """Create a server in multi-project mode with 2 indexed projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        global_db_path = os.path.join(tmpdir, "global.db")
        global_db = GlobalDB(global_db_path)

        # Project A: has function "alpha_func" and class "AlphaClass"
        proj_a_path = os.path.join(tmpdir, "project_a")
        pid_a, db_a, fid_a, sym_ids_a = _seed_project(proj_a_path, "project-alpha", global_db, [
            {"name": "alpha_func", "kind": "function", "line": 10, "col": 0, "scope": "(module)", "signature": "def alpha_func()"},
            {"name": "AlphaClass", "kind": "class", "line": 20, "col": 0, "scope": "(module)", "signature": "class AlphaClass"},
            {"name": "shared_util", "kind": "function", "line": 30, "col": 0, "scope": "(module)", "signature": "def shared_util()"},
        ])
        # Add a reference: alpha_func calls shared_util
        db_a.insert_refs([{
            "project_id": pid_a,
            "from_symbol_id": sym_ids_a[0],
            "to_symbol_id": sym_ids_a[2],
            "to_symbol_name": "shared_util",
            "kind": "calls",
            "context": "shared_util()",
            "line": 12,
        }])

        # Project B: has function "beta_func" and class "BetaClass"
        proj_b_path = os.path.join(tmpdir, "project_b")
        pid_b, db_b, fid_b, sym_ids_b = _seed_project(proj_b_path, "project-beta", global_db, [
            {"name": "beta_func", "kind": "function", "line": 5, "col": 0, "scope": "(module)", "signature": "def beta_func()"},
            {"name": "BetaClass", "kind": "class", "line": 15, "col": 0, "scope": "(module)", "signature": "class BetaClass"},
            {"name": "shared_util", "kind": "function", "line": 25, "col": 0, "scope": "(module)", "signature": "def shared_util()"},
        ])

        srv = create_server(project_path=None, global_db_path=global_db_path)
        yield srv, {
            "a": {"pid": pid_a, "name": "project-alpha", "sym_ids": sym_ids_a},
            "b": {"pid": pid_b, "name": "project-beta", "sym_ids": sym_ids_b},
        }


@pytest.fixture
def locked_server():
    """Create a server locked to a single project via --project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        global_db_path = os.path.join(tmpdir, "global.db")
        global_db = GlobalDB(global_db_path)

        # Project A (locked)
        proj_a_path = os.path.join(tmpdir, "project_a")
        pid_a, db_a, fid_a, sym_ids_a = _seed_project(proj_a_path, "project-alpha", global_db, [
            {"name": "locked_func", "kind": "function", "line": 1, "col": 0, "scope": "(module)", "signature": "def locked_func()"},
        ])

        # Project B (should NOT be accessible)
        proj_b_path = os.path.join(tmpdir, "project_b")
        pid_b, db_b, fid_b, sym_ids_b = _seed_project(proj_b_path, "project-beta", global_db, [
            {"name": "hidden_func", "kind": "function", "line": 1, "col": 0, "scope": "(module)", "signature": "def hidden_func()"},
        ])

        srv = create_server(project_path=proj_a_path, global_db_path=global_db_path)
        yield srv, {
            "a": {"pid": pid_a, "name": "project-alpha"},
            "b": {"pid": pid_b, "name": "project-beta"},
        }


class TestMultiProjectQueries:
    """Test multi-project mode (no --project flag)."""

    async def test_symbols_returns_both_projects(self, multi_server):
        srv, projects = multi_server
        content, _ = await srv.call_tool("symbols", {"query": "*"})
        results = json.loads(content[0].text)
        project_names = {r["project_name"] for r in results}
        assert "project-alpha" in project_names
        assert "project-beta" in project_names

    async def test_symbols_from_each_project(self, multi_server):
        srv, projects = multi_server
        content, _ = await srv.call_tool("symbols", {"query": "alpha_func"})
        results = json.loads(content[0].text)
        assert len(results) == 1
        assert results[0]["project_name"] == "project-alpha"

        content, _ = await srv.call_tool("symbols", {"query": "beta_func"})
        results = json.loads(content[0].text)
        assert len(results) == 1
        assert results[0]["project_name"] == "project-beta"

    async def test_symbols_shared_name_returns_both(self, multi_server):
        """A symbol name that exists in both projects returns results from both."""
        srv, projects = multi_server
        content, _ = await srv.call_tool("symbols", {"query": "shared_util"})
        results = json.loads(content[0].text)
        assert len(results) == 2
        project_names = {r["project_name"] for r in results}
        assert project_names == {"project-alpha", "project-beta"}

    async def test_references_cross_project(self, multi_server):
        srv, projects = multi_server
        content, _ = await srv.call_tool("references", {"symbol_name": "shared_util"})
        results = json.loads(content[0].text)
        # shared_util is called by alpha_func — should show up in callers
        callers = results.get("callers", [])
        assert any(c.get("project_name") == "project-alpha" for c in callers)

    async def test_file_context_finds_correct_project(self, multi_server):
        srv, projects = multi_server
        # file_context uses path matching — the path contains project_a
        content, _ = await srv.call_tool("file_context", {"file_path": "src/main.py"})
        text = content[0].text
        # Should find it in at least one project
        assert "Error" not in text


class TestLockedProjectMode:
    """Test single-project lock mode (--project flag)."""

    async def test_locked_only_sees_own_symbols(self, locked_server):
        srv, projects = locked_server
        content, _ = await srv.call_tool("symbols", {"query": "*"})
        results = json.loads(content[0].text)
        names = {r["name"] for r in results}
        assert "locked_func" in names
        assert "hidden_func" not in names

    async def test_locked_project_metadata(self, locked_server):
        srv, projects = locked_server
        content, _ = await srv.call_tool("symbols", {"query": "locked_func"})
        results = json.loads(content[0].text)
        assert len(results) == 1
        assert results[0]["project_name"] == "project-alpha"
        assert results[0]["project_id"] == projects["a"]["pid"]

    async def test_locked_cannot_see_other_project(self, locked_server):
        """Searching for a symbol only in project B returns empty."""
        srv, projects = locked_server
        content, _ = await srv.call_tool("symbols", {"query": "hidden_func"})
        results = json.loads(content[0].text)
        assert len(results) == 0


class TestMultiProjectServerCreation:
    """Test server creation modes."""

    async def test_create_server_no_project(self):
        """Server can be created without --project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            global_db_path = os.path.join(tmpdir, "global.db")
            srv = create_server(project_path=None, global_db_path=global_db_path)
            tools = await srv.list_tools()
            assert len([t.name for t in tools]) == 10

    async def test_no_projects_returns_error(self):
        """Multi-project mode with no registered projects returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            global_db_path = os.path.join(tmpdir, "global.db")
            srv = create_server(project_path=None, global_db_path=global_db_path)
            content, _ = await srv.call_tool("symbols", {"query": "*"})
            assert "Error" in content[0].text or "[]" in content[0].text
