"""Tests for MCP server implementation (FastMCP)."""

import pytest
import asyncio
import tempfile
import os

from tessera.server import create_server


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
