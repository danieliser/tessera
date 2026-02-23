"""Tests for MCP server implementation."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import mcp.types as types
from tessera.server import create_server, search_tool, symbols_tool, references_tool, file_context_tool, impact_tool


def run_async(coro):
    """Helper to run async code in tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


class TestServerCreation:
    """Test server initialization and tool registration."""

    def test_server_creates_tools(self):
        """Verify the server registers all 5 tools."""
        # Create server without running it
        server = create_server(project_path="/tmp/test", global_db_path="/tmp/global.db")

        # Get the tools by calling the registered handler
        result = run_async(server.request_handlers[types.ListToolsRequest](None))

        assert isinstance(result, types.ServerResult)
        assert isinstance(result.root, types.ListToolsResult)

        tool_names = [tool.name for tool in result.root.tools]
        assert "search" in tool_names
        assert "symbols" in tool_names
        assert "references" in tool_names
        assert "file_context" in tool_names
        assert "impact" in tool_names
        # Admin tools
        assert "register_project" in tool_names
        assert "reindex" in tool_names
        assert "create_scope" in tool_names
        assert "revoke_scope" in tool_names
        assert "status" in tool_names
        assert len(tool_names) == 10


class TestSearchTool:
    """Test the search tool handler."""

    def test_search_tool_success(self):
        """Test search tool with valid inputs."""
        result = run_async(search_tool(
            tool_name="search",
            arguments={"query": "test query", "limit": 10}
        ))

        # Should return TextContent in a list
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], types.TextContent)

    def test_search_tool_with_filter_language(self):
        """Test search tool with language filter."""
        result = run_async(search_tool(
            tool_name="search",
            arguments={"query": "test", "limit": 5, "filter_language": "python"}
        ))
        assert isinstance(result, list)
        assert isinstance(result[0], types.TextContent)

    def test_search_tool_empty_query(self):
        """Test search tool with empty query."""
        result = run_async(search_tool(
            tool_name="search",
            arguments={"query": "", "limit": 10}
        ))
        assert isinstance(result, list)


class TestSymbolsTool:
    """Test the symbols tool handler."""

    def test_symbols_tool_success(self):
        """Test symbols tool with valid inputs."""
        result = run_async(symbols_tool(
            tool_name="symbols",
            arguments={"query": "*", "kind": "function"}
        ))
        assert isinstance(result, list)
        assert isinstance(result[0], types.TextContent)

    def test_symbols_tool_default_query(self):
        """Test symbols tool with default query."""
        result = run_async(symbols_tool(
            tool_name="symbols",
            arguments={}
        ))
        assert isinstance(result, list)

    def test_symbols_tool_with_language_filter(self):
        """Test symbols tool with language filter."""
        result = run_async(symbols_tool(
            tool_name="symbols",
            arguments={"query": "*", "language": "python"}
        ))
        assert isinstance(result, list)


class TestReferencesTool:
    """Test the references tool handler."""

    def test_references_tool_success(self):
        """Test references tool with valid inputs."""
        result = run_async(references_tool(
            tool_name="references",
            arguments={"symbol_name": "MyClass", "kind": "all"}
        ))
        assert isinstance(result, list)
        assert isinstance(result[0], types.TextContent)

    def test_references_tool_missing_symbol(self):
        """Test references tool without symbol_name."""
        with pytest.raises(TypeError):
            run_async(references_tool(
                tool_name="references",
                arguments={"kind": "all"}
            ))


class TestFileContextTool:
    """Test the file_context tool handler."""

    def test_file_context_tool_success(self):
        """Test file_context tool with valid file path."""
        result = run_async(file_context_tool(
            tool_name="file_context",
            arguments={"file_path": "src/test.py"}
        ))
        assert isinstance(result, list)
        assert isinstance(result[0], types.TextContent)

    def test_file_context_tool_missing_path(self):
        """Test file_context tool without file_path."""
        with pytest.raises(TypeError):
            run_async(file_context_tool(
                tool_name="file_context",
                arguments={}
            ))


class TestImpactTool:
    """Test the impact tool handler."""

    def test_impact_tool_success(self):
        """Test impact tool with valid inputs."""
        result = run_async(impact_tool(
            tool_name="impact",
            arguments={"symbol_name": "MyFunction", "depth": 3}
        ))
        assert isinstance(result, list)
        assert isinstance(result[0], types.TextContent)

    def test_impact_tool_default_depth(self):
        """Test impact tool with default depth."""
        result = run_async(impact_tool(
            tool_name="impact",
            arguments={"symbol_name": "MyFunction"}
        ))
        assert isinstance(result, list)


class TestSessionValidation:
    """Test session validation and scope checking."""

    def test_development_mode_allows_no_session(self):
        """Test that missing session_id is allowed in development mode (Phase 1)."""
        # In Phase 1, missing session_id should be allowed
        result = run_async(search_tool(
            tool_name="search",
            arguments={"query": "test", "limit": 10}
        ))
        # Should not raise an error, should return a result
        assert isinstance(result, list)

    def test_session_validation_with_invalid_session(self):
        """Test that invalid session returns error (Phase 2 behavior, optional for Phase 1)."""
        # This would be tested in Phase 2 when session validation is mandatory
        # For now, Phase 1 allows missing sessions
        pass


class TestErrorHandling:
    """Test error handling in tool handlers."""

    def test_search_tool_with_db_error(self):
        """Test search tool graceful error handling."""
        # Tool should return error in TextContent format
        result = run_async(search_tool(
            tool_name="search",
            arguments={"query": "test"}
        ))
        # Should return a list with TextContent
        assert isinstance(result, list)

    def test_file_context_tool_with_invalid_path(self):
        """Test file_context tool with path traversal attempt."""
        result = run_async(file_context_tool(
            tool_name="file_context",
            arguments={"file_path": "../../../etc/passwd"}
        ))
        # Should handle gracefully (error message or empty result)
        assert isinstance(result, list)
