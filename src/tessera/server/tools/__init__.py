"""Tool registration for Tessera MCP server."""

from fastmcp import FastMCP

from ._admin import register_admin_tools
from ._analysis import register_analysis_tools
from ._collections import register_collection_tools
from ._scope import register_scope_tools
from ._search import register_search_tools
from ._symbols import register_symbol_tools


def register_tools(mcp: FastMCP) -> None:
    """Register all MCP tool handlers on the server instance."""
    register_search_tools(mcp)
    register_symbol_tools(mcp)
    register_analysis_tools(mcp)
    register_admin_tools(mcp)
    register_scope_tools(mcp)
    register_collection_tools(mcp)
