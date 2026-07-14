"""Tool registration for Tessera MCP server."""

from fastmcp import FastMCP

from ._admin import register_admin_tools
from ._analysis import register_analysis_tools
from ._collections import register_collection_tools
from ._events import register_event_tools
from ._explore import register_explore_tools
from ._scope import register_scope_tools
from ._search import register_search_tools
from ._symbols import register_symbol_tools


def register_tools(mcp: FastMCP, tool_profile: str = "advanced") -> None:
    """Register the compact or advanced MCP tool profile."""
    register_explore_tools(mcp)
    if tool_profile == "compact":
        return
    if tool_profile != "advanced":
        raise ValueError("tool_profile must be 'compact' or 'advanced'")
    register_search_tools(mcp)
    register_symbol_tools(mcp)
    register_analysis_tools(mcp)
    register_event_tools(mcp)
    register_admin_tools(mcp)
    register_scope_tools(mcp)
    register_collection_tools(mcp)
