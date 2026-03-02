"""Collection tools: create, add_to, list, delete collections."""

import asyncio
import json
import logging

from fastmcp import FastMCP

from .._state import _check_session, _log_audit

logger = logging.getLogger("tessera.server")


def register_collection_tools(mcp: FastMCP) -> None:
    """Register collection management tools (global scope only)."""

    @mcp.tool()
    async def create_collection_tool(name: str, project_ids: list[int] = None, session_id: str = "") -> str:
        """Create a new collection (global scope only)."""
        from .._state import _global_db

        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        project_ids = project_ids or []

        try:
            coll_id = await asyncio.to_thread(_global_db.create_collection, name, project_ids)
            collection = await asyncio.to_thread(_global_db.get_collection, coll_id)
            _log_audit("create_collection_tool", 1, scope_level="global")
            return json.dumps(collection, indent=2)
        except Exception as e:
            logger.exception("create_collection_tool error")
            _log_audit("create_collection_tool", 0, scope_level="global")
            return f"Error: {str(e)}"

    @mcp.tool()
    async def add_to_collection_tool(collection_id: int, project_id: int, session_id: str = "") -> str:
        """Add a project to a collection (global scope only)."""
        from .._state import _global_db

        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        try:
            await asyncio.to_thread(_global_db.add_project_to_collection, collection_id, project_id)
            collection = await asyncio.to_thread(_global_db.get_collection, collection_id)
            _log_audit("add_to_collection_tool", 1, scope_level="global")
            return json.dumps(collection, indent=2)
        except Exception as e:
            logger.exception("add_to_collection_tool error")
            _log_audit("add_to_collection_tool", 0, scope_level="global")
            return f"Error: {str(e)}"

    @mcp.tool()
    async def list_collections_tool(session_id: str = "") -> str:
        """List all collections (global scope only)."""
        from .._state import _global_db

        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        try:
            collections = await asyncio.to_thread(_global_db.list_collections)
            result = {"collections": collections}
            _log_audit("list_collections_tool", len(collections), scope_level="global")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception("list_collections_tool error")
            _log_audit("list_collections_tool", 0, scope_level="global")
            return f"Error: {str(e)}"

    @mcp.tool()
    async def delete_collection_tool(collection_id: int, session_id: str = "") -> str:
        """Delete a collection (global scope only)."""
        from .._state import _global_db

        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        try:
            await asyncio.to_thread(_global_db.delete_collection, collection_id)
            result = {"deleted": True, "collection_id": collection_id}
            _log_audit("delete_collection_tool", 1, scope_level="global")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception("delete_collection_tool error")
            _log_audit("delete_collection_tool", 0, scope_level="global")
            return f"Error: {str(e)}"
