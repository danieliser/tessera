"""Scope tools: create_scope, revoke_scope."""

import asyncio
import json
import logging

from fastmcp import FastMCP

from ...auth import create_scope, revoke_scope
from .._state import _check_session, _log_audit

logger = logging.getLogger("tessera.server")


def register_scope_tools(mcp: FastMCP) -> None:
    """Register scope management tools (global scope only)."""

    @mcp.tool()
    async def create_scope_tool(agent_id: str, scope_level: str, project_ids: list[int] = None, collection_ids: list[int] = None, ttl_minutes: int = 30, session_id: str = "") -> str:
        """Create a session token for a task agent (global scope only)."""
        from .._state import _global_db

        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        project_ids = project_ids or []
        collection_ids = collection_ids or []

        if scope_level not in ("project", "collection", "global"):
            return "Error: scope_level must be 'project', 'collection', or 'global'"

        if not _global_db:
            return "Error: Global database not initialized"

        try:
            sid = await asyncio.to_thread(
                create_scope, _global_db.conn, agent_id, scope_level,
                [str(p) for p in project_ids] if project_ids is not None else [],
                [str(c) for c in collection_ids] if collection_ids is not None else [],
                ttl_minutes=ttl_minutes
            )
            result = {"session_id": sid, "agent_id": agent_id, "scope_level": scope_level, "ttl_minutes": ttl_minutes}
            _log_audit("create_scope", 1, scope_level="global")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception("create_scope error")
            _log_audit("create_scope", 0, scope_level="global")
            return f"Error: {str(e)}"

    @mcp.tool()
    async def revoke_scope_tool(agent_id: str, session_id: str = "") -> str:
        """Revoke all sessions for an agent (global scope only)."""
        from .._state import _global_db

        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        try:
            count = await asyncio.to_thread(revoke_scope, _global_db.conn, agent_id)
            result = {"agent_id": agent_id, "sessions_revoked": count}
            _log_audit("revoke_scope", count, scope_level="global")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception("revoke_scope error")
            _log_audit("revoke_scope", 0, scope_level="global")
            return f"Error: {str(e)}"
