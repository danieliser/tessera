"""MCP server implementation for Tessera.

Exposes 10 tools via stdio transport:
  Core tools (project scope):
  - search: Hybrid semantic + keyword search
  - symbols: List/query symbols
  - references: Find all references to a symbol
  - file_context: Structural summary of a file
  - impact: What breaks if I change this symbol?

  Admin tools (global scope only):
  - register_project: Register a new project for indexing
  - reindex: Trigger re-indexing of a project
  - create_scope: Create a session token for a task agent
  - revoke_scope: Revoke an agent's session
  - status: Get system status

Session validation and scope gating is handled here.
Audit logging tracks: agent_id, scope_level, tool_called, result_count, timestamp.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from .db import ProjectDB, GlobalDB
from .auth import (
    create_scope, revoke_scope, validate_session, ScopeInfo,
    SessionNotFoundError, SessionExpiredError
)
from .indexer import IndexerPipeline

# Scope level hierarchy for comparison
_SCOPE_LEVELS = {"project": 0, "collection": 1, "global": 2}

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger(__name__ + ".audit")

# Global database references (set during lifespan)
_project_db: Optional[ProjectDB] = None
_global_db: Optional[GlobalDB] = None


def _log_audit(tool_name: str, result_count: int, agent_id: str = "dev", scope_level: str = "project"):
    """Log audit event to both Python logger and GlobalDB."""
    audit_logger.info(
        "tool_call",
        extra={
            "agent_id": agent_id,
            "scope_level": scope_level,
            "tool_called": tool_name,
            "result_count": result_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    if _global_db:
        try:
            _global_db.insert_audit(agent_id, scope_level, tool_name, result_count)
        except Exception:
            logger.warning("Failed to persist audit log to database")


def _check_session(arguments: dict[str, Any], required_level: str = "project") -> tuple[Optional[ScopeInfo], Optional[str]]:
    """Validate session from tool arguments.

    Returns:
        (scope_info, None) on success, or (None, error_message) on failure.
        If no session_id provided, returns (None, None) for dev mode fallback.
    """
    session_id = arguments.get("session_id")
    if not session_id:
        return None, None  # Dev mode

    if not _global_db:
        return None, "Error: Global database not initialized"

    try:
        scope = validate_session(_global_db.conn, session_id)
    except SessionNotFoundError:
        return None, "Error: Invalid session"
    except SessionExpiredError:
        return None, "Error: Session expired"

    if _SCOPE_LEVELS.get(scope.level, -1) < _SCOPE_LEVELS.get(required_level, 0):
        return None, f"Error: Insufficient scope. Required: {required_level}, have: {scope.level}"

    return scope, None


def create_server(project_path: str, global_db_path: str) -> FastMCP:
    """Create and configure the MCP server."""
    global _project_db, _global_db

    # Initialize databases
    Path(global_db_path).parent.mkdir(parents=True, exist_ok=True)

    _project_db = ProjectDB(project_path)
    _global_db = GlobalDB(global_db_path)

    mcp = FastMCP("tessera")

    # --- Core tools (project scope) ---

    @mcp.tool()
    async def search(query: str, limit: int = 10, filter_language: str = "", session_id: str = "") -> str:
        """Hybrid semantic + keyword search across indexed codebase."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        if not _project_db:
            _log_audit("search", 0, agent_id=agent_id)
            return "Error: Project database not initialized"

        try:
            results = await asyncio.to_thread(_project_db.keyword_search, query, limit)
            _log_audit("search", len(results), agent_id=agent_id)
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.exception("Search tool error")
            _log_audit("search", 0, agent_id=agent_id)
            return f"Error during search: {str(e)}"

    @mcp.tool()
    async def symbols(query: str = "*", kind: str = "", language: str = "", session_id: str = "") -> str:
        """List and query symbols (functions, classes, imports) in the codebase."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        if not _project_db:
            _log_audit("symbols", 0, agent_id=agent_id)
            return "Error: Project database not initialized"

        try:
            results = await asyncio.to_thread(
                _project_db.lookup_symbols, query,
                kind or None, language or None
            )
            _log_audit("symbols", len(results), agent_id=agent_id)
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.exception("Symbols tool error")
            _log_audit("symbols", 0, agent_id=agent_id)
            return f"Error querying symbols: {str(e)}"

    @mcp.tool()
    async def references(symbol_name: str, kind: str = "all", session_id: str = "") -> str:
        """Find all references to a specific symbol."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        if not _project_db:
            _log_audit("references", 0, agent_id=agent_id)
            return "Error: Project database not initialized"

        try:
            # Outgoing refs (what this symbol calls)
            outgoing = await asyncio.to_thread(_project_db.get_refs, symbol_name=symbol_name, kind=kind)
            # Incoming refs (who calls this symbol)
            callers = await asyncio.to_thread(_project_db.get_callers, symbol_name=symbol_name, kind=kind)

            results = {
                "outgoing": outgoing,
                "callers": [{"name": c["name"], "kind": c["kind"], "file_id": c["file_id"], "line": c["line"], "scope": c.get("scope", "")} for c in callers],
            }
            total = len(outgoing) + len(callers)
            _log_audit("references", total, agent_id=agent_id)
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.exception("References tool error")
            _log_audit("references", 0, agent_id=agent_id)
            return f"Error finding references: {str(e)}"

    @mcp.tool()
    async def file_context(file_path: str, session_id: str = "") -> str:
        """Get structural summary of a file (symbols, imports, relationships)."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        if ".." in file_path or file_path.startswith("/"):
            _log_audit("file_context", 0, agent_id=agent_id)
            return "Error: Invalid file path (path traversal detected)"

        if not _project_db:
            _log_audit("file_context", 0, agent_id=agent_id)
            return "Error: Project database not initialized"

        try:
            file_info = await asyncio.to_thread(_project_db.get_file, path=file_path)
            if not file_info:
                _log_audit("file_context", 0, agent_id=agent_id)
                return json.dumps(None, indent=2)
            file_id = file_info["id"]

            # Get symbols for this file
            all_symbols = await asyncio.to_thread(
                lambda: [dict(r) for r in _project_db.conn.execute(
                    "SELECT id, name, kind, line, col, scope, signature FROM symbols WHERE file_id = ? ORDER BY line",
                    (file_id,)
                ).fetchall()]
            )

            # Get refs from symbols in this file
            symbol_ids = [s["id"] for s in all_symbols]
            file_refs = []
            if symbol_ids:
                placeholders = ",".join("?" * len(symbol_ids))
                file_refs = await asyncio.to_thread(
                    lambda: [dict(r) for r in _project_db.conn.execute(
                        f"SELECT from_symbol_id, to_symbol_id, kind, line FROM refs WHERE from_symbol_id IN ({placeholders}) ORDER BY line",
                        symbol_ids
                    ).fetchall()]
                )

            result = {
                "file": file_info,
                "symbols": all_symbols,
                "references": file_refs,
            }
            _log_audit("file_context", 1, agent_id=agent_id)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception("File context tool error")
            _log_audit("file_context", 0, agent_id=agent_id)
            return f"Error retrieving file context: {str(e)}"

    @mcp.tool()
    async def impact(symbol_name: str, depth: int = 3, session_id: str = "") -> str:
        """Analyze what breaks if a symbol is changed — traverses the dependency graph."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        if not _project_db:
            _log_audit("impact", 0, agent_id=agent_id)
            return "Error: Project database not initialized"

        try:
            # Resolve symbol name to ID first
            symbols = await asyncio.to_thread(_project_db.lookup_symbols, symbol_name)
            if not symbols:
                _log_audit("impact", 0, agent_id=agent_id)
                return json.dumps([], indent=2)
            symbol_id = symbols[0]["id"]
            results = await asyncio.to_thread(_project_db.get_forward_refs, symbol_id, depth)
            _log_audit("impact", len(results), agent_id=agent_id)
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.exception("Impact tool error")
            _log_audit("impact", 0, agent_id=agent_id)
            return f"Error analyzing impact: {str(e)}"

    # --- Admin tools (global scope only) ---

    @mcp.tool()
    async def register_project(path: str, name: str, language: str = "", collection_id: int = 0, session_id: str = "") -> str:
        """Register a new project for indexing (global scope only)."""
        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        try:
            project_id = await asyncio.to_thread(
                _global_db.register_project, path, name,
                language or None, collection_id or None
            )
            project = await asyncio.to_thread(_global_db.get_project, project_id)
            if not project:
                _log_audit("register_project", 0, scope_level="global")
                return f"Error: Project registered (id={project_id}) but retrieval failed"
            _log_audit("register_project", 1, scope_level="global")
            return json.dumps(project, indent=2)
        except Exception as e:
            logger.exception("register_project error")
            _log_audit("register_project", 0, scope_level="global")
            return f"Error: {str(e)}"

    @mcp.tool()
    async def reindex(project_id: int, session_id: str = "") -> str:
        """Trigger re-indexing of a project (global scope only)."""
        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        try:
            project = await asyncio.to_thread(_global_db.get_project, project_id)
            if not project:
                _log_audit("reindex", 0, scope_level="global")
                return f"Error: Project {project_id} not found"

            pipeline = IndexerPipeline(
                project["path"],
                global_db=_global_db,
                languages=project.get("language", "").split(",") if project.get("language") else None
            )
            pipeline.project_id = project_id

            stats = await asyncio.to_thread(pipeline.index_project)
            result = {
                "project_id": project_id,
                "files_processed": stats.files_processed,
                "files_skipped": stats.files_skipped,
                "files_failed": stats.files_failed,
                "symbols_extracted": stats.symbols_extracted,
                "chunks_created": stats.chunks_created,
                "time_elapsed": round(stats.time_elapsed, 2),
            }
            _log_audit("reindex", stats.files_processed, scope_level="global")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception("reindex error")
            _log_audit("reindex", 0, scope_level="global")
            return f"Error: {str(e)}"

    @mcp.tool()
    async def create_scope_tool(agent_id: str, scope_level: str, project_ids: list[int] = [], collection_ids: list[int] = [], ttl_minutes: int = 30, session_id: str = "") -> str:
        """Create a session token for a task agent (global scope only)."""
        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

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

    @mcp.tool()
    async def status(project_id: int = 0, session_id: str = "") -> str:
        """Get system status: projects, jobs, audit log (global scope only)."""
        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        try:
            projects = await asyncio.to_thread(_global_db.list_projects)
            pending_jobs = await asyncio.to_thread(_global_db.get_pending_jobs)
            recent_audit = await asyncio.to_thread(_global_db.get_audit_log, 10)

            result = {
                "project_count": len(projects),
                "projects": [{"id": p["id"], "name": p["name"], "path": p["path"]} for p in projects],
                "pending_jobs": len(pending_jobs),
                "recent_audit_entries": len(recent_audit),
            }

            if project_id:
                project = await asyncio.to_thread(_global_db.get_project, project_id)
                if project:
                    result["project_detail"] = project

            _log_audit("status", 1, scope_level="global")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception("status error")
            _log_audit("status", 0, scope_level="global")
            return f"Error: {str(e)}"

    return mcp


async def run_server(project_path: str, global_db_path: Optional[str] = None) -> int:
    """Run the MCP server on stdio transport."""
    if not global_db_path:
        home = Path.home()
        global_db_path = str(home / ".tessera" / "global.db")

    mcp = create_server(project_path, global_db_path)

    try:
        await mcp.run_stdio_async()
    except Exception as e:
        logger.exception("Server error")
        return 1

    return 0
