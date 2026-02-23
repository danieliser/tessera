"""MCP server implementation for CodeMem.

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
import argparse
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

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


# Global server instance and databases
_server: Optional[Server] = None
_project_db: Optional[ProjectDB] = None
_global_db: Optional[GlobalDB] = None


def _log_audit(tool_name: str, result_count: int, agent_id: str = "dev", scope_level: str = "project"):
    """
    Log audit event to both Python logger and GlobalDB.

    Args:
        tool_name: Name of the tool that was called
        result_count: Number of results returned
        agent_id: ID of the agent making the call (default "dev" for Phase 1)
        scope_level: Scope level (project, collection, global)
    """
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
    # Persist to database if available
    if _global_db:
        try:
            _global_db.insert_audit(agent_id, scope_level, tool_name, result_count)
        except Exception:
            logger.warning("Failed to persist audit log to database")


def _check_session(arguments: dict[str, Any], required_level: str = "project") -> tuple[Optional[ScopeInfo], Optional[list[types.TextContent]]]:
    """Validate session from tool arguments.

    Args:
        arguments: Tool call arguments (may contain session_id)
        required_level: Minimum scope level required ('project', 'collection', 'global')

    Returns:
        (scope_info, None) on success, or (None, error_response) on failure.
        If no session_id provided, returns (None, None) for dev mode fallback.
    """
    session_id = arguments.get("session_id")
    if not session_id:
        return None, None  # Dev mode — no session required

    if not _global_db:
        return None, [types.TextContent(type="text", text="Error: Global database not initialized")]

    try:
        scope = validate_session(_global_db.conn, session_id)
    except SessionNotFoundError:
        return None, [types.TextContent(type="text", text="Error: Invalid session")]
    except SessionExpiredError:
        return None, [types.TextContent(type="text", text="Error: Session expired")]

    # Check scope level
    if _SCOPE_LEVELS.get(scope.level, -1) < _SCOPE_LEVELS.get(required_level, 0):
        return None, [types.TextContent(
            type="text",
            text=f"Error: Insufficient scope. Required: {required_level}, have: {scope.level}"
        )]

    return scope, None


async def search_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    Hybrid semantic + keyword search.

    Params:
      - query (str, required): Search query
      - limit (int, default 10): Max results
      - filter_language (str, optional): Filter by language

    Returns results with file_path, start_line, end_line, content snippet, score.
    """
    scope, err = _check_session(arguments, "project")
    if err:
        return err
    agent_id = scope.agent_id if scope else "dev"

    query = arguments.get("query", "")
    limit = arguments.get("limit", 10)

    if not _project_db:
        _log_audit("search", 0, agent_id=agent_id)
        return [types.TextContent(
            type="text",
            text="Error: Project database not initialized"
        )]

    try:
        results = await asyncio.to_thread(_project_db.keyword_search, query, limit)
        result_text = json.dumps(results, indent=2)
        _log_audit("search", len(results), agent_id=agent_id)
        return [types.TextContent(type="text", text=result_text)]
    except Exception as e:
        logger.exception("Search tool error")
        _log_audit("search", 0, agent_id=agent_id)
        return [types.TextContent(
            type="text",
            text=f"Error during search: {str(e)}"
        )]


async def symbols_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    List/query symbols.

    Params:
      - query (str, default "*"): Symbol name pattern
      - kind (str, optional): Filter by symbol kind (function, class, etc)
      - language (str, optional): Filter by language
      - file_pattern (str, optional): Filter by file pattern

    Returns list of symbol dicts.
    """
    scope, err = _check_session(arguments, "project")
    if err:
        return err
    agent_id = scope.agent_id if scope else "dev"

    query = arguments.get("query", "*")
    kind = arguments.get("kind")
    language = arguments.get("language")

    if not _project_db:
        _log_audit("symbols", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text="Error: Project database not initialized")]

    try:
        results = await asyncio.to_thread(_project_db.lookup_symbols, query, kind, language)
        result_text = json.dumps(results, indent=2)
        _log_audit("symbols", len(results), agent_id=agent_id)
        return [types.TextContent(type="text", text=result_text)]
    except Exception as e:
        logger.exception("Symbols tool error")
        _log_audit("symbols", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text=f"Error querying symbols: {str(e)}")]


async def references_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    Find all references to a symbol.

    Params:
      - symbol_name (str, required): Name of symbol to find references for
      - kind (str, default "all"): Filter by reference kind

    Returns list of reference dicts.
    """
    scope, err = _check_session(arguments, "project")
    if err:
        return err
    agent_id = scope.agent_id if scope else "dev"

    symbol_name = arguments.get("symbol_name")
    kind = arguments.get("kind", "all")

    if not symbol_name:
        raise TypeError("symbol_name is required")

    if not _project_db:
        _log_audit("references", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text="Error: Project database not initialized")]

    try:
        results = await asyncio.to_thread(_project_db.get_refs, symbol_name, kind)
        result_text = json.dumps(results, indent=2)
        _log_audit("references", len(results), agent_id=agent_id)
        return [types.TextContent(type="text", text=result_text)]
    except Exception as e:
        logger.exception("References tool error")
        _log_audit("references", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text=f"Error finding references: {str(e)}")]


async def file_context_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    Structural summary of a file.

    Params:
      - file_path (str, required): Path to file

    Returns file summary with symbols, imports.
    """
    scope, err = _check_session(arguments, "project")
    if err:
        return err
    agent_id = scope.agent_id if scope else "dev"

    file_path = arguments.get("file_path")

    if not file_path:
        raise TypeError("file_path is required")

    if ".." in file_path or file_path.startswith("/"):
        _log_audit("file_context", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text="Error: Invalid file path (path traversal detected)")]

    if not _project_db:
        _log_audit("file_context", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text="Error: Project database not initialized")]

    try:
        file_info = await asyncio.to_thread(_project_db.get_file, path=file_path)
        result_text = json.dumps(file_info, indent=2)
        _log_audit("file_context", 1 if file_info else 0, agent_id=agent_id)
        return [types.TextContent(type="text", text=result_text)]
    except Exception as e:
        logger.exception("File context tool error")
        _log_audit("file_context", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text=f"Error retrieving file context: {str(e)}")]


async def impact_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """
    What breaks if I change this symbol?

    Params:
      - symbol_name (str, required): Name of symbol to analyze
      - depth (int, default 3): How deep to traverse the dependency graph

    Returns affected symbols.
    """
    scope, err = _check_session(arguments, "project")
    if err:
        return err
    agent_id = scope.agent_id if scope else "dev"

    symbol_name = arguments.get("symbol_name")
    depth = arguments.get("depth", 3)

    if not symbol_name:
        raise TypeError("symbol_name is required")

    if not _project_db:
        _log_audit("impact", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text="Error: Project database not initialized")]

    try:
        results = await asyncio.to_thread(_project_db.get_forward_refs, symbol_name, depth)
        result_text = json.dumps(results, indent=2)
        _log_audit("impact", len(results), agent_id=agent_id)
        return [types.TextContent(type="text", text=result_text)]
    except Exception as e:
        logger.exception("Impact tool error")
        _log_audit("impact", 0, agent_id=agent_id)
        return [types.TextContent(type="text", text=f"Error analyzing impact: {str(e)}")]


async def register_project_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Register a new project for indexing (global scope only)."""
    scope, err = _check_session(arguments, "global")
    if err:
        return err

    path = arguments.get("path")
    name = arguments.get("name")

    if not path or not name:
        return [types.TextContent(type="text", text="Error: 'path' and 'name' are required")]

    if not _global_db:
        return [types.TextContent(type="text", text="Error: Global database not initialized")]

    try:
        language = arguments.get("language")
        collection_id = arguments.get("collection_id")
        project_id = await asyncio.to_thread(
            _global_db.register_project, path, name, language, collection_id
        )
        project = await asyncio.to_thread(_global_db.get_project, project_id)
        _log_audit("register_project", 1, scope_level="global")
        return [types.TextContent(type="text", text=json.dumps(project, indent=2))]
    except Exception as e:
        logger.exception("register_project error")
        _log_audit("register_project", 0, scope_level="global")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def reindex_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Trigger re-indexing of a project (global scope only)."""
    scope, err = _check_session(arguments, "global")
    if err:
        return err

    project_id = arguments.get("project_id")

    if not project_id:
        return [types.TextContent(type="text", text="Error: 'project_id' is required")]

    if not _global_db:
        return [types.TextContent(type="text", text="Error: Global database not initialized")]

    try:
        project = await asyncio.to_thread(_global_db.get_project, int(project_id))
        if not project:
            _log_audit("reindex", 0, scope_level="global")
            return [types.TextContent(type="text", text=f"Error: Project {project_id} not found")]

        pipeline = IndexerPipeline(
            project["path"],
            global_db=_global_db,
            languages=project.get("language", "").split(",") if project.get("language") else None
        )
        pipeline.project_id = int(project_id)

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
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.exception("reindex error")
        _log_audit("reindex", 0, scope_level="global")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def create_scope_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Create a session token for a task agent (global scope only)."""
    scope, err = _check_session(arguments, "global")
    if err:
        return err

    agent_id = arguments.get("agent_id")
    scope_level = arguments.get("scope_level")

    if not agent_id or not scope_level:
        return [types.TextContent(type="text", text="Error: 'agent_id' and 'scope_level' are required")]

    if scope_level not in ("project", "collection", "global"):
        return [types.TextContent(type="text", text="Error: scope_level must be 'project', 'collection', or 'global'")]

    if not _global_db:
        return [types.TextContent(type="text", text="Error: Global database not initialized")]

    try:
        project_ids = arguments.get("project_ids")
        collection_ids = arguments.get("collection_ids")
        ttl_minutes = arguments.get("ttl_minutes", 30)

        session_id = await asyncio.to_thread(
            create_scope, _global_db.conn, agent_id, scope_level,
            project_ids, collection_ids, ttl_minutes=ttl_minutes
        )
        result = {"session_id": session_id, "agent_id": agent_id, "scope_level": scope_level, "ttl_minutes": ttl_minutes}
        _log_audit("create_scope", 1, scope_level="global")
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.exception("create_scope error")
        _log_audit("create_scope", 0, scope_level="global")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def revoke_scope_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Revoke an agent's session (global scope only)."""
    scope, err = _check_session(arguments, "global")
    if err:
        return err

    agent_id = arguments.get("agent_id")

    if not agent_id:
        return [types.TextContent(type="text", text="Error: 'agent_id' is required")]

    if not _global_db:
        return [types.TextContent(type="text", text="Error: Global database not initialized")]

    try:
        count = await asyncio.to_thread(revoke_scope, _global_db.conn, agent_id)
        result = {"agent_id": agent_id, "sessions_revoked": count}
        _log_audit("revoke_scope", count, scope_level="global")
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.exception("revoke_scope error")
        _log_audit("revoke_scope", 0, scope_level="global")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def status_tool(tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Get system status (global scope only)."""
    scope, err = _check_session(arguments, "global")
    if err:
        return err

    if not _global_db:
        return [types.TextContent(type="text", text="Error: Global database not initialized")]

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

        project_id = arguments.get("project_id")
        if project_id:
            project = await asyncio.to_thread(_global_db.get_project, int(project_id))
            if project:
                result["project_detail"] = project

        _log_audit("status", 1, scope_level="global")
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.exception("status error")
        _log_audit("status", 0, scope_level="global")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


def create_server(project_path: str, global_db_path: str) -> Server:
    """
    Create and configure the MCP server.

    Args:
        project_path: Path to project directory to index
        global_db_path: Path to global.db file

    Returns:
        Configured Server instance
    """
    global _project_db, _global_db

    server = Server("tessera")

    # Initialize real databases
    project_db_path = str(Path(project_path) / ".tessera" / "project.db")
    Path(project_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(global_db_path).parent.mkdir(parents=True, exist_ok=True)

    _project_db = ProjectDB(project_db_path)
    _global_db = GlobalDB(global_db_path)

    # Register list_tools handler
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """Return the list of available tools."""
        return [
            types.Tool(
                name="search",
                description="Hybrid semantic + keyword search across codebase",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return",
                            "default": 10
                        },
                        "filter_language": {
                            "type": "string",
                            "description": "Filter results by language (optional)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="symbols",
                description="List and query symbols in the codebase",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Symbol name pattern",
                            "default": "*"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by symbol kind (function, class, etc)"
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by language"
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "Filter by file pattern"
                        }
                    }
                }
            ),
            types.Tool(
                name="references",
                description="Find all references to a specific symbol",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol_name": {
                            "type": "string",
                            "description": "Name of symbol to find references for"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by reference kind",
                            "default": "all"
                        }
                    },
                    "required": ["symbol_name"]
                }
            ),
            types.Tool(
                name="file_context",
                description="Get structural summary of a file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to file"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            types.Tool(
                name="impact",
                description="Analyze what breaks if a symbol is changed",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol_name": {
                            "type": "string",
                            "description": "Name of symbol to analyze"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Depth to traverse the dependency graph",
                            "default": 3
                        }
                    },
                    "required": ["symbol_name"]
                }
            ),
            # Admin tools (global scope only)
            types.Tool(
                name="register_project",
                description="Register a new project for indexing (global scope only)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute path to project directory"},
                        "name": {"type": "string", "description": "Project name"},
                        "language": {"type": "string", "description": "Comma-separated languages (optional)"},
                        "collection_id": {"type": "integer", "description": "Parent collection ID (optional)"},
                    },
                    "required": ["path", "name"]
                }
            ),
            types.Tool(
                name="reindex",
                description="Trigger re-indexing of a project (global scope only)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer", "description": "Project ID to reindex"},
                    },
                    "required": ["project_id"]
                }
            ),
            types.Tool(
                name="create_scope",
                description="Create a session token for a task agent (global scope only)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "Agent identifier"},
                        "scope_level": {"type": "string", "description": "Scope: project, collection, or global"},
                        "project_ids": {"type": "array", "items": {"type": "integer"}, "description": "Project IDs for project scope"},
                        "collection_ids": {"type": "array", "items": {"type": "integer"}, "description": "Collection IDs for collection scope"},
                        "ttl_minutes": {"type": "integer", "description": "Token lifetime in minutes", "default": 30},
                    },
                    "required": ["agent_id", "scope_level"]
                }
            ),
            types.Tool(
                name="revoke_scope",
                description="Revoke all sessions for an agent (global scope only)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "Agent ID to revoke sessions for"},
                    },
                    "required": ["agent_id"]
                }
            ),
            types.Tool(
                name="status",
                description="Get system status: projects, jobs, audit log (global scope only)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer", "description": "Get details for a specific project (optional)"},
                    }
                }
            ),
        ]

    # Register call_tool handler
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Route tool calls to appropriate handlers."""
        # Core tools
        if name == "search":
            return await search_tool(name, arguments)
        elif name == "symbols":
            return await symbols_tool(name, arguments)
        elif name == "references":
            return await references_tool(name, arguments)
        elif name == "file_context":
            return await file_context_tool(name, arguments)
        elif name == "impact":
            return await impact_tool(name, arguments)
        # Admin tools (global scope only)
        elif name == "register_project":
            return await register_project_tool(name, arguments)
        elif name == "reindex":
            return await reindex_tool(name, arguments)
        elif name == "create_scope":
            return await create_scope_tool(name, arguments)
        elif name == "revoke_scope":
            return await revoke_scope_tool(name, arguments)
        elif name == "status":
            return await status_tool(name, arguments)
        else:
            return [types.TextContent(
                type="text",
                text=f"Error: Unknown tool '{name}'"
            )]

    return server


async def run_server(project_path: str, global_db_path: Optional[str] = None) -> int:
    """
    Run the MCP server on stdio transport.

    Args:
        project_path: Path to project directory
        global_db_path: Path to global.db (optional, defaults to ~/.tessera/global.db)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not global_db_path:
        home = Path.home()
        global_db_path = str(home / ".tessera" / "global.db")

    # Create server
    server = create_server(project_path, global_db_path)

    # Run on stdio
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(
                    notification_options=types.NotificationOptions(),
                    experimental_capabilities={}
                ),
            )
    except Exception as e:
        logger.exception("Server error")
        return 1

    return 0


def main(args=None) -> int:
    """CLI entry point for serve command."""
    parser = argparse.ArgumentParser(description="Start CodeMem MCP server")
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument("--global-db", help="Path to global.db (optional)")

    parsed_args = parser.parse_args(args)

    # Run server
    return asyncio.run(run_server(parsed_args.project, parsed_args.global_db))


if __name__ == "__main__":
    import sys
    sys.exit(main())
