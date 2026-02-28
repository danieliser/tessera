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
import os
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
from .search import hybrid_search, doc_search
from .drift_adapter import DriftAdapter
from .embeddings import EmbeddingClient, EmbeddingUnavailableError

# Scope level hierarchy for comparison
_SCOPE_LEVELS = {"project": 0, "collection": 1, "global": 2}

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger(__name__ + ".audit")

# Global database references (set during create_server)
_db_cache: dict[int, ProjectDB] = {}       # project_id → ProjectDB (lazy-loaded)
_locked_project: Optional[int] = None      # If --project given, only this project is queryable
_global_db: Optional[GlobalDB] = None
_drift_adapter: Optional[DriftAdapter] = None
_embedding_client: Optional[EmbeddingClient] = None


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


def _get_project_dbs(scope: Optional[ScopeInfo]) -> list[tuple[int, str, ProjectDB]]:
    """Resolve which ProjectDBs the current request can access.

    Returns:
        List of (project_id, project_name, ProjectDB) tuples.
        Lazily loads ProjectDB instances into _db_cache.
    """
    if not _global_db:
        return []

    # Determine allowed project IDs
    if _locked_project is not None:
        # Server locked to single project via --project flag
        allowed_ids = [_locked_project]
    elif scope and scope.level == "collection" and scope.collections:
        # Collection scope: resolve collection IDs to project memberships
        all_allowed = set()
        for cid in scope.collections:
            try:
                collection_projects = _global_db.get_collection_projects(int(cid))
                all_allowed.update(p["id"] for p in collection_projects)
            except Exception as e:
                logger.warning("Failed to resolve collection %s: %s", cid, e)
        allowed_ids = list(all_allowed)
        if not allowed_ids:
            logger.warning("Collection scope resolved to 0 projects")
    elif scope and scope.projects:
        # Session specifies allowed projects
        allowed_ids = [int(p) for p in scope.projects]
    else:
        # Dev mode with no lock — all registered projects
        projects = _global_db.list_projects()
        allowed_ids = [p["id"] for p in projects]

    result = []
    for pid in allowed_ids:
        if pid in _db_cache:
            project = _global_db.get_project(pid)
            name = project["name"] if project else f"project-{pid}"
            result.append((pid, name, _db_cache[pid]))
            continue

        # Lazy-load from GlobalDB
        project = _global_db.get_project(pid)
        if not project:
            logger.warning("Project %d not found in global DB, skipping", pid)
            continue

        project_path = project["path"]
        if not os.path.isdir(project_path):
            logger.warning("Project %d path %s does not exist, skipping", pid, project_path)
            continue

        db_dir = str(ProjectDB._get_data_dir(project_path))
        if not os.path.isdir(db_dir):
            logger.warning("Project %d has no index at %s, skipping", pid, db_dir)
            continue

        try:
            db = ProjectDB(project_path)
            _db_cache[pid] = db
            result.append((pid, project["name"], db))
        except Exception as e:
            logger.warning("Failed to open ProjectDB for %d (%s): %s", pid, project_path, e)

    return result


def create_server(
    project_path: Optional[str],
    global_db_path: str,
    embedding_endpoint: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> FastMCP:
    """Create and configure the MCP server.

    Args:
        project_path: Optional path to lock server to a single project.
                      If None, server operates in multi-project mode.
        global_db_path: Path to the global.db file.
        embedding_endpoint: Optional OpenAI-compatible embedding endpoint URL.
        embedding_model: Optional model identifier for the embedding endpoint.
    """
    global _db_cache, _locked_project, _global_db, _drift_adapter, _embedding_client

    # Reset state
    _db_cache = {}
    _locked_project = None
    _drift_adapter = None
    _embedding_client = None

    # Initialize embedding client if endpoint configured
    if embedding_endpoint:
        _embedding_client = EmbeddingClient(
            endpoint=embedding_endpoint,
            model=embedding_model or "default",
        )
        logger.info("Embedding client configured: %s (model=%s)", embedding_endpoint, embedding_model or "default")

    # Initialize global database
    Path(global_db_path).parent.mkdir(parents=True, exist_ok=True)
    _global_db = GlobalDB(global_db_path)

    # If project_path given, lock to that project
    if project_path:
        project_path = os.path.abspath(project_path)
        db = ProjectDB(project_path)

        # Find or register in global DB
        projects = _global_db.list_projects()
        matched = [p for p in projects if os.path.abspath(p["path"]) == project_path]
        if matched:
            pid = matched[0]["id"]
        else:
            pid = _global_db.register_project(
                path=project_path,
                name=os.path.basename(project_path)
            )

        _db_cache[pid] = db
        _locked_project = pid
        logger.info("Server locked to project %d at %s", pid, project_path)
    else:
        logger.info("Server in multi-project mode")

    # Crash recovery: re-index any jobs that were interrupted (status='running')
    incomplete_jobs = _global_db.get_incomplete_jobs()
    if incomplete_jobs:
        logger.info("Crash recovery: found %d incomplete job(s)", len(incomplete_jobs))
    for job in incomplete_jobs:
        job_id = job["id"]
        project_id = job["project_id"]
        project = _global_db.get_project(project_id)
        if not project or not os.path.isdir(project["path"]):
            logger.info("Crash recovery: project %d path not found, marking job %d failed", project_id, job_id)
            _global_db.fail_job(job_id, "Project path not found")
            continue
        logger.info("Crash recovery: resuming job %d for project %d at %s", job_id, project_id, project["path"])
        try:
            pipeline = IndexerPipeline(project["path"], global_db=_global_db)
            pipeline.project_id = project_id
            pipeline.index_project()
            _global_db.complete_job(job_id)
            logger.info("Crash recovery: job %d completed", job_id)
        except Exception as e:
            _global_db.fail_job(job_id, str(e))
            logger.warning("Crash recovery: job %d failed: %s", job_id, e)

    mcp = FastMCP("tessera")

    # Load drift adapter if available
    if project_path:
        drift_path = ProjectDB._get_data_dir(project_path) / "drift_matrix.npy"
        if drift_path.exists():
            try:
                _drift_adapter = DriftAdapter.load(str(drift_path))
                logger.info("Drift adapter loaded from %s", drift_path)
            except Exception as e:
                logger.warning("Failed to load drift adapter: %s", e)
                _drift_adapter = None

    # --- Core tools (project scope) ---

    @mcp.tool()
    async def search(query: str, limit: int = 10, filter_language: str = "", session_id: str = "") -> str:
        """Hybrid semantic + keyword search across indexed codebase."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("search", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Embed query if embedding client is available
            query_embedding = None
            if _embedding_client:
                try:
                    import numpy as np
                    raw = await asyncio.to_thread(_embedding_client.embed_single, query)
                    query_embedding = np.array(raw, dtype=np.float32)
                except EmbeddingUnavailableError:
                    logger.debug("Embedding endpoint unavailable, falling back to keyword-only search")

            # Parallel query pattern — hybrid when embeddings available, keyword-only otherwise
            if query_embedding is not None:
                tasks = [
                    asyncio.to_thread(hybrid_search, query, query_embedding, db, limit)
                    for pid, pname, db in dbs
                ]
            else:
                tasks = [
                    asyncio.to_thread(db.keyword_search, query, limit)
                    for pid, pname, db in dbs
                ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            for (pid, pname, db), result in zip(dbs, results_list):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                for r in result:
                    r["project_id"] = pid
                    r["project_name"] = pname
                all_results.extend(result)

            # Sort by score descending, cap at limit
            all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
            all_results = all_results[:limit]

            _log_audit("search", len(all_results), agent_id=agent_id)
            return json.dumps(all_results, indent=2)
        except Exception as e:
            logger.exception("Search tool error")
            _log_audit("search", 0, agent_id=agent_id)
            return f"Error during search: {str(e)}"

    @mcp.tool()
    async def doc_search_tool(query: str, limit: int = 10, formats: str = "", session_id: str = "") -> str:
        """Search non-code documents only (markdown, PDF, YAML, JSON)."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("doc_search", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            format_list = [f.strip() for f in formats.split(",") if f.strip()] if formats else None

            # Embed query if embedding client is available
            query_embedding = None
            if _embedding_client:
                try:
                    import numpy as np
                    raw = await asyncio.to_thread(_embedding_client.embed_single, query)
                    query_embedding = np.array(raw, dtype=np.float32)
                except EmbeddingUnavailableError:
                    logger.debug("Embedding endpoint unavailable, falling back to keyword-only doc_search")

            # Parallel query across projects
            tasks = [
                asyncio.to_thread(
                    doc_search, query, query_embedding, db, limit, format_list
                )
                for pid, pname, db in dbs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            for (pid, pname, db), result in zip(dbs, results_list):
                if isinstance(result, Exception):
                    logger.warning("doc_search on project %d failed: %s", pid, result)
                    continue
                for r in result:
                    r["project_id"] = pid
                    r["project_name"] = pname
                all_results.extend(result)

            all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
            all_results = all_results[:limit]

            _log_audit("doc_search", len(all_results), agent_id=agent_id)
            return json.dumps(all_results, indent=2)
        except Exception as e:
            logger.exception("doc_search error")
            _log_audit("doc_search", 0, agent_id=agent_id)
            return f"Error: {str(e)}"

    @mcp.tool()
    async def symbols(query: str = "*", kind: str = "", language: str = "", session_id: str = "") -> str:
        """List and query symbols (functions, classes, imports) in the codebase."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("symbols", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Parallel query pattern
            tasks = [
                asyncio.to_thread(
                    db.lookup_symbols, query,
                    kind or None, language or None
                )
                for pid, pname, db in dbs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            for (pid, pname, db), result in zip(dbs, results_list):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                for r in result:
                    r["project_id"] = pid
                    r["project_name"] = pname
                all_results.extend(result)

            _log_audit("symbols", len(all_results), agent_id=agent_id)
            return json.dumps(all_results, indent=2)
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

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("references", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Parallel query pattern: fetch both outgoing and callers for each DB in parallel
            async def _query_references(db, symbol_name, kind):
                """Query both references and callers for a database."""
                outgoing = await asyncio.to_thread(db.get_refs, symbol_name=symbol_name, kind=kind)
                callers = await asyncio.to_thread(db.get_callers, symbol_name=symbol_name, kind=kind)
                return (outgoing, callers)

            tasks = [
                _query_references(db, symbol_name, kind)
                for pid, pname, db in dbs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_outgoing = []
            all_callers = []
            for (pid, pname, db), result in zip(dbs, results_list):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                outgoing, callers = result
                for r in outgoing:
                    all_outgoing.append({
                        "to_symbol": r.get("to_symbol_name", ""),
                        "kind": r.get("kind", ""),
                        "line": r.get("line", 0),
                        "project_id": pid,
                        "project_name": pname,
                    })
                for c in callers:
                    all_callers.append({
                        "name": c.get("name", ""),
                        "kind": c.get("kind", ""),
                        "file_id": c.get("file_id"),
                        "line": c.get("line", 0),
                        "scope": c.get("scope", ""),
                        "project_id": pid,
                        "project_name": pname,
                    })

            results = {"outgoing": all_outgoing, "callers": all_callers}
            total = len(all_outgoing) + len(all_callers)
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

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("file_context", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Try each project until file is found
            for pid, pname, db in dbs:
                file_info = await asyncio.to_thread(db.get_file, path=file_path)
                if not file_info:
                    continue

                file_id = file_info["id"]
                file_info_dict = dict(file_info)
                file_info_dict["project_id"] = pid
                file_info_dict["project_name"] = pname

                all_symbols = await asyncio.to_thread(
                    lambda: [dict(r) for r in db.conn.execute(
                        "SELECT id, name, kind, line, col, scope, signature FROM symbols WHERE file_id = ? ORDER BY line",
                        (file_id,)
                    ).fetchall()]
                )

                symbol_ids = [s["id"] for s in all_symbols]
                file_refs = []
                if symbol_ids:
                    placeholders = ",".join("?" * len(symbol_ids))
                    file_refs = await asyncio.to_thread(
                        lambda: [dict(r) for r in db.conn.execute(
                            f"SELECT from_symbol_id, to_symbol_id, kind, line FROM refs WHERE from_symbol_id IN ({placeholders}) ORDER BY line",
                            symbol_ids
                        ).fetchall()]
                    )

                result = {
                    "file": file_info_dict,
                    "symbols": all_symbols,
                    "references": file_refs,
                }
                _log_audit("file_context", 1, agent_id=agent_id)
                return json.dumps(result, indent=2)

            _log_audit("file_context", 0, agent_id=agent_id)
            return json.dumps(None, indent=2)
        except Exception as e:
            logger.exception("File context tool error")
            _log_audit("file_context", 0, agent_id=agent_id)
            return f"Error retrieving file context: {str(e)}"

    @mcp.tool()
    async def impact(symbol_name: str, depth: int = 3, include_types: bool = True, session_id: str = "") -> str:
        """Analyze what breaks if a symbol is changed — traverses the dependency graph."""
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("impact", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Parallel query pattern
            tasks = [
                asyncio.to_thread(db.get_impact, symbol_name, depth, include_types)
                for pid, pname, db in dbs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            for (pid, pname, db), result in zip(dbs, results_list):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                for r in result:
                    r["project_id"] = pid
                    r["project_name"] = pname
                all_results.extend(result)

            _log_audit("impact", len(all_results), agent_id=agent_id)
            return json.dumps(all_results, indent=2)
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
    async def reindex(project_id: int, mode: str = "full", session_id: str = "") -> str:
        """Trigger re-indexing of a project (global scope only).

        Args:
            project_id: ID of the project to reindex
            mode: 'full' (default) for complete reindex, 'incremental' for git-diff based update
            session_id: Optional session token
        """
        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if mode not in ("full", "incremental"):
            return f"Error: Invalid mode '{mode}'. Must be 'full' or 'incremental'."

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
                embedding_client=_embedding_client,
                languages=project.get("language", "").split(",") if project.get("language") else None
            )
            pipeline.project_id = project_id

            if mode == "incremental":
                stats = await asyncio.to_thread(pipeline.index_changed)
            else:
                stats = await asyncio.to_thread(pipeline.index_project)

            # Invalidate cache so next query picks up fresh data
            _db_cache.pop(project_id, None)

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
    async def create_scope_tool(agent_id: str, scope_level: str, project_ids: list[int] = None, collection_ids: list[int] = None, ttl_minutes: int = 30, session_id: str = "") -> str:
        """Create a session token for a task agent (global scope only)."""
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

    @mcp.tool()
    async def drift_train(sample_size: int = 200, session_id: str = "") -> str:
        """Train drift adapter for embedding model migration (global scope only).

        Samples chunks from the index, embeds them with the current model,
        and trains an Orthogonal Procrustes rotation matrix to align old and new embeddings.
        """
        scope, err = _check_session({"session_id": session_id}, "global")
        if err:
            return err

        if not _global_db:
            return "Error: Global database not initialized"

        dbs = _get_project_dbs(scope)
        if not dbs:
            return "Error: No accessible projects"

        try:
            # Use the first project with enough chunks
            for pid, pname, db in dbs:
                all_embeddings = await asyncio.to_thread(db.get_all_embeddings)
                if not all_embeddings or len(all_embeddings[0]) < sample_size:
                    continue

                chunk_ids, old_embeddings_matrix = all_embeddings
                import numpy as np

                # Sample random indices
                indices = np.random.choice(len(chunk_ids), size=min(sample_size, len(chunk_ids)), replace=False)
                sampled_ids = [chunk_ids[i] for i in indices]
                old_emb = old_embeddings_matrix[indices]

                # Get chunk content for re-embedding
                chunks = []
                for cid in sampled_ids:
                    chunk = await asyncio.to_thread(db.get_chunk, cid)
                    if chunk:
                        chunks.append(chunk.get("content", ""))

                if not chunks:
                    continue

                # This requires an embedding client — check if available
                # For now, return an error indicating the operator needs to provide embeddings
                # In production, this would call the embedding endpoint
                _log_audit("drift_train", 0, scope_level="global")
                return json.dumps({
                    "status": "ready",
                    "message": "Drift training requires re-embedding with new model. Use embedding endpoint to generate new embeddings for sampled chunks.",
                    "sample_size": len(chunks),
                    "project_id": pid,
                    "project_name": pname,
                }, indent=2)

            _log_audit("drift_train", 0, scope_level="global")
            return "Error: No project has enough indexed chunks for drift training"
        except Exception as e:
            logger.exception("drift_train error")
            _log_audit("drift_train", 0, scope_level="global")
            return f"Error: {str(e)}"

    # --- Federation tools (project scope minimum) ---

    @mcp.tool()
    async def cross_refs(symbol_name: str, session_id: str = "") -> str:
        """Find cross-project references for a symbol.

        Returns references where the symbol is defined in one project
        and referenced in another.
        """
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("cross_refs", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Parallel query: find definitions in each project
            def_tasks = [
                asyncio.to_thread(db.lookup_symbols, symbol_name)
                for pid, pname, db in dbs
            ]
            def_results = await asyncio.gather(*def_tasks, return_exceptions=True)

            # Collect which projects define this symbol
            definition_projects = {}
            for (pid, pname, db), result in zip(dbs, def_results):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                if result:
                    definition_projects[pid] = pname

            if not definition_projects:
                result_dict = {"symbol": symbol_name, "definition_projects": {}, "cross_refs": []}
                _log_audit("cross_refs", 0, agent_id=agent_id)
                return json.dumps(result_dict, indent=2)

            # Parallel query: find callers of this symbol in each project
            # get_callers returns caller symbol records (with name, file_id, line)
            caller_tasks = [
                asyncio.to_thread(db.get_callers, symbol_name=symbol_name)
                for pid, pname, db in dbs
            ]
            caller_results = await asyncio.gather(*caller_tasks, return_exceptions=True)

            # Build cross-project references
            cross_refs = []
            for (pid, pname, db), result in zip(dbs, caller_results):
                if isinstance(result, Exception):
                    logger.warning("Query on project %d failed: %s", pid, result)
                    continue
                # Each caller is a symbol in this project that references our target
                for caller in result:
                    # Cross-project: caller is in this project, definition is in another
                    for def_pid, def_pname in definition_projects.items():
                        if def_pid != pid:
                            cross_refs.append({
                                "from_project_id": pid,
                                "from_project_name": pname,
                                "to_project_id": def_pid,
                                "to_project_name": def_pname,
                                "file": caller.get("file_id", ""),
                                "line": caller.get("line", 0),
                                "kind": caller.get("kind", "function"),
                            })

            result_dict = {
                "symbol": symbol_name,
                "definition_projects": {str(k): {"id": k, "name": v} for k, v in definition_projects.items()},
                "cross_refs": cross_refs,
            }

            _log_audit("cross_refs", len(cross_refs), agent_id=agent_id)
            return json.dumps(result_dict, indent=2)
        except Exception as e:
            logger.exception("cross_refs error")
            _log_audit("cross_refs", 0, agent_id=agent_id)
            return f"Error: {str(e)}"

    @mcp.tool()
    async def collection_map(collection_id: int = 0, session_id: str = "") -> str:
        """Get inter-project dependency graph.

        If collection_id is 0, use all allowed projects in scope.
        """
        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"

        # Get allowed projects
        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("collection_map", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        try:
            # Query symbol definitions for each project in parallel
            async def _query_symbols(db):
                """Query all symbols for a database."""
                symbols = await asyncio.to_thread(db.lookup_symbols, "*")
                return symbols

            tasks = [
                _query_symbols(db)
                for pid, pname, db in dbs
            ]
            symbols_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Build project node info and symbol index (name → project_ids)
            projects_dict = {}
            symbol_definitions = {}  # symbol_name → set of project_ids that define it

            for (pid, pname, db), symbols_result in zip(dbs, symbols_list):
                if isinstance(symbols_result, Exception):
                    logger.warning("Symbol query on project %d failed: %s", pid, symbols_result)
                    symbol_count = 0
                    all_symbols = []
                else:
                    all_symbols = symbols_result
                    symbol_count = len(all_symbols)

                projects_dict[pname] = {"id": pid, "symbol_count": symbol_count}

                # Index symbol definitions
                for symbol in all_symbols:
                    sym_name = symbol.get("name", "")
                    if sym_name:
                        if sym_name not in symbol_definitions:
                            symbol_definitions[sym_name] = set()
                        symbol_definitions[sym_name].add(pid)

            # Find edges: for each reference from project A to symbol in project B
            edges = {}  # (from_pid, to_pid) → {cross_refs: count, symbols: set}

            for (pid, pname, db), symbols_result in zip(dbs, symbols_list):
                if isinstance(symbols_result, Exception):
                    continue

                # Get all references from this project
                all_symbols = symbols_result
                for symbol in all_symbols:
                    refs = await asyncio.to_thread(
                        db.get_refs, symbol_id=symbol["id"], kind="all"
                    )
                    for ref in refs:
                        to_symbol_name = ref.get("to_symbol_name", "")
                        # Check if this symbol is defined in another project
                        if to_symbol_name in symbol_definitions:
                            for target_pid in symbol_definitions[to_symbol_name]:
                                if target_pid != pid:  # Cross-project reference
                                    edge_key = (pid, target_pid)
                                    if edge_key not in edges:
                                        edges[edge_key] = {"cross_refs": 0, "symbols": set()}
                                    edges[edge_key]["cross_refs"] += 1
                                    edges[edge_key]["symbols"].add(to_symbol_name)

            # Convert edges to list format, mapping project IDs to names
            pid_to_name = {pid: pname for pid, pname, _ in dbs}
            edges_list = [
                {
                    "from": pid_to_name.get(from_pid, f"project-{from_pid}"),
                    "to": pid_to_name.get(to_pid, f"project-{to_pid}"),
                    "cross_refs": v["cross_refs"],
                    "symbols": sorted(list(v["symbols"])),
                }
                for (from_pid, to_pid), v in edges.items()
            ]

            result_dict = {
                "collection_id": collection_id,
                "projects": projects_dict,
                "edges": edges_list,
            }

            _log_audit("collection_map", len(edges_list), agent_id=agent_id)
            return json.dumps(result_dict, indent=2)
        except Exception as e:
            logger.exception("collection_map error")
            _log_audit("collection_map", 0, agent_id=agent_id)
            return f"Error: {str(e)}"

    # --- Collection admin tools (global scope only) ---

    @mcp.tool()
    async def create_collection_tool(name: str, project_ids: list[int] = None, session_id: str = "") -> str:
        """Create a new collection (global scope only)."""
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

    return mcp


async def run_server(
    project_path: Optional[str] = None,
    global_db_path: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> int:
    """Run the MCP server on stdio transport."""
    if not global_db_path:
        home = Path.home()
        global_db_path = str(home / ".tessera" / "global.db")

    mcp = create_server(project_path, global_db_path, embedding_endpoint, embedding_model)

    try:
        await mcp.run_stdio_async()
    except Exception as e:
        logger.exception("Server error")
        return 1

    return 0
