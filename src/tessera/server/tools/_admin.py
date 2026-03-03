"""Admin tools: register_project, reindex, status, drift_train."""

import asyncio
import json
import logging

from fastmcp import FastMCP

from ...db import ProjectDB
from ...graph import evict_lru_graph, load_project_graph
from ...indexer import IndexerPipeline
from .._state import _check_session, _get_project_dbs, _log_audit

logger = logging.getLogger("tessera.server")


def register_admin_tools(mcp: FastMCP) -> None:
    """Register admin tools (global scope only)."""

    @mcp.tool()
    async def register_project(path: str, name: str, language: str = "", collection_id: int = 0, session_id: str = "") -> str:
        """Register a new project for indexing (global scope only)."""
        from .._state import _global_db

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
    async def reindex(project_id: int, mode: str = "full", force: bool = False, session_id: str = "") -> str:
        """Trigger re-indexing of a project (global scope only).

        Args:
            project_id: ID of the project to reindex
            mode: 'full' (default) for complete reindex, 'incremental' for git-diff based update
            force: If True, re-index all files even if unchanged (use after parser/extractor changes)
            session_id: Optional session token
        """
        from .._state import _db_cache, _embedding_client, _global_db, _graph_lock, _graph_stats, _project_graphs

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
                stats = await asyncio.to_thread(pipeline.index_project, force=force)

            # Invalidate cache so next query picks up fresh data
            _db_cache.pop(project_id, None)

            # Rebuild graph after reindex
            try:
                db = ProjectDB(project["path"])
                graph = load_project_graph(db, project_id)
                with _graph_lock:
                    _project_graphs[project_id] = graph
                _graph_stats[project_id] = {
                    'edge_count': graph.edge_count,
                    'symbol_count': graph.n_symbols,
                    'loaded_at': graph.loaded_at,
                    'sparse': graph.is_sparse_fallback(),
                }
                logger.info("Rebuilt graph for project %d after reindex", project_id)

                # Check LRU cap after adding new graph
                evict_lru_graph(_project_graphs)
            except Exception as e:
                logger.warning("Failed to rebuild graph for project %d: %s", project_id, e)

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
    async def status(project_id: int = 0, session_id: str = "") -> str:
        """Get system status: projects, jobs, audit log (global scope only)."""
        from .._state import _global_db

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
        from .._state import _global_db

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
                old_embeddings_matrix[indices]

                # Get chunk content for re-embedding
                chunks = []
                for cid in sampled_ids:
                    chunk = await asyncio.to_thread(db.get_chunk, cid)
                    if chunk:
                        chunks.append(chunk.get("content", ""))

                if not chunks:
                    continue

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
