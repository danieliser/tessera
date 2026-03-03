"""Server state: module globals, audit logging, session validation, DB resolution."""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..auth import ScopeInfo, SessionExpiredError, SessionNotFoundError, validate_session
from ..db import GlobalDB, ProjectDB
from ..drift_adapter import DriftAdapter
from ..embeddings import EmbeddingClient, FastembedClient, FastembedReranker, create_embedding_client, create_reranker
from ..graph import MAX_CACHED_GRAPHS, evict_lru_graph, load_project_graph
from ..indexer import IndexerPipeline
from ..indexer._helpers import compute_parser_digest

# Scope level hierarchy for comparison
_SCOPE_LEVELS = {"project": 0, "collection": 1, "global": 2}

logger = logging.getLogger("tessera.server")
audit_logger = logging.getLogger("tessera.server.audit")

# Global database references (set during create_server)
_db_cache: dict[int, ProjectDB] = {}       # project_id → ProjectDB (lazy-loaded)
_stale_projects: set[int] = set()       # project IDs whose index was built with an older parser
_locked_project: int | None = None      # If --project given, only this project is queryable
_global_db: GlobalDB | None = None
_drift_adapter: DriftAdapter | None = None
_embedding_client: EmbeddingClient | FastembedClient | None = None
_reranker: FastembedReranker | None = None
_project_graphs: dict = {}  # project_id → ProjectGraph
_graph_stats: dict[int, dict] = {}  # project_id → metadata
_graph_lock = threading.Lock()  # Explicit lock for graph swap (don't rely on GIL)


def _log_audit(tool_name: str, result_count: int, agent_id: str = "dev", scope_level: str = "project", ppr_used: bool = False):
    """Log audit event to both Python logger and GlobalDB."""
    audit_logger.info(
        "tool_call",
        extra={
            "agent_id": agent_id,
            "scope_level": scope_level,
            "tool_called": tool_name,
            "result_count": result_count,
            "ppr_used": ppr_used,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    if _global_db:
        try:
            _global_db.insert_audit(agent_id, scope_level, tool_name, result_count)
        except Exception:
            logger.warning("Failed to persist audit log to database")


def _check_session(arguments: dict[str, Any], required_level: str = "project") -> tuple[ScopeInfo | None, str | None]:
    """Validate session from tool arguments.

    Returns:
        (scope_info, None) on success, or (None, error_message) on failure.
        If no session_id provided, returns (None, None) for dev mode fallback.
    """
    session_id = arguments.get("session_id") or os.environ.get("TESSERA_SESSION_ID")
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


def _stale_index_warning(project_ids: list[int]) -> str:
    """Return a warning string if any of the given projects have stale indexes."""
    stale = [pid for pid in project_ids if pid in _stale_projects]
    if not stale:
        return ""
    names = []
    if _global_db:
        for pid in stale:
            p = _global_db.get_project(pid)
            names.append(p["name"] if p else str(pid))
    return (
        f"⚠ Stale index detected for: {', '.join(names or [str(s) for s in stale])}. "
        "The parser has changed since last indexing. Run `reindex(project_id=..., force=True)` to update.\n\n"
    )


def _get_project_dbs(scope: ScopeInfo | None) -> list[tuple[int, str, ProjectDB]]:
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
            # Check if index was built with current parser version
            stored_digest = db.get_meta("parser_digest")
            if stored_digest and stored_digest != compute_parser_digest():
                _stale_projects.add(pid)
                logger.warning(
                    "Project %d (%s) index is stale — run reindex with force=True",
                    pid, project["name"],
                )
            elif not stored_digest:
                # Pre-digest index — mark stale to be safe
                _stale_projects.add(pid)
            result.append((pid, project["name"], db))
        except Exception as e:
            logger.warning("Failed to open ProjectDB for %d (%s): %s", pid, project_path, e)

    return result


def _init_essential(
    project_path: str | None,
    global_db_path: str,
    embedding_endpoint: str | None = None,
    embedding_model: str | None = None,
    embedding_provider: str = "auto",
    reranking_model: str | None = None,
    no_reranking: bool = False,
) -> None:
    """Fast init: DB connections, embedding client, project lock. No heavy I/O."""
    global _db_cache, _locked_project, _global_db, _drift_adapter, _embedding_client, _reranker, _project_graphs, _graph_stats, _stale_projects

    # Reset state
    _db_cache = {}
    _stale_projects = set()
    _locked_project = None
    _drift_adapter = None
    _embedding_client = None
    _reranker = None
    _project_graphs = {}
    _graph_stats = {}

    # Initialize embedding client via provider factory
    _embedding_client = create_embedding_client(
        provider=embedding_provider,
        embedding_endpoint=embedding_endpoint,
        embedding_model=embedding_model,
    )

    # Initialize reranker (fastembed only, auto-skip if not installed)
    _reranker = create_reranker(
        provider=embedding_provider,
        reranking_model=reranking_model,
        enabled=not no_reranking,
    )

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


def _init_background(project_path: str | None = None) -> None:
    """Slow init: crash recovery, graph loading, drift adapter. Runs in background."""
    global _drift_adapter

    if not _global_db:
        return

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

    # Load graphs for all known projects
    import time as _time
    total_graph_start = _time.perf_counter()

    projects = _global_db.list_projects()
    logger.info("Loading PPR graphs for %d projects...", len(projects))

    for project in projects:
        pid = project["id"]
        try:
            db = ProjectDB(project["path"])
            graph = load_project_graph(db, pid)
            with _graph_lock:
                _project_graphs[pid] = graph
            _graph_stats[pid] = {
                'edge_count': graph.edge_count,
                'symbol_count': graph.n_symbols,
                'loaded_at': graph.loaded_at,
                'sparse': graph.is_sparse_fallback(),
            }
            logger.info(
                "Loaded graph for project %d (%d symbols, %d edges%s)",
                pid, graph.n_symbols, graph.edge_count,
                ", SPARSE - PPR skipped" if graph.is_sparse_fallback() else ""
            )
        except Exception as e:
            logger.warning("Failed to load graph for project %d: %s", pid, e)

    total_ms = (_time.perf_counter() - total_graph_start) * 1000
    logger.info("Total graph startup: %.1fms (%d graphs loaded)", total_ms, len(_project_graphs))
    if total_ms > 5000:
        logger.warning("Graph startup exceeded 5s threshold (%.1fms)", total_ms)

    # Check LRU cap
    while len(_project_graphs) > MAX_CACHED_GRAPHS:
        evict_lru_graph(_project_graphs)

    # Log total memory estimate
    total_memory = sum(g.estimated_memory_bytes for g in _project_graphs.values())
    if total_memory > 500 * 1024 * 1024:  # 500MB
        logger.warning("Graph cache using ~%.1fMB (>500MB threshold)", total_memory / (1024*1024))
    else:
        logger.info("Graph cache memory estimate: %.1fMB", total_memory / (1024*1024))

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

    logger.info("Background init complete")


def _init_state(
    project_path: str | None,
    global_db_path: str,
    embedding_endpoint: str | None = None,
    embedding_model: str | None = None,
    embedding_provider: str = "auto",
    reranking_model: str | None = None,
    no_reranking: bool = False,
) -> None:
    """Full synchronous init (for tests and CLI). Runs both essential and background."""
    _init_essential(
        project_path, global_db_path, embedding_endpoint, embedding_model,
        embedding_provider, reranking_model, no_reranking,
    )
    _init_background(project_path)
