"""MCP server implementation for Tessera.

Exposes tools via stdio transport for code search, symbol navigation,
reference tracing, impact analysis, and project management.
"""

# Re-export _state module so that attribute access on `server._global_db` etc.
# reads the live mutable value from _state (not a stale copy).
from tessera.server import _state
from tessera.server._app import _create_hmr_app, app, create_server, run_server
from tessera.server._state import (
    _SCOPE_LEVELS,
    _check_session,
    _get_project_dbs,
    _init_background,
    _init_essential,
    _init_state,
    _log_audit,
)


def __getattr__(name: str):
    """Proxy mutable globals from _state so `server._global_db` etc. work."""
    _MUTABLE_GLOBALS = {
        "_db_cache", "_locked_project", "_global_db", "_drift_adapter",
        "_embedding_client", "_project_graphs", "_graph_stats", "_graph_lock",
    }
    if name in _MUTABLE_GLOBALS:
        return getattr(_state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "app",
    "create_server",
    "run_server",
    "_create_hmr_app",
    "_check_session",
    "_get_project_dbs",
    "_init_background",
    "_init_essential",
    "_init_state",
    "_log_audit",
    "_SCOPE_LEVELS",
]
