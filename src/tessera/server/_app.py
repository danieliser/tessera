"""Server creation and lifecycle management."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP

from ._state import _init_background, _init_essential, _init_state
from .tools import register_tools

logger = logging.getLogger("tessera.server")


def create_server(
    project_path: str | None,
    global_db_path: str,
    embedding_endpoint: str | None = None,
    embedding_model: str | None = None,
) -> FastMCP:
    """Create and configure the MCP server (synchronous, for tests and CLI).

    Runs _init_state immediately, then registers tools. Use _create_hmr_app()
    for the lifespan-based pattern that defers init.
    """
    _init_state(project_path, global_db_path, embedding_endpoint, embedding_model)
    mcp = FastMCP("tessera")
    register_tools(mcp)
    return mcp


def _create_hmr_app() -> FastMCP:
    """Create a FastMCP app for mcp-hmr with deferred initialization.

    Returns instantly — tools registered, no I/O. Essential init (DB, embedding
    client) runs in lifespan. Heavy work (crash recovery, graph loading) fires
    as a background task so the MCP handshake completes immediately.
    """
    @asynccontextmanager
    async def _lifespan(server):
        project_path = os.environ.get("TESSERA_PROJECT_PATH")
        global_db_path = os.environ.get(
            "TESSERA_GLOBAL_DB",
            str(Path.home() / ".tessera" / "global.db"),
        )
        embedding_endpoint = os.environ.get("TESSERA_EMBEDDING_ENDPOINT")
        embedding_model = os.environ.get("TESSERA_EMBEDDING_MODEL")

        # Fast: DB connections and project lock (~10ms)
        _init_essential(project_path, global_db_path, embedding_endpoint, embedding_model)
        logger.info("Tessera server ready (essential init complete)")

        # Slow: crash recovery + graph loading in background thread
        bg_task = asyncio.create_task(
            asyncio.to_thread(_init_background, project_path)
        )
        try:
            yield
        finally:
            if not bg_task.done():
                bg_task.cancel()

    mcp = FastMCP("tessera", lifespan=_lifespan)
    register_tools(mcp)
    return mcp


async def run_server(
    project_path: str | None = None,
    global_db_path: str | None = None,
    embedding_endpoint: str | None = None,
    embedding_model: str | None = None,
) -> int:
    """Run the MCP server on stdio transport."""
    if not global_db_path:
        home = Path.home()
        global_db_path = str(home / ".tessera" / "global.db")

    mcp = create_server(project_path, global_db_path, embedding_endpoint, embedding_model)

    try:
        await mcp.run_stdio_async()
    except Exception:
        logger.exception("Server error")
        return 1

    return 0


# Module-level app for mcp-hmr discovery.
# Created instantly (no I/O) — heavy init deferred to lifespan.
app = _create_hmr_app()
