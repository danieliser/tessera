"""Compact, task-oriented MCP exploration tool."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from fastmcp import FastMCP

from ...embeddings import EmbeddingUnavailableError
from ...search import extract_snippet, hybrid_search
from .._state import _check_session, _get_project_dbs, _log_audit

logger = logging.getLogger("tessera.server")

_MAX_RESULTS = 10
_MAX_FILES = 3
_MAX_FILE_SYMBOLS = 12
_MAX_GRAPH_SYMBOLS = 3
_MAX_GRAPH_EDGES = 8


def _file_outline(db: Any, file_id: int) -> list[dict[str, Any]]:
    """Return a bounded structural outline for one indexed file."""
    rows = db.conn.execute(
        """
        SELECT name, kind, line, scope, signature
        FROM symbols WHERE file_id = ? ORDER BY line LIMIT ?
        """,
        (file_id, _MAX_FILE_SYMBOLS),
    ).fetchall()
    return [dict(row) for row in rows]


def _symbol_graph(db: Any, query: str) -> list[dict[str, Any]]:
    """Return small, exact-name graph slices for identifier-like query tokens."""
    graph: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for token in re.findall(r"[A-Za-z_]\w+", query):
        if token in seen_names:
            continue
        seen_names.add(token)
        matches = db.lookup_symbols(token)
        exact = next((symbol for symbol in matches if symbol.get("name") == token), None)
        if exact is None:
            continue
        outgoing = db.get_refs(symbol_id=exact["id"])[:_MAX_GRAPH_EDGES]
        callers = db.get_callers(symbol_name=token)[:_MAX_GRAPH_EDGES]
        graph.append({
            "symbol": {
                "name": exact["name"],
                "kind": exact["kind"],
                "file_id": exact["file_id"],
                "line": exact["line"],
                "signature": exact.get("signature", ""),
            },
            "outgoing": [
                {
                    "symbol": ref.get("to_symbol_name", ""),
                    "kind": ref.get("kind", ""),
                    "line": ref.get("line", 0),
                }
                for ref in outgoing
            ],
            "callers": [
                {
                    "name": caller.get("name", ""),
                    "kind": caller.get("kind", ""),
                    "file_id": caller.get("file_id"),
                    "line": caller.get("line", 0),
                }
                for caller in callers
            ],
        })
        if len(graph) == _MAX_GRAPH_SYMBOLS:
            break
    return graph


def register_explore_tools(mcp: FastMCP) -> None:
    """Register the compact default exploration entry point."""

    @mcp.tool()
    async def explore(query: str, max_results: int = 5, session_id: str = "") -> str:
        """Investigate a codebase question in one bounded, agent-ready context pack.

        Use this as the default starting point for finding an implementation,
        understanding how a feature works, or assessing a likely change area.
        It combines hybrid code search with per-file symbol outlines and exact
        reference/caller slices when identifiers in the query match symbols.

        Args:
            query: Natural-language question or identifier to investigate.
            max_results: Maximum ranked code results (1-10; default 5).
            session_id: Optional scope-gated session token.
        """
        from .._state import _embedding_client

        scope, err = _check_session({"session_id": session_id}, "project")
        if err:
            return err
        agent_id = scope.agent_id if scope else "dev"
        dbs = _get_project_dbs(scope)
        if not dbs:
            _log_audit("explore", 0, agent_id=agent_id)
            return "Error: No accessible projects"

        limit = max(1, min(max_results, _MAX_RESULTS))
        query_embedding = None
        if _embedding_client:
            try:
                import numpy as np

                raw = await asyncio.to_thread(_embedding_client.embed_query, query)
                query_embedding = np.array(raw, dtype=np.float32)
            except EmbeddingUnavailableError:
                logger.debug("Embedding endpoint unavailable; explore is using keyword retrieval")

        def collect_project(project_id: int, project_name: str, db: Any) -> dict[str, Any]:
            results = hybrid_search(
                query, query_embedding, db, None, limit,
                source_type=["code"], file_dedup=True,
            )
            packed_results = []
            files: list[dict[str, Any]] = []
            seen_file_ids: set[int] = set()
            for result in results:
                snippet = extract_snippet(result.get("content", ""), query)
                packed_results.append({
                    "project_id": project_id,
                    "project_name": project_name,
                    "path": result.get("file_path", ""),
                    "start_line": result.get("start_line", 0) + 1,
                    "end_line": result.get("end_line", 0) + 1,
                    "score": round(float(result.get("score", 0.0)), 6),
                    "snippet": snippet.get("snippet", ""),
                    "source_type": result.get("source_type", "code"),
                })
                file_id = result.get("file_id")
                if not file_id or file_id in seen_file_ids or len(files) >= _MAX_FILES:
                    continue
                seen_file_ids.add(file_id)
                file_info = db.get_file(file_id=file_id)
                if file_info:
                    files.append({
                        "project_id": project_id,
                        "project_name": project_name,
                        "path": file_info.get("path", ""),
                        "language": file_info.get("language", ""),
                        "symbols": _file_outline(db, file_id),
                    })
            return {
                "results": packed_results,
                "files": files,
                "symbols": _symbol_graph(db, query),
            }

        try:
            tasks = [
                asyncio.to_thread(collect_project, project_id, project_name, db)
                for project_id, project_name, db in dbs
            ]
            project_packs = await asyncio.gather(*tasks, return_exceptions=True)
            results: list[dict[str, Any]] = []
            files: list[dict[str, Any]] = []
            symbols: list[dict[str, Any]] = []
            for pack in project_packs:
                if isinstance(pack, BaseException):
                    logger.warning("Explore query failed for a project: %s", pack)
                    continue
                results.extend(pack["results"])
                files.extend(pack["files"])
                symbols.extend(pack["symbols"])

            results.sort(key=lambda item: item["score"], reverse=True)
            payload = {
                "query": query,
                "results": results[:limit],
                "files": files[:_MAX_FILES],
                "symbols": symbols[:_MAX_GRAPH_SYMBOLS],
                "next_steps": [
                    "Start with the highest-ranked snippet and file outline.",
                    "Use the advanced tool profile only when you need deeper, targeted analysis.",
                ],
            }
            _log_audit("explore", len(payload["results"]), agent_id=agent_id)
            return json.dumps(payload, indent=2)
        except Exception as exc:
            logger.exception("Explore tool error")
            _log_audit("explore", 0, agent_id=agent_id)
            return f"Error exploring codebase: {exc}"
