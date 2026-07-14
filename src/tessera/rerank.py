"""Bounded candidate preparation for two-stage retrieval and reranking."""

from __future__ import annotations

import json
from typing import Any

RERANK_POOL_MULTIPLIER = 3
MAX_RERANK_POOL_SIZE = 50
RERANK_FETCH_MULTIPLIER = 5
MAX_RERANK_FETCH_SIZE = 250
MAX_RERANK_DOCUMENT_CHARS = 4096
MAX_RERANK_SYMBOLS = 8
RERANK_POOL_SELECTION = "file_coverage_first"


def rerank_candidate_budget(limit: int, *, reranker_active: bool) -> int:
    """Return the project-local retrieval budget for a visible result limit.

    The multiplier preserves the server's pre-existing nominal ``3 * limit``
    rerank pool. The cap bounds extra cross-encoder work but never lowers an
    explicitly requested visible limit.
    """
    if limit <= 0 or not reranker_active:
        return limit
    return max(limit, min(limit * RERANK_POOL_MULTIPLIER, MAX_RERANK_POOL_SIZE))


def rerank_retrieval_budget(candidate_budget: int, *, reranker_active: bool) -> int:
    """Return a bounded raw fetch depth from which to build a diverse pool.

    Five matches the existing over-fetch policy used by ``hybrid_search`` when
    it must find unique files. Here the raw candidates remain available for the
    explicit coverage-first selector, so second chunks can backfill the pool.
    """
    if candidate_budget <= 0 or not reranker_active:
        return candidate_budget
    return max(
        candidate_budget,
        min(candidate_budget * RERANK_FETCH_MULTIPLIER, MAX_RERANK_FETCH_SIZE),
    )


def _project_key(result: dict[str, Any]) -> tuple[str, Any]:
    if result.get("project_id") is not None:
        return ("id", result["project_id"])
    return ("name", result.get("project_name", ""))


def _file_key(result: dict[str, Any], position: int) -> tuple[Any, ...]:
    project = _project_key(result)
    if result.get("file_id") is not None:
        return (*project, "file_id", result["file_id"])
    path = str(result.get("file_path", "")).replace("\\", "/")
    if path:
        return (*project, "path", path)
    if result.get("id") is not None:
        return (*project, "chunk", result["id"])
    return (*project, "position", position)


def select_rerank_candidates(
    results: list[dict[str, Any]],
    pool_size: int,
) -> list[dict[str, Any]]:
    """Select a stable coverage-first pool without hiding useful second chunks.

    The input is already in retrieval-score order. The first pass takes the
    best candidate for every project-local file. If capacity remains, a second
    pass fills it in the original order, which permits multiple chunks from a
    file without letting them starve all other files.
    """
    if pool_size <= 0:
        return []

    selected_positions: list[int] = []
    selected_set: set[int] = set()
    seen_files: set[tuple[Any, ...]] = set()
    for position, result in enumerate(results):
        key = _file_key(result, position)
        if key in seen_files:
            continue
        seen_files.add(key)
        selected_positions.append(position)
        selected_set.add(position)
        if len(selected_positions) == pool_size:
            return [results[index] for index in selected_positions]

    for position in range(len(results)):
        if position in selected_set:
            continue
        selected_positions.append(position)
        if len(selected_positions) == pool_size:
            break

    return [results[index] for index in selected_positions]


def _parse_symbol_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(value, (list, tuple)):
        return []

    symbol_ids: list[int] = []
    seen: set[int] = set()
    for raw_id in value:
        try:
            symbol_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if symbol_id in seen:
            continue
        seen.add(symbol_id)
        symbol_ids.append(symbol_id)
        if len(symbol_ids) == MAX_RERANK_SYMBOLS:
            break
    return symbol_ids


def _symbol_context(result: dict[str, Any], db: Any | None) -> list[str]:
    if db is None or result.get("id") is None:
        return []
    try:
        chunk = db.get_chunk(result["id"])
    except Exception:
        return []
    if not chunk:
        return []

    context: list[str] = []
    for symbol_id in _parse_symbol_ids(chunk.get("symbol_ids")):
        try:
            symbol = db.get_symbol(symbol_id)
        except Exception:
            continue
        if not symbol:
            continue
        name = str(symbol.get("name", "")).strip()
        scope = str(symbol.get("scope", "")).strip()
        qualified_name = f"{scope}.{name}" if name and scope and scope not in {"(module)", "module"} else name
        kind = str(symbol.get("kind", "symbol")).strip() or "symbol"
        signature = " ".join(str(symbol.get("signature", "")).split())
        label = f"{kind} {qualified_name}".strip()
        if signature and signature != qualified_name:
            label = f"{label} :: {signature}"
        if label:
            context.append(label)
    return context


def _line_range(result: dict[str, Any]) -> str:
    start = result.get("start_line")
    end = result.get("end_line")
    if isinstance(start, int):
        start += 1
    if isinstance(end, int):
        end += 1
    if start is None and end is None:
        return "unknown"
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def build_rerank_document(
    result: dict[str, Any],
    db: Any | None = None,
    *,
    max_chars: int = MAX_RERANK_DOCUMENT_CHARS,
) -> str:
    """Build one bounded reranker document with stable provenance context."""
    symbols = _symbol_context(result, db)
    lines = [
        "[Tessera candidate]",
        f"Project: {result.get('project_name') or result.get('project_id') or 'unknown'}",
        f"Path: {result.get('file_path') or 'unknown'}",
        f"Source: {result.get('source_type') or 'unknown'}",
        f"Lines: {_line_range(result)}",
    ]
    if symbols:
        lines.append(f"Symbols: {' | '.join(symbols)}")
    if result.get("section_heading"):
        lines.append(f"Section: {result['section_heading']}")
    if result.get("key_path"):
        lines.append(f"Key: {result['key_path']}")

    content = str(result.get("content") or result.get("snippet") or "")
    document = "\n".join([*lines, "", "Content:", content])
    return document[: max(0, max_chars)]


def build_rerank_documents(
    results: list[dict[str, Any]],
    db_by_project: dict[Any, Any],
) -> list[str]:
    """Build bounded documents for a federated candidate pool."""
    return [build_rerank_document(result, db_by_project.get(result.get("project_id"))) for result in results]
