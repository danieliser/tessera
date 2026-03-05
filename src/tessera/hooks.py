"""Tessera hook system — WordPress-style async hooks for pipeline extensibility.

Uses async_hooks.AsyncHooks to provide filter and action hooks at key pipeline
points. Model profiles register/skip hooks based on empirical benchmark results.

Hook points:
    tessera.embedding_texts  (filter) — transform chunk texts before embedding
    tessera.chunk_results    (filter) — modify chunks after AST chunking
    tessera.before_index_file (action) — fired before processing a file
    tessera.after_index_file  (action) — fired after processing a file
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from async_hooks import AsyncHooks

if TYPE_CHECKING:
    from .model_profiles import ModelProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook name constants
# ---------------------------------------------------------------------------

EMBEDDING_TEXTS = "tessera.embedding_texts"
CHUNK_RESULTS = "tessera.chunk_results"
BEFORE_INDEX_FILE = "tessera.before_index_file"
AFTER_INDEX_FILE = "tessera.after_index_file"

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_hooks: AsyncHooks | None = None


def get_hooks() -> AsyncHooks:
    """Get the global hooks manager, creating it if needed."""
    global _hooks
    if _hooks is None:
        _hooks = AsyncHooks(action_timeout_seconds=60, filter_timeout_seconds=None)
    return _hooks


def reset_hooks() -> None:
    """Reset the global hooks manager. For testing."""
    global _hooks
    _hooks = AsyncHooks(action_timeout_seconds=60, filter_timeout_seconds=None)


# ---------------------------------------------------------------------------
# Built-in filter: scope prefix
# ---------------------------------------------------------------------------

def _scope_prefix_filter(texts: list[str], *, chunks: list, symbols: list, rel_path: str) -> list[str]:
    """Prepend natural language scope context to chunk texts before embedding.

    Example output: "Cookies class, set_cookie method in Cookies.php: <code>"

    Registered on tessera.embedding_texts when profile.scope_prefix is True.
    """
    if not symbols:
        return texts

    sym_ranges = sorted(
        [(s.line, s.end_line or s.line, s.name, s.kind, s.scope) for s in symbols],
        key=lambda x: (x[0], -(x[1] - x[0])),
    )

    filename = os.path.basename(rel_path)
    result = []

    for text, chunk in zip(texts, chunks):
        overlapping = [
            (name, kind, scope)
            for start, end, name, kind, scope in sym_ranges
            if start <= chunk.end_line and end >= chunk.start_line
        ]

        if not overlapping:
            result.append(text)
            continue

        best_method = None
        best_class = None
        for name, kind, scope in overlapping:
            if kind in ("method", "function") and not best_method:
                best_method = (name, scope)
            elif kind == "class" and not best_class:
                best_class = name

        parts = []
        if best_class:
            parts.append(f"{best_class} class")
        elif best_method and best_method[1]:
            parts.append(f"{best_method[1]} class")
        if best_method:
            parts.append(f"{best_method[0]} method")

        if parts:
            prefix = ", ".join(parts) + f" in {filename}: "
        else:
            prefix = f"{filename}: "

        result.append(prefix + text)

    return result


# ---------------------------------------------------------------------------
# Profile-driven hook setup
# ---------------------------------------------------------------------------

_registered_callback_ids: list[str] = []


def setup_model_hooks(profile: ModelProfile | None) -> None:
    """Register built-in hooks based on a model profile's optimization flags.

    Clears any previously registered built-in hooks first, so this is safe
    to call multiple times (e.g., when switching models).
    """
    hooks = get_hooks()

    # Clear previous built-in registrations
    for cb_id in _registered_callback_ids:
        hooks.remove_filter(EMBEDDING_TEXTS, cb_id)
    _registered_callback_ids.clear()

    if profile is None:
        return

    if profile.scope_prefix:
        cb_id = hooks.add_filter(EMBEDDING_TEXTS, _scope_prefix_filter, priority=10)
        _registered_callback_ids.append(cb_id)
        logger.info(
            "Scope prefix filter enabled for %s (dims=%d, tokens=%d)",
            profile.display_name, profile.dimensions, profile.max_tokens,
        )
    else:
        logger.debug(
            "Scope prefix filter disabled for %s",
            profile.display_name,
        )
