"""Shared database utilities: path validation and FTS5 query sanitization."""

import logging
from pathlib import Path

from ..exceptions import PathTraversalError

logger = logging.getLogger(__name__)


def normalize_and_validate(project_root: Path, user_path: str) -> Path:
    """Normalize and validate a path to prevent traversal attacks.

    Args:
        project_root: The allowed base directory
        user_path: User-provided path (relative or absolute)

    Returns:
        Validated absolute path

    Raises:
        PathTraversalError: If path escapes project_root
    """
    project_root = Path(project_root).resolve()
    full_path = (project_root / user_path).resolve()

    # Ensure path is within project_root
    try:
        full_path.relative_to(project_root)
    except ValueError:
        raise PathTraversalError(
            f"Path {user_path} escapes project root {project_root}"
        ) from None

    return full_path


def sanitize_fts5_query(query: str, allow_advanced: bool = False) -> str:
    """Sanitize a query string for safe use with FTS5 MATCH.

    Args:
        query: Raw query string
        allow_advanced: If True, preserve recognized FTS5 syntax
            (quoted phrases, NOT, *, NEAR). If False, escape everything.

    Returns:
        Safe FTS5 query string
    """
    if not query or not query.strip():
        return ""

    query = query.strip()

    if allow_advanced:
        # Pass through recognized FTS5 syntax, only escape unbalanced parens
        # Replace unmatched parens that would break FTS5
        # Count parens — if unbalanced, strip them all
        if query.count("(") != query.count(")"):
            query = query.replace("(", " ").replace(")", " ")
        return query

    # Basic mode: wrap each token in double quotes to escape all operators
    # This makes "error NOT warning" search for all three words literally
    # and prevents "hybrid*" from being a prefix search
    tokens = query.split()
    quoted = []
    for token in tokens:
        # Strip existing quotes to avoid double-quoting
        clean = token.replace('"', '')
        if clean:
            quoted.append(f'"{clean}"')
    return " ".join(quoted)
