"""Shared database utilities: path validation and FTS5 query sanitization."""

import logging
import re
from pathlib import Path

from ..exceptions import PathTraversalError

logger = logging.getLogger(__name__)

PATH_QUERY_TOKEN_LIMIT = 32

_IDENTIFIER_ATOM_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CAMEL_ACRONYM_BOUNDARY_RE = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_LETTER_DIGIT_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def identifier_search_tokens(value: str) -> list[str]:
    """Return stable exact and style-independent tokens for an identifier.

    The transformation is deliberately language-agnostic. It preserves each
    case-folded alphanumeric atom, splits camel/acronym/digit boundaries, and
    adds the compact spelling so camel, snake, and kebab forms can meet in the
    same sparse index.
    """
    atoms = _IDENTIFIER_ATOM_RE.findall(value)
    tokens = [atom.casefold() for atom in atoms]
    pieces: list[str] = []
    for atom in atoms:
        expanded = _CAMEL_ACRONYM_BOUNDARY_RE.sub(" ", atom)
        expanded = _CAMEL_CASE_BOUNDARY_RE.sub(" ", expanded)
        expanded = _LETTER_DIGIT_BOUNDARY_RE.sub(" ", expanded)
        pieces.extend(piece.casefold() for piece in _IDENTIFIER_ATOM_RE.findall(expanded))
    tokens.extend(pieces)
    if pieces:
        tokens.append("".join(pieces))
    return _ordered_unique(tokens)


def build_path_search_fields(path: str) -> tuple[str, str, str, str, str, str]:
    """Build the six deterministic text views stored by ``file_paths_fts``."""
    normalized = path.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.strip("/")

    components = [component for component in normalized.split("/") if component]
    basename = components[-1] if components else normalized
    if basename.startswith(".") and basename.count(".") == 1:
        stem = basename
    elif "." in basename:
        stem = basename.rsplit(".", 1)[0]
    else:
        stem = basename
    directories = " ".join(components[:-1])
    module_components = [*components[:-1], stem] if stem else components[:-1]
    module = ".".join(module_components)

    aliases: list[str] = []
    for component in components:
        aliases.extend(identifier_search_tokens(component))
    aliases.extend(identifier_search_tokens(stem))
    alias_text = " ".join(_ordered_unique(aliases))
    return normalized, basename, stem, directories, module, alias_text


def build_path_fts_query(query: str) -> str:
    """Build a bounded literal OR query for the path FTS channel."""
    tokens = identifier_search_tokens(query)[:PATH_QUERY_TOKEN_LIMIT]
    return " OR ".join(f'"{token}"' for token in tokens)


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
