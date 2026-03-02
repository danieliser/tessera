"""SQLite database layer for Tessera.

Manages both global (~/.tessera/global.db) and per-project (.tessera/index.db)
SQLite databases.
"""

from tessera.db._global import GlobalDB
from tessera.db._project import ProjectDB
from tessera.db._utils import normalize_and_validate, sanitize_fts5_query

__all__ = [
    "GlobalDB",
    "ProjectDB",
    "normalize_and_validate",
    "sanitize_fts5_query",
]
