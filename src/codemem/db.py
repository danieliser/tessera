"""SQLite database layer for CodeMem Phase 1.

Manages both global (~/.codemem/global.db) and per-project (.codemem/index.db)
SQLite databases. All queries use parameterized placeholders to prevent SQL injection.

Architecture:
  - GlobalDB: Project registration, collections, sessions, audit log
  - ProjectDB: Files, symbols, references, edges, chunks, embeddings, FTS5
"""

import sqlite3
import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


class PathTraversalError(Exception):
    """Raised when a path escapes the project root."""
    pass


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
        )

    return full_path


class GlobalDB:
    """Manages ~/.codemem/global.db for project metadata, collections, sessions."""

    def __init__(self, db_path: str = None):
        """Initialize global database connection.

        Args:
            db_path: Path to global.db. Defaults to ~/.codemem/global.db
        """
        if db_path is None:
            codemem_dir = Path.home() / ".codemem"
            codemem_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(codemem_dir / "global.db")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")

        self._create_schema()

    def _create_schema(self):
        """Create global database schema if it doesn't exist."""
        with self.conn:
            # Collections table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    projects_json TEXT,
                    scope_id TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Projects table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    language TEXT,
                    collection_id INTEGER,
                    scope_id TEXT UNIQUE,
                    indexed_at TIMESTAMP,
                    last_indexed_commit TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(collection_id) REFERENCES collections(id)
                )
            """)

            # Sessions table (schema matches auth.py create_scope/validate_session)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    projects_list TEXT,
                    collections_list TEXT,
                    capabilities TEXT,
                    created_at TEXT NOT NULL,
                    valid_until TEXT NOT NULL
                )
            """)

            # Indexing jobs table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS indexing_jobs (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error TEXT,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                )
            """)

            # Audit log table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    scope_level TEXT NOT NULL,
                    tool_called TEXT NOT NULL,
                    result_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def register_project(
        self,
        path: str,
        name: str,
        language: str = None,
        collection_id: int = None
    ) -> int:
        """Register a new project in the global database.

        Args:
            path: Absolute path to project
            name: Project name
            language: Programming language (optional)
            collection_id: ID of parent collection (optional)

        Returns:
            Project ID
        """
        scope_id = str(uuid.uuid4())

        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO projects (path, name, language, collection_id, scope_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (path, name, language, collection_id, scope_id)
            )
            return cursor.lastrowid

    def get_project(
        self,
        project_id: int = None,
        path: str = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a project by ID or path.

        Args:
            project_id: Project ID
            path: Project path

        Returns:
            Project record as dict, or None if not found
        """
        if project_id is not None:
            cursor = self.conn.execute(
                "SELECT * FROM projects WHERE id = ?",
                (project_id,)
            )
        elif path is not None:
            cursor = self.conn.execute(
                "SELECT * FROM projects WHERE path = ?",
                (path,)
            )
        else:
            raise ValueError("Must provide project_id or path")

        row = cursor.fetchone()
        return dict(row) if row else None

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all registered projects.

        Returns:
            List of project records as dicts
        """
        cursor = self.conn.execute("SELECT * FROM projects ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]

    def update_last_indexed_commit(self, project_id: int, commit_hash: str) -> None:
        """Store the last indexed commit hash for a project."""
        with self.conn:
            self.conn.execute(
                "UPDATE projects SET last_indexed_commit = ?, indexed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (commit_hash, project_id)
            )

    # --- Job queue methods ---

    def create_job(self, project_id: int) -> int:
        """Create a new indexing job.

        Args:
            project_id: ID of the project to index

        Returns:
            Job ID
        """
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO indexing_jobs (project_id, status) VALUES (?, 'pending')",
                (project_id,)
            )
            return cursor.lastrowid

    def start_job(self, job_id: int) -> None:
        """Mark a job as running."""
        with self.conn:
            self.conn.execute(
                "UPDATE indexing_jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job_id,)
            )

    def complete_job(self, job_id: int) -> None:
        """Mark a job as completed."""
        with self.conn:
            self.conn.execute(
                "UPDATE indexing_jobs SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job_id,)
            )

    def fail_job(self, job_id: int, error: str) -> None:
        """Mark a job as failed.

        Args:
            job_id: Job ID
            error: Error message
        """
        with self.conn:
            self.conn.execute(
                "UPDATE indexing_jobs SET status = 'failed', error = ?, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (error, job_id)
            )

    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get all pending or running jobs (for crash recovery).

        Returns jobs with status 'pending' or 'running' — running jobs
        at startup indicate a crash during indexing.
        """
        cursor = self.conn.execute(
            "SELECT * FROM indexing_jobs WHERE status IN ('pending', 'running') ORDER BY id"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        cursor = self.conn.execute(
            "SELECT * FROM indexing_jobs WHERE id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def reset_crashed_jobs(self) -> int:
        """Reset any 'running' jobs to 'pending' (crash recovery on startup).

        Returns:
            Number of jobs reset
        """
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE indexing_jobs SET status = 'pending', started_at = NULL WHERE status = 'running'"
            )
            return cursor.rowcount

    # --- Audit log methods ---

    def insert_audit(self, agent_id: str, scope_level: str, tool_called: str, result_count: int) -> None:
        """Record an audit log entry.

        Args:
            agent_id: ID of the agent making the call
            scope_level: Scope level (project, collection, global)
            tool_called: Name of the tool called
            result_count: Number of results returned
        """
        with self.conn:
            self.conn.execute(
                """INSERT INTO audit_log (agent_id, scope_level, tool_called, result_count)
                   VALUES (?, ?, ?, ?)""",
                (agent_id, scope_level, tool_called, result_count)
            )

    def get_audit_log(self, limit: int = 100, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query the audit log.

        Args:
            limit: Max records to return
            agent_id: Filter by agent ID (optional)

        Returns:
            List of audit log records
        """
        if agent_id:
            cursor = self.conn.execute(
                "SELECT * FROM audit_log WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?",
                (agent_id, limit)
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close the global database connection."""
        self.conn.close()


class ProjectDB:
    """Manages <project>/.codemem/index.db for symbols, references, and embeddings."""

    def __init__(self, project_path: str):
        """Initialize project database connection.

        Args:
            project_path: Absolute path to the project root
        """
        project_path = Path(project_path)
        codemem_dir = project_path / ".codemem"
        codemem_dir.mkdir(parents=True, exist_ok=True)

        self.project_path = project_path
        self.db_path = str(codemem_dir / "index.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")

        self._create_schema()

    def _create_schema(self):
        """Create project database schema if it doesn't exist."""
        with self.conn:
            # Files table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    language TEXT,
                    hash TEXT,
                    index_status TEXT DEFAULT 'pending',
                    indexed_at TIMESTAMP,
                    UNIQUE(project_id, path)
                )
            """)

            # Symbols table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    file_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    line INTEGER,
                    col INTEGER,
                    scope TEXT,
                    signature TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(id)
                )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_project_name "
                "ON symbols(project_id, name)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id)"
            )

            # Refs table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS refs (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    from_symbol_id INTEGER NOT NULL,
                    to_symbol_id INTEGER,
                    kind TEXT NOT NULL,
                    context TEXT,
                    line INTEGER,
                    FOREIGN KEY(from_symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY(to_symbol_id) REFERENCES symbols(id)
                )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_refs_from ON refs(from_symbol_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_refs_to ON refs(to_symbol_id)"
            )

            # Edges table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    from_id INTEGER NOT NULL,
                    to_id INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    FOREIGN KEY(from_id) REFERENCES symbols(id),
                    FOREIGN KEY(to_id) REFERENCES symbols(id)
                )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_from_to "
                "ON edges(from_id, to_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_to_from "
                "ON edges(to_id, from_id)"
            )

            # Chunk metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_meta (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    file_id INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    symbol_ids TEXT,
                    ast_type TEXT,
                    chunk_type TEXT,
                    content TEXT NOT NULL,
                    length INTEGER,
                    FOREIGN KEY(file_id) REFERENCES files(id)
                )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunk_meta(file_id)"
            )

            # FTS5 virtual table for full-text search
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    chunk_id UNINDEXED,
                    file_path UNINDEXED
                )
            """)

            # Embeddings table (BLOBs)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunk_meta(id)
                )
            """)

    def validate_path(self, file_path: str) -> str:
        """Validate and normalize a file path within the project.

        Args:
            file_path: File path relative or absolute

        Returns:
            Validated absolute path as string

        Raises:
            PathTraversalError: If path escapes project root
        """
        normalized = normalize_and_validate(self.project_path, file_path)
        return str(normalized)

    # File operations

    def upsert_file(
        self,
        project_id: int,
        path: str,
        language: str,
        file_hash: str
    ) -> int:
        """Upsert a file record (insert or update).

        Args:
            project_id: Project ID
            path: File path
            language: Programming language
            file_hash: Content hash

        Returns:
            File ID
        """
        # Validate path
        self.validate_path(path)

        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO files (project_id, path, language, hash, index_status)
                VALUES (?, ?, ?, ?, 'pending')
                ON CONFLICT(project_id, path) DO UPDATE SET
                    language = excluded.language,
                    hash = excluded.hash,
                    index_status = 'pending'
                RETURNING id
                """,
                (project_id, path, language, file_hash)
            )
            return cursor.fetchone()[0]

    def get_file(
        self,
        file_id: int = None,
        path: str = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a file record.

        Args:
            file_id: File ID
            path: File path

        Returns:
            File record as dict, or None if not found
        """
        if file_id is not None:
            cursor = self.conn.execute(
                "SELECT * FROM files WHERE id = ?",
                (file_id,)
            )
        elif path is not None:
            cursor = self.conn.execute(
                "SELECT * FROM files WHERE path = ?",
                (path,)
            )
        else:
            raise ValueError("Must provide file_id or path")

        row = cursor.fetchone()
        return dict(row) if row else None

    def update_file_status(self, file_id: int, status: str):
        """Update file indexing status.

        Args:
            file_id: File ID
            status: New status (e.g., 'indexed', 'error')
        """
        with self.conn:
            self.conn.execute(
                """
                UPDATE files
                SET index_status = ?, indexed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, file_id)
            )

    def delete_file_data(self, rel_path: str) -> None:
        """Delete all indexed data for a file by relative path.

        Removes the file record and cascading data (symbols, references, chunks).

        Args:
            rel_path: Relative file path within the project
        """
        with self.conn:
            # Find the file
            cursor = self.conn.execute("SELECT id FROM files WHERE path = ?", (rel_path,))
            row = cursor.fetchone()
            if not row:
                return

            file_id = row["id"]
            # Delete chunks and their embeddings
            self.conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT id FROM chunk_meta WHERE file_id = ?)", (file_id,))
            self.conn.execute("DELETE FROM chunk_meta WHERE file_id = ?", (file_id,))
            # Delete refs linked to this file's symbols
            self.conn.execute("""
                DELETE FROM refs WHERE from_symbol_id IN (SELECT id FROM symbols WHERE file_id = ?)
                   OR to_symbol_id IN (SELECT id FROM symbols WHERE file_id = ?)
            """, (file_id, file_id))
            self.conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
            # Delete edges involving this file's symbols
            self.conn.execute("""
                DELETE FROM edges WHERE from_id IN (SELECT id FROM symbols WHERE file_id = ?)
                   OR to_id IN (SELECT id FROM symbols WHERE file_id = ?)
            """, (file_id, file_id))
            # Delete the file record
            self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def get_pending_files(self) -> List[Dict[str, Any]]:
        """Get all files with 'pending' index status.

        Returns:
            List of file records
        """
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE index_status = 'pending' ORDER BY path"
        )
        return [dict(row) for row in cursor.fetchall()]

    # Symbol operations

    def insert_symbols(self, symbols: List[Dict[str, Any]]) -> List[int]:
        """Batch insert symbols.

        Args:
            symbols: List of symbol records with keys:
                - project_id, file_id, name, kind, line, col, scope, signature

        Returns:
            List of inserted symbol IDs
        """
        ids = []
        with self.conn:
            for sym in symbols:
                cursor = self.conn.execute(
                    """
                    INSERT INTO symbols (
                        project_id, file_id, name, kind, line, col, scope, signature
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sym.get("project_id"),
                        sym.get("file_id"),
                        sym.get("name"),
                        sym.get("kind"),
                        sym.get("line"),
                        sym.get("col"),
                        sym.get("scope"),
                        sym.get("signature")
                    )
                )
                ids.append(cursor.lastrowid)
        return ids

    def lookup_symbols(
        self,
        query: str = "*",
        kind: str = None,
        language: str = None,
        file_pattern: str = None,
        project_id: int = None
    ) -> List[Dict[str, Any]]:
        """Look up symbols with optional filtering.

        Args:
            query: Symbol name pattern (* for wildcard, or exact name)
            kind: Filter by symbol kind (function, class, variable, etc.)
            language: Filter by file language
            file_pattern: Filter by file path pattern
            project_id: Filter by project

        Returns:
            List of matching symbol records
        """
        sql = """
            SELECT s.* FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE 1=1
        """
        params = []

        if project_id is not None:
            sql += " AND s.project_id = ?"
            params.append(project_id)

        if query != "*":
            # Simple wildcard matching
            if "*" in query:
                pattern = query.replace("*", "%")
                sql += " AND s.name LIKE ?"
                params.append(pattern)
            else:
                sql += " AND s.name = ?"
                params.append(query)

        if kind is not None:
            sql += " AND s.kind = ?"
            params.append(kind)

        if language is not None:
            sql += " AND f.language = ?"
            params.append(language)

        if file_pattern is not None:
            pattern = file_pattern.replace("*", "%")
            sql += " AND f.path LIKE ?"
            params.append(pattern)

        sql += " ORDER BY s.name"

        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_symbol(self, symbol_id: int) -> Optional[Dict[str, Any]]:
        """Get a symbol by ID.

        Args:
            symbol_id: Symbol ID

        Returns:
            Symbol record as dict, or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM symbols WHERE id = ?",
            (symbol_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # Reference operations

    def insert_refs(self, refs: List[Dict[str, Any]]) -> List[int]:
        """Batch insert references.

        Args:
            refs: List of reference records with keys:
                - project_id, from_symbol_id, to_symbol_id, kind, context, line

        Returns:
            List of inserted reference IDs
        """
        ids = []
        with self.conn:
            for ref in refs:
                cursor = self.conn.execute(
                    """
                    INSERT INTO refs (
                        project_id, from_symbol_id, to_symbol_id, kind, context, line
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ref.get("project_id"),
                        ref.get("from_symbol_id"),
                        ref.get("to_symbol_id"),
                        ref.get("kind"),
                        ref.get("context"),
                        ref.get("line")
                    )
                )
                ids.append(cursor.lastrowid)
        return ids

    def get_refs(
        self,
        symbol_id: int = None,
        symbol_name: str = None,
        kind: str = "all"
    ) -> List[Dict[str, Any]]:
        """Get references for a symbol.

        Args:
            symbol_id: Symbol ID (for from_symbol_id)
            symbol_name: Symbol name (looks up by name)
            kind: Reference kind filter ("all" for no filter)

        Returns:
            List of reference records
        """
        sql = "SELECT * FROM refs WHERE 1=1"
        params = []

        if symbol_id is not None:
            sql += " AND from_symbol_id = ?"
            params.append(symbol_id)
        elif symbol_name is not None:
            # Subquery to find symbol by name
            sql += " AND from_symbol_id IN (SELECT id FROM symbols WHERE name = ?)"
            params.append(symbol_name)
        else:
            raise ValueError("Must provide symbol_id or symbol_name")

        if kind != "all":
            sql += " AND kind = ?"
            params.append(kind)

        sql += " ORDER BY line"

        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    # Edge operations

    def insert_edges(self, edges: List[Dict[str, Any]]) -> List[int]:
        """Batch insert edges in the call graph.

        Args:
            edges: List of edge records with keys:
                - project_id, from_id, to_id, type, weight

        Returns:
            List of inserted edge IDs
        """
        ids = []
        with self.conn:
            for edge in edges:
                cursor = self.conn.execute(
                    """
                    INSERT INTO edges (
                        project_id, from_id, to_id, type, weight
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        edge.get("project_id"),
                        edge.get("from_id"),
                        edge.get("to_id"),
                        edge.get("type"),
                        edge.get("weight", 1.0)
                    )
                )
                ids.append(cursor.lastrowid)
        return ids

    def get_forward_refs(
        self,
        symbol_id: int,
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get forward references (callees) for a symbol up to depth.

        Args:
            symbol_id: Starting symbol ID
            depth: Max traversal depth (number of edges to follow)

        Returns:
            List of reachable symbols (forward graph traversal)
        """
        # Simple 1-hop for depth=1; deeper queries use recursive CTE
        if depth == 1:
            cursor = self.conn.execute(
                """
                SELECT s.* FROM symbols s
                JOIN edges e ON e.to_id = s.id
                WHERE e.from_id = ?
                ORDER BY s.name
                """,
                (symbol_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        else:
            # Recursive CTE for arbitrary depth
            # depth=N means traverse up to N edges, so we need f.depth <= N
            cursor = self.conn.execute(
                """
                WITH RECURSIVE forward_refs AS (
                    SELECT id, 0 as depth FROM symbols WHERE id = ?
                    UNION ALL
                    SELECT e.to_id, f.depth + 1
                    FROM edges e
                    JOIN forward_refs f ON e.from_id = f.id
                    WHERE f.depth < ?
                )
                SELECT s.* FROM symbols s
                WHERE s.id IN (SELECT id FROM forward_refs WHERE depth > 0)
                ORDER BY s.name
                """,
                (symbol_id, depth)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_backward_refs(self, symbol_id: int) -> List[Dict[str, Any]]:
        """Get backward references (callers) for a symbol.

        Args:
            symbol_id: Symbol ID

        Returns:
            List of symbols that reference this one
        """
        cursor = self.conn.execute(
            """
            SELECT s.* FROM symbols s
            JOIN edges e ON e.from_id = s.id
            WHERE e.to_id = ?
            ORDER BY s.name
            """,
            (symbol_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # Chunk operations

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> List[int]:
        """Batch insert chunks with FTS5 and embeddings.

        Args:
            chunks: List of chunk records with keys:
                - project_id, file_id, start_line, end_line, symbol_ids, ast_type,
                  chunk_type, content, length, embedding (optional, np.ndarray)

        Returns:
            List of inserted chunk IDs
        """
        ids = []
        with self.conn:
            for chunk in chunks:
                # Insert into chunk_meta
                cursor = self.conn.execute(
                    """
                    INSERT INTO chunk_meta (
                        project_id, file_id, start_line, end_line, symbol_ids,
                        ast_type, chunk_type, content, length
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.get("project_id"),
                        chunk.get("file_id"),
                        chunk.get("start_line"),
                        chunk.get("end_line"),
                        json.dumps(chunk.get("symbol_ids", [])),
                        chunk.get("ast_type"),
                        chunk.get("chunk_type"),
                        chunk.get("content"),
                        chunk.get("length")
                    )
                )
                chunk_id = cursor.lastrowid
                ids.append(chunk_id)

                # Insert into FTS5
                file_info = self.get_file(file_id=chunk["file_id"])
                file_path = file_info["path"] if file_info else ""

                self.conn.execute(
                    """
                    INSERT INTO chunks_fts (content, chunk_id, file_path)
                    VALUES (?, ?, ?)
                    """,
                    (chunk.get("content"), chunk_id, file_path)
                )

                # Insert embedding if provided
                if "embedding" in chunk and chunk["embedding"] is not None:
                    embedding = chunk["embedding"]
                    if isinstance(embedding, np.ndarray):
                        embedding_bytes = embedding.astype(np.float32).tobytes()
                    else:
                        embedding_bytes = embedding

                    self.conn.execute(
                        """
                        INSERT INTO chunk_embeddings (chunk_id, embedding)
                        VALUES (?, ?)
                        """,
                        (chunk_id, embedding_bytes)
                    )

        return ids

    def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search on chunks using FTS5 BM25.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching chunks with score
        """
        cursor = self.conn.execute(
            """
            SELECT cm.*, fts.rank as score
            FROM chunks_fts fts
            JOIN chunk_meta cm ON fts.chunk_id = cm.id
            WHERE chunks_fts MATCH ?
            ORDER BY fts.rank DESC
            LIMIT ?
            """,
            (query, limit)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk record as dict, or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM chunk_meta WHERE id = ?",
            (chunk_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_embeddings(self) -> Tuple[List[int], np.ndarray]:
        """Load all chunk embeddings into a numpy matrix.

        Efficient for <20K chunks. Used for vector search.

        Returns:
            Tuple of (chunk_ids, embeddings_matrix) where embeddings_matrix
            is shape (n_chunks, embedding_dim) float32
        """
        cursor = self.conn.execute(
            """
            SELECT chunk_id, embedding FROM chunk_embeddings
            ORDER BY chunk_id
            """
        )

        rows = cursor.fetchall()
        if not rows:
            return [], np.array([], dtype=np.float32)

        chunk_ids = []
        embeddings = []

        for chunk_id, embedding_bytes in rows:
            chunk_ids.append(chunk_id)
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding)

        # Stack into matrix
        embeddings_matrix = np.stack(embeddings, axis=0)

        return chunk_ids, embeddings_matrix

    def clear_file_data(self, file_id: int):
        """Clear all data for a file (for re-indexing).

        Deletes symbols, refs, edges, chunks, and embeddings associated with file.

        Args:
            file_id: File ID
        """
        with self.conn:
            # Get all chunk IDs for this file
            cursor = self.conn.execute(
                "SELECT id FROM chunk_meta WHERE file_id = ?",
                (file_id,)
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]

            # Delete embeddings
            for chunk_id in chunk_ids:
                self.conn.execute(
                    "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
                    (chunk_id,)
                )

            # Delete from FTS5
            for chunk_id in chunk_ids:
                self.conn.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    (chunk_id,)
                )

            # Delete chunks
            self.conn.execute(
                "DELETE FROM chunk_meta WHERE file_id = ?",
                (file_id,)
            )

            # Get all symbol IDs for this file
            cursor = self.conn.execute(
                "SELECT id FROM symbols WHERE file_id = ?",
                (file_id,)
            )
            symbol_ids = [row[0] for row in cursor.fetchall()]

            # Delete edges referencing these symbols
            for symbol_id in symbol_ids:
                self.conn.execute(
                    "DELETE FROM edges WHERE from_id = ? OR to_id = ?",
                    (symbol_id, symbol_id)
                )

            # Delete refs
            for symbol_id in symbol_ids:
                self.conn.execute(
                    "DELETE FROM refs WHERE from_symbol_id = ? OR to_symbol_id = ?",
                    (symbol_id, symbol_id)
                )

            # Delete symbols
            self.conn.execute(
                "DELETE FROM symbols WHERE file_id = ?",
                (file_id,)
            )

    def close(self):
        """Close the project database connection."""
        self.conn.close()


if __name__ == "__main__":
    """Basic verification of database operations."""
    import tempfile
    import shutil

    # Test GlobalDB
    print("Testing GlobalDB...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "global.db")
        gdb = GlobalDB(db_path=db_path)

        # Register a project
        project_id = gdb.register_project(
            path="/path/to/project",
            name="test-project",
            language="python"
        )
        print(f"  Registered project with ID: {project_id}")

        # Retrieve project
        proj = gdb.get_project(project_id=project_id)
        assert proj is not None
        assert proj["name"] == "test-project"
        print(f"  Retrieved project: {proj['name']}")

        # List projects
        projects = gdb.list_projects()
        assert len(projects) == 1
        print(f"  Listed {len(projects)} project(s)")

        gdb.close()

    # Test ProjectDB
    print("\nTesting ProjectDB...")
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb = ProjectDB(tmpdir)

        # Upsert a file
        file_id = pdb.upsert_file(
            project_id=1,
            path="test.py",
            language="python",
            file_hash="abc123"
        )
        print(f"  Upserted file with ID: {file_id}")

        # Get file
        file_rec = pdb.get_file(file_id=file_id)
        assert file_rec is not None
        assert file_rec["path"] == "test.py"
        print(f"  Retrieved file: {file_rec['path']}")

        # Insert symbols
        symbol_ids = pdb.insert_symbols([
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "my_function",
                "kind": "function",
                "line": 10,
                "col": 0,
                "scope": "module",
                "signature": "def my_function(x, y)"
            },
            {
                "project_id": 1,
                "file_id": file_id,
                "name": "MyClass",
                "kind": "class",
                "line": 20,
                "col": 0,
                "scope": "module",
                "signature": "class MyClass"
            }
        ])
        print(f"  Inserted {len(symbol_ids)} symbols")

        # Look up symbol
        syms = pdb.lookup_symbols(query="my_function", kind="function")
        assert len(syms) == 1
        assert syms[0]["name"] == "my_function"
        print(f"  Looked up symbol: {syms[0]['name']}")

        # Insert references
        ref_ids = pdb.insert_refs([
            {
                "project_id": 1,
                "from_symbol_id": symbol_ids[0],
                "to_symbol_id": symbol_ids[1],
                "kind": "calls",
                "context": "instantiation",
                "line": 12
            }
        ])
        print(f"  Inserted {len(ref_ids)} reference(s)")

        # Get references
        refs = pdb.get_refs(symbol_id=symbol_ids[0])
        assert len(refs) == 1
        print(f"  Retrieved {len(refs)} reference(s)")

        # Insert edges
        edge_ids = pdb.insert_edges([
            {
                "project_id": 1,
                "from_id": symbol_ids[0],
                "to_id": symbol_ids[1],
                "type": "calls",
                "weight": 1.0
            }
        ])
        print(f"  Inserted {len(edge_ids)} edge(s)")

        # Get forward refs
        forward = pdb.get_forward_refs(symbol_ids[0], depth=1)
        assert len(forward) == 1
        print(f"  Forward refs: {len(forward)}")

        # Insert chunks with embeddings
        test_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        chunk_ids = pdb.insert_chunks([
            {
                "project_id": 1,
                "file_id": file_id,
                "start_line": 10,
                "end_line": 15,
                "symbol_ids": [symbol_ids[0]],
                "ast_type": "function_definition",
                "chunk_type": "function",
                "content": "def my_function(x, y):\n    return x + y",
                "length": 40,
                "embedding": test_embedding
            }
        ])
        print(f"  Inserted {len(chunk_ids)} chunk(s) with embedding")

        # Get all embeddings
        loaded_ids, embeddings = pdb.get_all_embeddings()
        assert len(loaded_ids) == 1
        assert embeddings.shape == (1, 4)
        print(f"  Loaded embeddings: shape {embeddings.shape}")

        # Keyword search
        results = pdb.keyword_search("function", limit=5)
        print(f"  Keyword search results: {len(results)}")

        # Test path validation
        try:
            validated = pdb.validate_path("subdir/file.py")
            print(f"  Validated path: {validated}")
        except PathTraversalError as e:
            print(f"  Path validation error: {e}")

        # Test path traversal protection
        try:
            pdb.validate_path("../../etc/passwd")
            print("  ERROR: Path traversal not caught!")
        except PathTraversalError:
            print("  Path traversal correctly blocked")

        # Clear file data
        pdb.clear_file_data(file_id)
        syms_after = pdb.lookup_symbols()
        assert len(syms_after) == 0
        print("  Cleared file data successfully")

        pdb.close()

    print("\nAll tests passed!")
