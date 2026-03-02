"""GlobalDB: project registration, collections, sessions, audit log."""

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GlobalDB:
    """Manages ~/.tessera/global.db for project metadata, collections, sessions."""

    def __init__(self, db_path: str = None):
        """Initialize global database connection.

        Args:
            db_path: Path to global.db. Defaults to ~/.tessera/global.db
        """
        if db_path is None:
            tessera_dir = Path.home() / ".tessera"
            tessera_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(tessera_dir / "global.db")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

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
    ) -> dict[str, Any] | None:
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

    def list_projects(self) -> list[dict[str, Any]]:
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

    def get_pending_jobs(self) -> list[dict[str, Any]]:
        """Get all pending or running jobs (for crash recovery).

        Returns jobs with status 'pending' or 'running' — running jobs
        at startup indicate a crash during indexing.
        """
        cursor = self.conn.execute(
            "SELECT * FROM indexing_jobs WHERE status IN ('pending', 'running') ORDER BY id"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_job_status(self, job_id: int) -> dict[str, Any] | None:
        """Get status of a specific job."""
        cursor = self.conn.execute(
            "SELECT * FROM indexing_jobs WHERE id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_incomplete_jobs(self) -> list[dict[str, Any]]:
        """Get all jobs with status 'running' (interrupted at crash time).

        Returns:
            List of dicts with 'id' (job_id) and 'project_id' keys.
        """
        cursor = self.conn.execute(
            "SELECT id, project_id FROM indexing_jobs WHERE status = 'running' ORDER BY id"
        )
        return [{"id": row["id"], "project_id": row["project_id"]} for row in cursor.fetchall()]

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

    def get_audit_log(self, limit: int = 100, agent_id: str | None = None) -> list[dict[str, Any]]:
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

    # --- Collection CRUD methods ---

    def create_collection(self, name: str, project_ids: list[int] = None) -> int:
        """Create a new collection, optionally assigning projects.

        Args:
            name: Unique collection name
            project_ids: Optional list of project IDs to assign

        Returns:
            Collection ID

        Raises:
            sqlite3.IntegrityError: If name already exists (UNIQUE constraint)
        """
        scope_id = str(uuid.uuid4())
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO collections (name, scope_id) VALUES (?, ?)",
                (name, scope_id)
            )
            collection_id = cursor.lastrowid
            if project_ids:
                for project_id in project_ids:
                    self.conn.execute(
                        "UPDATE projects SET collection_id = ? WHERE id = ?",
                        (collection_id, project_id)
                    )
        return collection_id

    def get_collection(self, collection_id: int) -> dict[str, Any] | None:
        """Retrieve a collection with its metadata and member projects.

        Args:
            collection_id: Collection ID

        Returns:
            Dict with {id, name, scope_id, created_at, projects: [...]}, or None if not found
        """
        cursor = self.conn.execute(
            "SELECT id, name, scope_id, created_at FROM collections WHERE id = ?",
            (collection_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        coll = dict(row)
        coll["projects"] = self.get_collection_projects(collection_id)
        return coll

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections ordered by name.

        Returns:
            List of collection dicts, each with {id, name, scope_id, created_at, projects: [...]}
        """
        cursor = self.conn.execute(
            "SELECT id, name, scope_id, created_at FROM collections ORDER BY name"
        )
        collections = []
        for row in cursor.fetchall():
            coll = dict(row)
            coll["projects"] = self.get_collection_projects(coll["id"])
            collections.append(coll)
        return collections

    def get_collection_projects(self, collection_id: int) -> list[dict[str, Any]]:
        """Get member projects of a collection.

        Args:
            collection_id: Collection ID

        Returns:
            List of project dicts {id, name, path, language} ordered by name
        """
        cursor = self.conn.execute(
            "SELECT id, name, path, language FROM projects WHERE collection_id = ? ORDER BY name",
            (collection_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def add_project_to_collection(self, collection_id: int, project_id: int) -> None:
        """Add a project to a collection.

        Args:
            collection_id: Collection ID
            project_id: Project ID

        Raises:
            ValueError: If collection_id or project_id do not exist
        """
        # Validate collection exists
        cursor = self.conn.execute(
            "SELECT id FROM collections WHERE id = ?",
            (collection_id,)
        )
        if not cursor.fetchone():
            raise ValueError(f"Collection {collection_id} not found")

        # Validate project exists
        cursor = self.conn.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,)
        )
        if not cursor.fetchone():
            raise ValueError(f"Project {project_id} not found")

        # Update project's collection_id FK
        with self.conn:
            self.conn.execute(
                "UPDATE projects SET collection_id = ? WHERE id = ?",
                (collection_id, project_id)
            )

    def delete_collection(self, collection_id: int) -> None:
        """Delete a collection and clear its project assignments.

        Args:
            collection_id: Collection ID
        """
        with self.conn:
            # Clear collection_id from member projects (SET NULL)
            self.conn.execute(
                "UPDATE projects SET collection_id = NULL WHERE collection_id = ?",
                (collection_id,)
            )
            # Delete the collection row
            self.conn.execute(
                "DELETE FROM collections WHERE id = ?",
                (collection_id,)
            )

    def close(self):
        """Close the global database connection."""
        self.conn.close()


