"""Tests for session token distribution and scope-gated access control.

Tests all functions in src/codemem/auth.py:
- create_scope: session token generation
- validate_session: token validation and scope retrieval
- check_scope: scope containment checks
- revoke_scope: agent scope revocation
- cleanup_expired_sessions: expired token cleanup
- normalize_and_validate_path: path traversal protection
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from codemem.auth import (
    create_scope,
    validate_session,
    check_scope,
    revoke_scope,
    cleanup_expired_sessions,
    normalize_and_validate_path,
    SessionNotFoundError,
    SessionExpiredError,
    ForbiddenScopeError,
    PathTraversalError,
    ScopeInfo,
)


@pytest.fixture
def db_conn():
    """Create an in-memory SQLite database with sessions table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE sessions (
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
    conn.commit()
    yield conn
    conn.close()


class TestCreateScope:
    """Test session creation."""

    def test_create_scope_project_level(self, db_conn):
        """Create a project-level scope."""
        session_id = create_scope(
            db_conn,
            agent_id="agent-1",
            level="project",
            projects=["project-a", "project-b"],
        )
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID4 format

        # Verify it's in the database
        row = db_conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert row is not None
        assert row[1] == "agent-1"  # agent_id
        assert row[2] == "project"  # level

    def test_create_scope_collection_level(self, db_conn):
        """Create a collection-level scope."""
        session_id = create_scope(
            db_conn,
            agent_id="agent-2",
            level="collection",
            projects=["proj-1"],
            collections=["collection-x"],
            capabilities=["search", "index"],
        )
        assert isinstance(session_id, str)

        row = db_conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert row[2] == "collection"  # level

    def test_create_scope_global_level(self, db_conn):
        """Create a global-level scope."""
        session_id = create_scope(
            db_conn,
            agent_id="agent-admin",
            level="global",
            projects=[],
            capabilities=["admin"],
        )
        assert isinstance(session_id, str)

        row = db_conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert row[2] == "global"  # level

    def test_create_scope_custom_ttl(self, db_conn):
        """Create a scope with custom TTL."""
        session_id = create_scope(
            db_conn,
            agent_id="agent-3",
            level="project",
            projects=["proj-a"],
            ttl_minutes=60,
        )

        # Verify valid_until is approximately 60 minutes from now
        row = db_conn.execute(
            "SELECT created_at, valid_until FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        created = datetime.fromisoformat(row[0])
        valid_until = datetime.fromisoformat(row[1])
        diff = (valid_until - created).total_seconds() / 60
        assert 59 <= diff <= 61  # Allow 1 minute margin


class TestValidateSession:
    """Test session validation."""

    def test_validate_session_valid(self, db_conn):
        """Validate a valid, non-expired session."""
        session_id = create_scope(
            db_conn,
            agent_id="agent-1",
            level="project",
            projects=["proj-a", "proj-b"],
            capabilities=["search"],
        )

        scope = validate_session(db_conn, session_id)

        assert isinstance(scope, ScopeInfo)
        assert scope.session_id == session_id
        assert scope.agent_id == "agent-1"
        assert scope.level == "project"
        assert scope.projects == ["proj-a", "proj-b"]
        assert scope.capabilities == ["search"]

    def test_validate_session_not_found(self, db_conn):
        """Raise SessionNotFoundError for non-existent session."""
        with pytest.raises(SessionNotFoundError):
            validate_session(db_conn, "nonexistent-session")

    def test_validate_session_expired(self, db_conn):
        """Raise SessionExpiredError for expired session."""
        # Manually insert an expired session
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        session_id = "expired-session"
        db_conn.execute(
            """
            INSERT INTO sessions
            (session_id, agent_id, level, projects_list, valid_until, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                "agent-1",
                "project",
                '["proj-a"]',
                (now - timedelta(minutes=1)).isoformat(),
                now.isoformat(),
            ),
        )
        db_conn.commit()

        with pytest.raises(SessionExpiredError):
            validate_session(db_conn, session_id)


class TestCheckScope:
    """Test scope containment checks."""

    def test_check_scope_project_level_allowed(self, db_conn):
        """Project-level scope allows projects in scope.projects."""
        session_id = create_scope(
            db_conn,
            agent_id="agent-1",
            level="project",
            projects=["proj-a", "proj-b"],
        )
        scope = validate_session(db_conn, session_id)

        assert check_scope(scope, "proj-a") is True
        assert check_scope(scope, "proj-b") is True

    def test_check_scope_project_level_denied(self, db_conn):
        """Project-level scope denies projects outside scope.projects."""
        session_id = create_scope(
            db_conn,
            agent_id="agent-1",
            level="project",
            projects=["proj-a"],
        )
        scope = validate_session(db_conn, session_id)

        assert check_scope(scope, "proj-x") is False

    def test_check_scope_global_level_allowed(self, db_conn):
        """Global-level scope allows any project."""
        session_id = create_scope(
            db_conn,
            agent_id="agent-admin",
            level="global",
            projects=[],
        )
        scope = validate_session(db_conn, session_id)

        assert check_scope(scope, "any-project") is True
        assert check_scope(scope, "another-project") is True

    def test_check_scope_collection_level(self, db_conn):
        """Collection-level scope checks against collections."""
        # For now, collection-level requires a way to map collections to projects
        # This test documents the expected behavior
        session_id = create_scope(
            db_conn,
            agent_id="agent-1",
            level="collection",
            projects=["proj-a", "proj-b"],
            collections=["coll-1"],
        )
        scope = validate_session(db_conn, session_id)

        # Collection-level: deny-by-default unless project in collections' project list
        assert check_scope(scope, "proj-a") is True
        assert check_scope(scope, "proj-unknown") is False


class TestRevokeScope:
    """Test scope revocation."""

    def test_revoke_scope_single_agent(self, db_conn):
        """Revoke all sessions for a single agent."""
        session_1 = create_scope(
            db_conn, agent_id="agent-1", level="project", projects=["proj-a"]
        )
        session_2 = create_scope(
            db_conn, agent_id="agent-1", level="project", projects=["proj-b"]
        )
        session_3 = create_scope(
            db_conn, agent_id="agent-2", level="project", projects=["proj-c"]
        )

        count = revoke_scope(db_conn, "agent-1")
        assert count == 2

        # Verify sessions are gone
        with pytest.raises(SessionNotFoundError):
            validate_session(db_conn, session_1)
        with pytest.raises(SessionNotFoundError):
            validate_session(db_conn, session_2)

        # Verify agent-2 session still exists
        scope = validate_session(db_conn, session_3)
        assert scope.agent_id == "agent-2"

    def test_revoke_scope_no_sessions(self, db_conn):
        """Revoking agent with no sessions returns 0."""
        count = revoke_scope(db_conn, "nonexistent-agent")
        assert count == 0


class TestCleanupExpiredSessions:
    """Test expired session cleanup."""

    def test_cleanup_expired_sessions(self, db_conn):
        """Delete all expired sessions."""
        from datetime import datetime, timedelta

        now = datetime.utcnow()

        # Create valid session
        valid_session = create_scope(
            db_conn,
            agent_id="agent-1",
            level="project",
            projects=["proj-a"],
            ttl_minutes=30,
        )

        # Create expired session manually
        expired_session = "expired-session"
        db_conn.execute(
            """
            INSERT INTO sessions
            (session_id, agent_id, level, projects_list, valid_until, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                expired_session,
                "agent-2",
                "project",
                '["proj-b"]',
                (now - timedelta(minutes=1)).isoformat(),
                now.isoformat(),
            ),
        )
        db_conn.commit()

        count = cleanup_expired_sessions(db_conn)
        assert count == 1

        # Verify valid session still exists
        scope = validate_session(db_conn, valid_session)
        assert scope.agent_id == "agent-1"

        # Verify expired session is gone
        with pytest.raises(SessionNotFoundError):
            validate_session(db_conn, expired_session)


class TestNormalizeAndValidatePath:
    """Test path traversal protection."""

    def test_normalize_simple_path(self):
        """Normalize a simple relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            result = normalize_and_validate_path(project_root, "src/main.py")
            # Check that result is within project_root (accounting for symlink resolution)
            assert Path(result).relative_to(Path(project_root).resolve())
            assert "src/main.py" in result

    def test_normalize_with_subdirs(self):
        """Normalize a path with multiple subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            result = normalize_and_validate_path(
                project_root, "src/lib/utils/helper.py"
            )
            # Check that result is within project_root and has the right structure
            assert Path(result).relative_to(Path(project_root).resolve())
            assert "src/lib/utils/helper.py" in result

    def test_prevent_path_traversal_dotdot(self):
        """Block path traversal with .. (parent directory access)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            with pytest.raises(PathTraversalError):
                normalize_and_validate_path(project_root, "../../etc/passwd")

    def test_prevent_path_traversal_absolute(self):
        """Block absolute paths outside project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            with pytest.raises(PathTraversalError):
                normalize_and_validate_path(project_root, "/etc/passwd")

    def test_prevent_path_traversal_symlink(self, tmp_path):
        """Block symlink escapes."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        symlink = project_root / "link"
        symlink.symlink_to(outside)

        # Accessing through symlink that escapes should be blocked
        with pytest.raises(PathTraversalError):
            normalize_and_validate_path(str(project_root), "link/../../../etc/passwd")

    def test_path_already_absolute(self):
        """Handle paths that are already absolute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            # Absolute path within project_root should be allowed
            file_path = str(Path(project_root) / "src" / "main.py")
            result = normalize_and_validate_path(project_root, file_path)
            # Both should resolve to the same location (accounting for symlink expansion)
            assert Path(result).resolve() == Path(file_path).resolve()

    def test_path_escape_via_traversal(self):
        """Block paths that escape project root after normalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            # Create a nested structure
            nested = Path(project_root) / "a" / "b" / "c"
            nested.mkdir(parents=True)

            # Try to escape from nested location - go up too many levels
            with pytest.raises(PathTraversalError):
                normalize_and_validate_path(project_root, "a/b/c/../../../../etc/passwd")
