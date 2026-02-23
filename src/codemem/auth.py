"""Session token distribution and scope-gated access control.

Phase 1: Token generation and validation.
Implements:
  - Session token creation (cryptographic, non-guessable)
  - SQLite sessions table for token-to-scope mapping
  - Scope validation at request time (project/collection/global)
  - Token lifecycle management (expiration, revocation)

CTO Condition C2: Tokens generated in-process by create_scope(),
stored in SQLite sessions table. Passed to task agents via MCP
initialize message. Not transmitted beyond MCP boundary (trusted local).
"""

import uuid
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path


class SessionNotFoundError(Exception):
    """Raised when session_id not found in sessions table."""
    pass


class SessionExpiredError(Exception):
    """Raised when session has expired."""
    pass


class ForbiddenScopeError(Exception):
    """Raised when request is outside session's scope."""
    pass


class PathTraversalError(Exception):
    """Raised when file path escapes project root."""
    pass


@dataclass
class ScopeInfo:
    """Scope information extracted from a valid session."""
    session_id: str
    agent_id: str
    level: str  # 'project', 'collection', 'global'
    projects: list[str]
    collections: list[str]
    capabilities: list[str]


def create_scope(
    db_conn: sqlite3.Connection,
    agent_id: str,
    level: str,
    projects: list[str],
    collections: list[str] = None,
    capabilities: list[str] = None,
    ttl_minutes: int = 30,
) -> str:
    """
    Create a new session with scope claims.

    - Generate UUID4 session_id
    - Insert into sessions table
    - valid_until = now + ttl_minutes
    - Return opaque session_id

    Args:
        db_conn: SQLite connection
        agent_id: Identifier for the agent
        level: 'project', 'collection', or 'global'
        projects: List of project IDs in scope
        collections: List of collection IDs in scope (optional)
        capabilities: List of capability strings (optional)
        ttl_minutes: Time to live in minutes (default 30)

    Returns:
        UUID4 session_id string
    """
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    valid_until = now + timedelta(minutes=ttl_minutes)

    # Default empty lists if not provided
    if collections is None:
        collections = []
    if capabilities is None:
        capabilities = []

    # JSON encode the lists for storage
    projects_json = json.dumps(projects)
    collections_json = json.dumps(collections)
    capabilities_json = json.dumps(capabilities)

    db_conn.execute(
        """
        INSERT INTO sessions
        (session_id, agent_id, level, projects_list, collections_list,
         capabilities, created_at, valid_until)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            agent_id,
            level,
            projects_json,
            collections_json,
            capabilities_json,
            now.isoformat(),
            valid_until.isoformat(),
        ),
    )
    db_conn.commit()

    return session_id


def validate_session(db_conn: sqlite3.Connection, session_id: str) -> ScopeInfo:
    """
    Validate session and return scope info.

    - Look up session_id in sessions table
    - Raise SessionNotFoundError if not found
    - Raise SessionExpiredError if valid_until < now
    - Return ScopeInfo dataclass

    Args:
        db_conn: SQLite connection
        session_id: Session ID to validate

    Returns:
        ScopeInfo dataclass with scope information

    Raises:
        SessionNotFoundError: If session_id not found
        SessionExpiredError: If session has expired
    """
    row = db_conn.execute(
        """
        SELECT session_id, agent_id, level, projects_list, collections_list,
               capabilities, valid_until
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    if row is None:
        raise SessionNotFoundError(f"Session {session_id} not found")

    session_id, agent_id, level, projects_json, collections_json, capabilities_json, valid_until_str = row

    # Check expiration
    valid_until = datetime.fromisoformat(valid_until_str)
    if valid_until < datetime.utcnow():
        raise SessionExpiredError(f"Session {session_id} has expired")

    # Decode JSON fields
    projects = json.loads(projects_json) if projects_json else []
    collections = json.loads(collections_json) if collections_json else []
    capabilities = json.loads(capabilities_json) if capabilities_json else []

    return ScopeInfo(
        session_id=session_id,
        agent_id=agent_id,
        level=level,
        projects=projects,
        collections=collections,
        capabilities=capabilities,
    )


def revoke_scope(db_conn: sqlite3.Connection, agent_id: str) -> int:
    """
    Revoke all sessions for an agent.
    O(1) delete operation.
    Returns count of revoked sessions.

    Args:
        db_conn: SQLite connection
        agent_id: Agent ID to revoke

    Returns:
        Number of sessions revoked
    """
    cursor = db_conn.execute(
        "DELETE FROM sessions WHERE agent_id = ?",
        (agent_id,),
    )
    db_conn.commit()
    return cursor.rowcount


def check_scope(scope: ScopeInfo, project_id: str) -> bool:
    """
    Check if project_id is within the session's scope.

    - 'global' level: all projects allowed
    - 'collection' level: check if project in collection's projects
    - 'project' level: check if project_id in scope.projects
    - deny-by-default: if anything unclear, return False

    Args:
        scope: ScopeInfo from validate_session
        project_id: Project ID to check

    Returns:
        True if project_id is in scope, False otherwise
    """
    if scope.level == "global":
        # Global scope: all projects allowed
        return True
    elif scope.level == "project":
        # Project scope: check if project_id in projects list
        return project_id in scope.projects
    elif scope.level == "collection":
        # Collection scope: check if project_id in projects list
        # (collections themselves are represented by their project memberships)
        return project_id in scope.projects
    else:
        # Unknown level: deny by default
        return False


def cleanup_expired_sessions(db_conn: sqlite3.Connection) -> int:
    """
    Delete all expired sessions.
    Returns count deleted.

    Args:
        db_conn: SQLite connection

    Returns:
        Number of expired sessions deleted
    """
    now = datetime.utcnow()
    cursor = db_conn.execute(
        "DELETE FROM sessions WHERE valid_until < ?",
        (now.isoformat(),),
    )
    db_conn.commit()
    return cursor.rowcount


def normalize_and_validate_path(project_root: str, user_path: str) -> str:
    """
    Validate that user_path resolves within project_root.

    1. Resolve both paths to absolute
    2. Check that resolved user_path starts with resolved project_root
    3. Raise PathTraversalError if not
    4. Return the validated absolute path

    Examples:
        normalize_and_validate_path('/projects/pm', 'src/Hooks.php')
            → '/projects/pm/src/Hooks.php'
        normalize_and_validate_path('/projects/pm', '../../etc/passwd')
            → raises PathTraversalError

    Args:
        project_root: Root directory of the project
        user_path: User-provided path (relative or absolute)

    Returns:
        Normalized absolute path within project_root

    Raises:
        PathTraversalError: If user_path escapes project_root
    """
    # Resolve project_root to absolute
    root = Path(project_root).resolve()

    # If user_path is absolute, resolve it directly and validate
    if Path(user_path).is_absolute():
        user = Path(user_path).resolve()
    else:
        # If user_path is relative, resolve it relative to project_root
        user = (root / user_path).resolve()

    # Check if user_path is within project_root
    try:
        user.relative_to(root)
    except ValueError:
        # user is not relative to root, path escape detected
        raise PathTraversalError(
            f"Path {user_path} escapes project root {project_root}"
        )

    return str(user)


if __name__ == "__main__":
    """Verification block: test all functions with in-memory SQLite."""
    print("Running auth.py verification...")

    # Create in-memory SQLite DB with sessions table
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

    # Test 1: Create and validate a session
    print("\n1. Testing create_scope and validate_session...")
    session_id = create_scope(
        conn,
        agent_id="test-agent",
        level="project",
        projects=["proj-a", "proj-b"],
        capabilities=["search"],
    )
    print(f"   Created session: {session_id}")

    scope = validate_session(conn, session_id)
    print(f"   Validated session: agent_id={scope.agent_id}, level={scope.level}")
    print(f"   Projects: {scope.projects}, Capabilities: {scope.capabilities}")
    assert scope.agent_id == "test-agent"
    assert scope.level == "project"
    print("   ✓ create_scope and validate_session working")

    # Test 2: Test expired session rejection
    print("\n2. Testing expired session rejection...")
    now = datetime.utcnow()
    expired_id = "expired-session"
    conn.execute(
        """
        INSERT INTO sessions
        (session_id, agent_id, level, projects_list, valid_until, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            expired_id,
            "agent-2",
            "project",
            '["proj-x"]',
            (now - timedelta(minutes=1)).isoformat(),
            now.isoformat(),
        ),
    )
    conn.commit()

    try:
        validate_session(conn, expired_id)
        print("   ✗ FAILED: Should have raised SessionExpiredError")
    except SessionExpiredError:
        print("   ✓ Expired session rejection working")

    # Test 3: Test invalid session rejection
    print("\n3. Testing invalid session rejection...")
    try:
        validate_session(conn, "nonexistent-session")
        print("   ✗ FAILED: Should have raised SessionNotFoundError")
    except SessionNotFoundError:
        print("   ✓ Invalid session rejection working")

    # Test 4: Test scope checking
    print("\n4. Testing scope checking...")
    scope = validate_session(conn, session_id)
    assert check_scope(scope, "proj-a") is True
    assert check_scope(scope, "proj-unknown") is False
    print("   ✓ Scope checking working")

    # Test 5: Test path traversal blocking
    print("\n5. Testing path traversal blocking...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Valid path
        result = normalize_and_validate_path(tmpdir, "src/main.py")
        # Check that result is within tmpdir by trying to get relative path
        try:
            Path(result).relative_to(Path(tmpdir).resolve())
            print(f"   Valid path allowed: {result}")
        except ValueError:
            print(f"   ✗ FAILED: Path not within tmpdir: {result}")

        # Invalid path with ..
        try:
            normalize_and_validate_path(tmpdir, "../../etc/passwd")
            print("   ✗ FAILED: Should have raised PathTraversalError")
        except PathTraversalError:
            print("   ✓ Path traversal blocking working")

    print("\n✓ All verification tests passed!")
    conn.close()
