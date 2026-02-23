"""Phase 2 integration tests: admin tools, job queue, audit log, session validation."""

import os
import json
import tempfile
import sqlite3
import asyncio
from pathlib import Path

import pytest

from tessera.db import GlobalDB, ProjectDB
from tessera.auth import create_scope, validate_session, revoke_scope, SessionNotFoundError, SessionExpiredError
from tessera.indexer import IndexerPipeline
from tessera.server import (
    create_server, search_tool, register_project_tool, reindex_tool,
    create_scope_tool, revoke_scope_tool, status_tool, _check_session
)
import tessera.server as server_mod


def run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for project and global DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = os.path.join(tmpdir, "project")
        os.makedirs(project_dir)
        global_db_path = os.path.join(tmpdir, "global.db")
        yield project_dir, global_db_path


@pytest.fixture
def global_db(temp_dirs):
    """Create a real GlobalDB instance."""
    _, global_db_path = temp_dirs
    db = GlobalDB(global_db_path)
    yield db
    db.close()


@pytest.fixture
def wired_server(temp_dirs):
    """Create a server wired to real databases."""
    project_dir, global_db_path = temp_dirs
    server = create_server(project_dir, global_db_path)
    yield server


class TestServerWiring:
    """Test that server is wired to real databases."""

    def test_server_uses_real_db(self, wired_server):
        """Server creates real ProjectDB and GlobalDB, not stubs."""
        assert server_mod._project_db is not None
        assert server_mod._global_db is not None
        assert isinstance(server_mod._project_db, ProjectDB)
        assert isinstance(server_mod._global_db, GlobalDB)

    def test_server_lists_10_tools(self, wired_server):
        """Server registers all 10 tools (5 core + 5 admin)."""
        import mcp.types as types
        result = run_async(wired_server.request_handlers[types.ListToolsRequest](None))
        tool_names = [t.name for t in result.root.tools]
        assert len(tool_names) == 10
        assert "register_project" in tool_names
        assert "status" in tool_names


class TestJobQueue:
    """Test job queue lifecycle."""

    def test_job_lifecycle(self, global_db):
        """Job progresses: pending → running → completed."""
        project_id = global_db.register_project("/tmp/test", "test")
        job_id = global_db.create_job(project_id)

        job = global_db.get_job_status(job_id)
        assert job["status"] == "pending"

        global_db.start_job(job_id)
        job = global_db.get_job_status(job_id)
        assert job["status"] == "running"
        assert job["started_at"] is not None

        global_db.complete_job(job_id)
        job = global_db.get_job_status(job_id)
        assert job["status"] == "completed"
        assert job["completed_at"] is not None

    def test_job_failure(self, global_db):
        """Failed jobs record error message."""
        project_id = global_db.register_project("/tmp/test2", "test2")
        job_id = global_db.create_job(project_id)
        global_db.start_job(job_id)
        global_db.fail_job(job_id, "disk full")

        job = global_db.get_job_status(job_id)
        assert job["status"] == "failed"
        assert job["error"] == "disk full"

    def test_crash_recovery(self, global_db):
        """Running jobs at startup detected and reset to pending."""
        project_id = global_db.register_project("/tmp/test3", "test3")
        job_id = global_db.create_job(project_id)
        global_db.start_job(job_id)

        # Simulate crash — job left as 'running'
        pending = global_db.get_pending_jobs()
        assert any(j["id"] == job_id and j["status"] == "running" for j in pending)

        # Reset crashed jobs
        count = global_db.reset_crashed_jobs()
        assert count == 1

        job = global_db.get_job_status(job_id)
        assert job["status"] == "pending"

    def test_get_pending_includes_running(self, global_db):
        """get_pending_jobs returns both pending and running."""
        project_id = global_db.register_project("/tmp/test4", "test4")
        j1 = global_db.create_job(project_id)
        j2 = global_db.create_job(project_id)
        global_db.start_job(j2)

        pending = global_db.get_pending_jobs()
        ids = [j["id"] for j in pending]
        assert j1 in ids
        assert j2 in ids


class TestAuditLog:
    """Test audit log persistence."""

    def test_audit_insert_and_query(self, global_db):
        """Audit records are persisted and queryable."""
        global_db.insert_audit("agent-1", "project", "search", 5)
        global_db.insert_audit("agent-1", "project", "symbols", 3)
        global_db.insert_audit("agent-2", "global", "status", 1)

        all_logs = global_db.get_audit_log()
        assert len(all_logs) == 3

        agent1_logs = global_db.get_audit_log(agent_id="agent-1")
        assert len(agent1_logs) == 2

    def test_audit_fields_populated(self, global_db):
        """All audit log fields have expected values."""
        global_db.insert_audit("agent-x", "collection", "references", 10)
        logs = global_db.get_audit_log()
        log = logs[0]
        assert log["agent_id"] == "agent-x"
        assert log["scope_level"] == "collection"
        assert log["tool_called"] == "references"
        assert log["result_count"] == 10
        assert log["timestamp"] is not None

    def test_audit_persisted_via_server(self, wired_server):
        """Tool calls create audit records in the database."""
        run_async(search_tool("search", {"query": "test"}))
        logs = server_mod._global_db.get_audit_log()
        assert len(logs) >= 1
        assert logs[0]["tool_called"] == "search"


class TestSessionValidation:
    """Test session validation and scope gating."""

    def test_valid_session_passes(self, wired_server):
        """Valid session allows tool execution."""
        session_id = create_scope(
            server_mod._global_db.conn, "agent-1", "project",
            projects=["1"], ttl_minutes=30
        )
        result = run_async(search_tool("search", {"query": "test", "session_id": session_id}))
        assert isinstance(result, list)
        # Should not be an error about session
        text = result[0].text
        assert "Invalid session" not in text
        assert "Session expired" not in text

    def test_invalid_session_rejected(self, wired_server):
        """Invalid session returns error."""
        result = run_async(search_tool("search", {"query": "test", "session_id": "bogus-session"}))
        assert "Invalid session" in result[0].text

    def test_expired_session_rejected(self, wired_server):
        """Expired session returns error."""
        session_id = create_scope(
            server_mod._global_db.conn, "agent-exp", "project",
            projects=["1"], ttl_minutes=0  # immediately expired
        )
        result = run_async(search_tool("search", {"query": "test", "session_id": session_id}))
        assert "Session expired" in result[0].text

    def test_no_session_dev_mode(self, wired_server):
        """Missing session_id falls back to dev mode (allowed)."""
        result = run_async(search_tool("search", {"query": "test"}))
        assert isinstance(result, list)
        assert "Invalid session" not in result[0].text

    def test_revoked_session_rejected(self, wired_server):
        """Revoked session returns error."""
        session_id = create_scope(
            server_mod._global_db.conn, "agent-rev", "project",
            projects=["1"], ttl_minutes=30
        )
        revoke_scope(server_mod._global_db.conn, "agent-rev")
        result = run_async(search_tool("search", {"query": "test", "session_id": session_id}))
        assert "Invalid session" in result[0].text


class TestScopeGating:
    """Test scope level enforcement."""

    def test_project_scope_cannot_admin(self, wired_server):
        """Project-scope session cannot call admin tools."""
        session_id = create_scope(
            server_mod._global_db.conn, "task-agent", "project",
            projects=["1"], ttl_minutes=30
        )
        result = run_async(register_project_tool(
            "register_project",
            {"path": "/tmp/new", "name": "new", "session_id": session_id}
        ))
        assert "Insufficient scope" in result[0].text

    def test_global_scope_can_admin(self, wired_server):
        """Global-scope session can call admin tools."""
        session_id = create_scope(
            server_mod._global_db.conn, "persistence-agent", "global",
            projects=[], ttl_minutes=30
        )
        result = run_async(status_tool("status", {"session_id": session_id}))
        text = result[0].text
        assert "Insufficient scope" not in text
        data = json.loads(text)
        assert "project_count" in data

    def test_dev_mode_allows_admin(self, wired_server):
        """Dev mode (no session_id) allows admin tools."""
        result = run_async(status_tool("status", {}))
        data = json.loads(result[0].text)
        assert "project_count" in data


class TestAdminTools:
    """Test admin MCP tools end-to-end."""

    def test_register_project_tool(self, wired_server):
        """register_project creates a project entry."""
        result = run_async(register_project_tool(
            "register_project",
            {"path": "/tmp/myproject", "name": "myproject", "language": "python"}
        ))
        data = json.loads(result[0].text)
        assert data["name"] == "myproject"
        assert data["path"] == "/tmp/myproject"
        assert "id" in data

    def test_status_tool(self, wired_server):
        """status tool returns system overview."""
        # Register a project first
        run_async(register_project_tool(
            "register_project",
            {"path": "/tmp/proj1", "name": "proj1"}
        ))
        result = run_async(status_tool("status", {}))
        data = json.loads(result[0].text)
        assert data["project_count"] >= 1

    def test_create_and_revoke_scope_tools(self, wired_server):
        """create_scope and revoke_scope tools work end-to-end."""
        # Create scope
        result = run_async(create_scope_tool(
            "create_scope",
            {"agent_id": "test-agent", "scope_level": "project", "project_ids": [1]}
        ))
        data = json.loads(result[0].text)
        assert "session_id" in data
        session_id = data["session_id"]

        # Validate it works
        scope = validate_session(server_mod._global_db.conn, session_id)
        assert scope.agent_id == "test-agent"

        # Revoke
        result = run_async(revoke_scope_tool(
            "revoke_scope",
            {"agent_id": "test-agent"}
        ))
        data = json.loads(result[0].text)
        assert data["sessions_revoked"] >= 1

        # Verify revoked
        with pytest.raises(SessionNotFoundError):
            validate_session(server_mod._global_db.conn, session_id)


class TestDeleteFileData:
    """Test cascade deletion for incremental reindex."""

    def test_delete_file_removes_all_data(self):
        """delete_file_data removes file, symbols, refs, chunks, embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "project.db")
            pdb = ProjectDB(db_path)

            # Insert a file
            file_id = pdb.upsert_file(1, "test.py", "python", "abc123")
            pdb.update_file_status(file_id, "indexed")

            # Insert symbols
            pdb.insert_symbols([
                {"project_id": 1, "file_id": file_id, "name": "foo", "kind": "function", "line": 1, "col": 0, "scope": None, "signature": None}
            ])

            # Verify data exists
            symbols = pdb.lookup_symbols("foo")
            assert len(symbols) > 0

            # Delete file data
            pdb.delete_file_data("test.py")

            # Verify data removed
            symbols = pdb.lookup_symbols("foo")
            assert len(symbols) == 0

            file_info = pdb.get_file("test.py")
            assert file_info is None

            pdb.close()
