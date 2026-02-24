"""Phase 2 integration tests: admin tools, job queue, audit log, session validation."""

import os
import json
import tempfile

import pytest

from tessera.db import GlobalDB, ProjectDB
from tessera.auth import create_scope, validate_session, revoke_scope, SessionNotFoundError
from tessera.server import create_server
import tessera.server as server_mod


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


async def call(server, tool_name, args=None):
    """Helper to call a tool and return the text content."""
    content, _ = await server.call_tool(tool_name, args or {})
    return content


class TestServerWiring:
    """Test that server is wired to real databases."""

    def test_server_uses_real_db(self, wired_server):
        assert server_mod._project_db is not None
        assert server_mod._global_db is not None
        assert isinstance(server_mod._project_db, ProjectDB)
        assert isinstance(server_mod._global_db, GlobalDB)

    async def test_server_lists_10_tools(self, wired_server):
        tools = await wired_server.list_tools()
        tool_names = [t.name for t in tools]
        assert len(tool_names) == 10
        assert "register_project" in tool_names
        assert "status" in tool_names


class TestJobQueue:
    """Test job queue lifecycle."""

    def test_job_lifecycle(self, global_db):
        project_id = global_db.register_project("/tmp/test", "test")
        job_id = global_db.create_job(project_id)
        assert global_db.get_job_status(job_id)["status"] == "pending"

        global_db.start_job(job_id)
        job = global_db.get_job_status(job_id)
        assert job["status"] == "running"
        assert job["started_at"] is not None

        global_db.complete_job(job_id)
        job = global_db.get_job_status(job_id)
        assert job["status"] == "completed"
        assert job["completed_at"] is not None

    def test_job_failure(self, global_db):
        project_id = global_db.register_project("/tmp/test2", "test2")
        job_id = global_db.create_job(project_id)
        global_db.start_job(job_id)
        global_db.fail_job(job_id, "disk full")

        job = global_db.get_job_status(job_id)
        assert job["status"] == "failed"
        assert job["error"] == "disk full"

    def test_crash_recovery(self, global_db):
        project_id = global_db.register_project("/tmp/test3", "test3")
        job_id = global_db.create_job(project_id)
        global_db.start_job(job_id)

        pending = global_db.get_pending_jobs()
        assert any(j["id"] == job_id and j["status"] == "running" for j in pending)

        count = global_db.reset_crashed_jobs()
        assert count == 1
        assert global_db.get_job_status(job_id)["status"] == "pending"

    def test_get_pending_includes_running(self, global_db):
        project_id = global_db.register_project("/tmp/test4", "test4")
        j1 = global_db.create_job(project_id)
        j2 = global_db.create_job(project_id)
        global_db.start_job(j2)

        ids = [j["id"] for j in global_db.get_pending_jobs()]
        assert j1 in ids
        assert j2 in ids


class TestAuditLog:
    """Test audit log persistence."""

    def test_audit_insert_and_query(self, global_db):
        global_db.insert_audit("agent-1", "project", "search", 5)
        global_db.insert_audit("agent-1", "project", "symbols", 3)
        global_db.insert_audit("agent-2", "global", "status", 1)

        assert len(global_db.get_audit_log()) == 3
        assert len(global_db.get_audit_log(agent_id="agent-1")) == 2

    def test_audit_fields_populated(self, global_db):
        global_db.insert_audit("agent-x", "collection", "references", 10)
        log = global_db.get_audit_log()[0]
        assert log["agent_id"] == "agent-x"
        assert log["scope_level"] == "collection"
        assert log["tool_called"] == "references"
        assert log["result_count"] == 10
        assert log["timestamp"] is not None

    async def test_audit_persisted_via_server(self, wired_server):
        await wired_server.call_tool("search", {"query": "test"})
        logs = server_mod._global_db.get_audit_log()
        assert len(logs) >= 1
        assert logs[0]["tool_called"] == "search"


class TestSessionValidation:
    """Test session validation and scope gating."""

    async def test_valid_session_passes(self, wired_server):
        session_id = create_scope(
            server_mod._global_db.conn, "agent-1", "project",
            projects=["1"], ttl_minutes=30
        )
        content = await call(wired_server, "search", {"query": "test", "session_id": session_id})
        text = content[0].text
        assert "Invalid session" not in text
        assert "Session expired" not in text

    async def test_invalid_session_rejected(self, wired_server):
        content = await call(wired_server, "search", {"query": "test", "session_id": "bogus"})
        assert "Invalid session" in content[0].text

    async def test_expired_session_rejected(self, wired_server):
        session_id = create_scope(
            server_mod._global_db.conn, "agent-exp", "project",
            projects=["1"], ttl_minutes=0
        )
        content = await call(wired_server, "search", {"query": "test", "session_id": session_id})
        assert "Session expired" in content[0].text

    async def test_no_session_dev_mode(self, wired_server):
        content = await call(wired_server, "search", {"query": "test"})
        assert "Invalid session" not in content[0].text

    async def test_revoked_session_rejected(self, wired_server):
        session_id = create_scope(
            server_mod._global_db.conn, "agent-rev", "project",
            projects=["1"], ttl_minutes=30
        )
        revoke_scope(server_mod._global_db.conn, "agent-rev")
        content = await call(wired_server, "search", {"query": "test", "session_id": session_id})
        assert "Invalid session" in content[0].text


class TestScopeGating:
    """Test scope level enforcement."""

    async def test_project_scope_cannot_admin(self, wired_server):
        session_id = create_scope(
            server_mod._global_db.conn, "task-agent", "project",
            projects=["1"], ttl_minutes=30
        )
        content = await call(wired_server, "register_project", {
            "path": "/tmp/new", "name": "new", "session_id": session_id
        })
        assert "Insufficient scope" in content[0].text

    async def test_global_scope_can_admin(self, wired_server):
        session_id = create_scope(
            server_mod._global_db.conn, "persistence-agent", "global",
            projects=[], ttl_minutes=30
        )
        content = await call(wired_server, "status", {"session_id": session_id})
        text = content[0].text
        assert "Insufficient scope" not in text
        data = json.loads(text)
        assert "project_count" in data

    async def test_dev_mode_allows_admin(self, wired_server):
        content = await call(wired_server, "status", {})
        data = json.loads(content[0].text)
        assert "project_count" in data


class TestAdminTools:
    """Test admin MCP tools end-to-end."""

    async def test_register_project_tool(self, wired_server):
        content = await call(wired_server, "register_project", {
            "path": "/tmp/myproject", "name": "myproject", "language": "python"
        })
        data = json.loads(content[0].text)
        assert data["name"] == "myproject"
        assert data["path"] == "/tmp/myproject"
        assert "id" in data

    async def test_status_tool(self, wired_server):
        await call(wired_server, "register_project", {"path": "/tmp/p1", "name": "p1"})
        content = await call(wired_server, "status", {})
        data = json.loads(content[0].text)
        assert data["project_count"] >= 1

    async def test_create_and_revoke_scope_tools(self, wired_server):
        content = await call(wired_server, "create_scope_tool", {
            "agent_id": "test-agent", "scope_level": "project", "project_ids": [1]
        })
        data = json.loads(content[0].text)
        assert "session_id" in data
        session_id = data["session_id"]

        scope = validate_session(server_mod._global_db.conn, session_id)
        assert scope.agent_id == "test-agent"

        content = await call(wired_server, "revoke_scope_tool", {"agent_id": "test-agent"})
        data = json.loads(content[0].text)
        assert data["sessions_revoked"] >= 1

        with pytest.raises(SessionNotFoundError):
            validate_session(server_mod._global_db.conn, session_id)


class TestDeleteFileData:
    """Test cascade deletion for incremental reindex."""

    def test_delete_file_removes_all_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "project.db")
            pdb = ProjectDB(db_path)

            file_id = pdb.upsert_file(1, "test.py", "python", "abc123")
            pdb.update_file_status(file_id, "indexed")

            pdb.insert_symbols([
                {"project_id": 1, "file_id": file_id, "name": "foo", "kind": "function", "line": 1, "col": 0, "scope": None, "signature": None}
            ])

            assert len(pdb.lookup_symbols("foo")) > 0

            pdb.delete_file_data("test.py")

            assert len(pdb.lookup_symbols("foo")) == 0
            assert pdb.get_file("test.py") is None

            pdb.close()
