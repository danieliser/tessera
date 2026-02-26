"""Tests for Phase 3 Federation: cross_refs, collection_map, and collection admin tools."""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import sqlite3

from tessera.server import create_server, _check_session, _get_project_dbs
from tessera.db import GlobalDB, ProjectDB
from tessera.auth import create_scope, validate_session


async def get_json(server, tool_name, args=None):
    """Call a tool and parse the JSON result."""
    content, _ = await server.call_tool(tool_name, args or {})
    return json.loads(content[0].text)


class TestCrossRefs:
    """Tests for cross_refs MCP tool."""

    def test_cross_refs_single_project_no_matches(self, tmp_path):
        """No cross-refs if symbol defined and referenced in same project."""
        global_db_path = tmp_path / "global.db"
        global_db = GlobalDB(str(global_db_path))

        # Register one project
        project_path = tmp_path / "project_a"
        project_path.mkdir()
        pid_a = global_db.register_project(str(project_path), "project-a")

        # Create project DB with symbol definition
        db_a = ProjectDB(str(project_path))
        file_id_a = db_a.upsert_file(pid_a, "hooks.php", "php", "hash1")
        sym_a = db_a.insert_symbols([{
            "name": "hook_name",
            "kind": "function",
            "file_id": file_id_a,
            "line": 10,
            "col": 0,
            "scope": "global",
            "project_id": pid_a,
        }])

        # Reference to same symbol in same project
        db_a.insert_refs([{
            "project_id": pid_a,
            "from_symbol_id": sym_a[0],
            "to_symbol_id": sym_a[0],
            "to_symbol_name": "hook_name",
            "kind": "calls",
            "context": "",
            "line": 42,
        }])

        mcp = create_server(None, str(global_db_path))

        # Call the tool
        result_dict = asyncio.run(get_json(mcp, "cross_refs", {"symbol_name": "hook_name", "session_id": ""}))

        # Should have no cross-refs (same project)
        assert result_dict["symbol"] == "hook_name"
        assert len(result_dict["cross_refs"]) == 0

    def test_cross_refs_two_projects_with_matches(self, tmp_path):
        """Cross-refs when symbol defined in A, referenced in B."""
        global_db_path = tmp_path / "global.db"
        global_db = GlobalDB(str(global_db_path))

        # Register two projects
        project_a_path = tmp_path / "project_a"
        project_a_path.mkdir()
        pid_a = global_db.register_project(str(project_a_path), "plugin-a")

        project_b_path = tmp_path / "project_b"
        project_b_path.mkdir()
        pid_b = global_db.register_project(str(project_b_path), "plugin-b")

        # Project A: define hook_name
        db_a = ProjectDB(str(project_a_path))
        file_a = db_a.upsert_file(pid_a, "hooks.php", "php", "hash1")
        sym_a = db_a.insert_symbols([{
            "name": "hook_name",
            "kind": "function",
            "file_id": file_a,
            "line": 10,
            "col": 0,
            "scope": "global",
            "project_id": pid_a,
        }])

        # Project B: reference hook_name (to_symbol_id unresolved)
        db_b = ProjectDB(str(project_b_path))
        file_b = db_b.upsert_file(pid_b, "main.php", "php", "hash2")
        sym_b = db_b.insert_symbols([{
            "name": "apply_hook",
            "kind": "function",
            "file_id": file_b,
            "line": 5,
            "col": 0,
            "scope": "global",
            "project_id": pid_b,
        }])
        db_b.insert_refs([{
            "project_id": pid_b,
            "from_symbol_id": sym_b[0],
            "to_symbol_id": None,  # Cross-project ref, unresolved
            "to_symbol_name": "hook_name",
            "kind": "calls",
            "context": "",
            "line": 42,
        }])

        mcp = create_server(None, str(global_db_path))

        result_dict = asyncio.run(get_json(mcp, "cross_refs", {"symbol_name": "hook_name", "session_id": ""}))

        assert result_dict["symbol"] == "hook_name"
        assert pid_a in [p["id"] for p in result_dict["definition_projects"].values()]
        assert len(result_dict["cross_refs"]) == 1
        assert result_dict["cross_refs"][0]["from_project_id"] == pid_b
        assert result_dict["cross_refs"][0]["to_project_id"] == pid_a


class TestCollectionMap:
    """Tests for collection_map MCP tool."""

    def test_collection_map_single_project(self, tmp_path):
        """collection_map with single project shows no edges."""
        global_db_path = tmp_path / "global.db"
        global_db = GlobalDB(str(global_db_path))

        project_path = tmp_path / "project_a"
        project_path.mkdir()
        pid_a = global_db.register_project(str(project_path), "plugin-a")

        # Create simple project DB
        db_a = ProjectDB(str(project_path))
        file_a = db_a.upsert_file(pid_a, "hooks.php", "php", "hash1")
        db_a.insert_symbols([{
            "name": "hook_a",
            "kind": "function",
            "file_id": file_a,
            "line": 10,
            "col": 0,
            "scope": "global",
            "project_id": pid_a,
        }])

        mcp = create_server(None, str(global_db_path))

        result_dict = asyncio.run(get_json(mcp, "collection_map", {"collection_id": 0, "session_id": ""}))

        assert "projects" in result_dict
        assert "plugin-a" in result_dict["projects"]
        assert result_dict["projects"]["plugin-a"]["id"] == pid_a
        assert "edges" in result_dict
        assert len(result_dict["edges"]) == 0  # No cross-project edges

    def test_collection_map_two_projects_with_edge(self, tmp_path):
        """collection_map shows edges between projects with cross-refs."""
        global_db_path = tmp_path / "global.db"
        global_db = GlobalDB(str(global_db_path))

        # Create and register two projects
        project_a_path = tmp_path / "project_a"
        project_a_path.mkdir()
        pid_a = global_db.register_project(str(project_a_path), "plugin-a")

        project_b_path = tmp_path / "project_b"
        project_b_path.mkdir()
        pid_b = global_db.register_project(str(project_b_path), "plugin-b")

        # Project A: define hook_name
        db_a = ProjectDB(str(project_a_path))
        file_a = db_a.upsert_file(pid_a, "hooks.php", "php", "hash1")
        sym_a = db_a.insert_symbols([{
            "name": "do_action",
            "kind": "function",
            "file_id": file_a,
            "line": 10,
            "col": 0,
            "scope": "global",
            "project_id": pid_a,
        }])

        # Project B: reference do_action
        db_b = ProjectDB(str(project_b_path))
        file_b = db_b.upsert_file(pid_b, "main.php", "php", "hash2")
        sym_b = db_b.insert_symbols([{
            "name": "apply_hook",
            "kind": "function",
            "file_id": file_b,
            "line": 5,
            "col": 0,
            "scope": "global",
            "project_id": pid_b,
        }])
        db_b.insert_refs([{
            "project_id": pid_b,
            "from_symbol_id": sym_b[0],
            "to_symbol_id": None,
            "to_symbol_name": "do_action",
            "kind": "calls",
            "context": "",
            "line": 42,
        }])

        mcp = create_server(None, str(global_db_path))

        result_dict = asyncio.run(get_json(mcp, "collection_map", {"collection_id": 0, "session_id": ""}))

        assert "plugin-a" in result_dict["projects"]
        assert "plugin-b" in result_dict["projects"]

        # Should have at least one edge from B to A
        edges = result_dict["edges"]
        assert len(edges) > 0
        found_edge = False
        for edge in edges:
            if edge["from"] == "plugin-b" and edge["to"] == "plugin-a":
                found_edge = True
                assert edge["cross_refs"] >= 1
                assert "do_action" in edge.get("symbols", [])
        assert found_edge, "Expected edge from plugin-b to plugin-a"


class TestCollectionScopeEnforcement:
    """Tests for collection-level scope enforcement in _get_project_dbs()."""

    def test_collection_scope_restricts_projects(self, tmp_path):
        """Collection-scoped session resolves to correct project IDs."""
        global_db_path = tmp_path / "global.db"

        # Setup projects and collection
        global_db = GlobalDB(str(global_db_path))
        project_a_path = tmp_path / "project_a"
        project_a_path.mkdir()
        pid_a = global_db.register_project(str(project_a_path), "plugin-a")

        project_b_path = tmp_path / "project_b"
        project_b_path.mkdir()
        pid_b = global_db.register_project(str(project_b_path), "plugin-b")

        # Create collection with only plugin-a
        coll_id = global_db.create_collection("test-collection", [pid_a])

        # Create session with collection scope (use collection ID, not name)
        session_id = create_scope(
            global_db.conn,
            agent_id="test_agent",
            level="collection",
            projects=[],
            collections=[str(coll_id)],
            ttl_minutes=30
        )

        # Validate session
        scope = validate_session(global_db.conn, session_id)
        assert scope.level == "collection"
        assert scope.collections == [str(coll_id)]

        # Verify _get_project_dbs logic resolves collection correctly
        # (We can't test the full _get_project_dbs without indexed projects,
        # but we can verify the collection resolution logic by checking
        # that get_collection_projects returns the right projects)
        collection_projects = global_db.get_collection_projects(coll_id)
        collection_pids = [p["id"] for p in collection_projects]

        assert pid_a in collection_pids, f"Project A ({pid_a}) should be in collection"
        assert pid_b not in collection_pids, f"Project B ({pid_b}) should NOT be in collection"

        global_db.close()


class TestCollectionManagementTools:
    """Tests for collection admin tools."""

    async def test_create_collection_tool(self, tmp_path):
        """create_collection_tool creates a new collection."""
        global_db_path = tmp_path / "global.db"
        mcp = create_server(None, str(global_db_path))

        # Get global DB and create admin session
        global_db = GlobalDB(str(global_db_path))
        admin_session_id = create_scope(
            global_db.conn,
            agent_id="admin",
            level="global",
            projects=[],
            collections=[],
            ttl_minutes=30
        )

        result_dict = await get_json(mcp, "create_collection_tool", {
            "name": "test-collection",
            "project_ids": [],
            "session_id": admin_session_id
        })

        assert result_dict["name"] == "test-collection"
        assert "id" in result_dict

    async def test_add_to_collection_tool(self, tmp_path):
        """add_to_collection_tool adds a project to a collection."""
        global_db_path = tmp_path / "global.db"
        global_db = GlobalDB(str(global_db_path))

        # Setup: create project and collection
        project_path = tmp_path / "project"
        project_path.mkdir()
        pid = global_db.register_project(str(project_path), "test-project")
        coll_id = global_db.create_collection("test-coll")

        mcp = create_server(None, str(global_db_path))

        admin_session_id = create_scope(
            global_db.conn,
            agent_id="admin",
            level="global",
            projects=[],
            collections=[],
            ttl_minutes=30
        )

        result_dict = await get_json(mcp, "add_to_collection_tool", {
            "collection_id": coll_id,
            "project_id": pid,
            "session_id": admin_session_id
        })

        assert "success" in result_dict or "id" in result_dict

    async def test_list_collections_tool(self, tmp_path):
        """list_collections_tool lists all collections."""
        global_db_path = tmp_path / "global.db"
        global_db = GlobalDB(str(global_db_path))

        # Create a collection
        global_db.create_collection("coll-1")

        mcp = create_server(None, str(global_db_path))

        admin_session_id = create_scope(
            global_db.conn,
            agent_id="admin",
            level="global",
            projects=[],
            collections=[],
            ttl_minutes=30
        )

        result_dict = await get_json(mcp, "list_collections_tool", {
            "session_id": admin_session_id
        })

        assert "collections" in result_dict
        names = [c["name"] for c in result_dict["collections"]]
        assert "coll-1" in names

    async def test_delete_collection_tool(self, tmp_path):
        """delete_collection_tool deletes a collection."""
        global_db_path = tmp_path / "global.db"
        global_db = GlobalDB(str(global_db_path))

        coll_id = global_db.create_collection("to-delete")

        mcp = create_server(None, str(global_db_path))

        admin_session_id = create_scope(
            global_db.conn,
            agent_id="admin",
            level="global",
            projects=[],
            collections=[],
            ttl_minutes=30
        )

        result_dict = await get_json(mcp, "delete_collection_tool", {
            "collection_id": coll_id,
            "session_id": admin_session_id
        })

        assert "success" in result_dict or result_dict.get("deleted")

    async def test_collection_tool_requires_global_scope(self, tmp_path):
        """Collection admin tools require global scope."""
        global_db_path = tmp_path / "global.db"
        global_db = GlobalDB(str(global_db_path))

        # Create project-scoped session
        project_path = tmp_path / "project"
        project_path.mkdir()
        pid = global_db.register_project(str(project_path), "test-project")

        project_session_id = create_scope(
            global_db.conn,
            agent_id="user",
            level="project",
            projects=[str(pid)],
            collections=[],
            ttl_minutes=30
        )

        mcp = create_server(None, str(global_db_path))

        # Should fail with insufficient scope
        content, _ = await mcp.call_tool("create_collection_tool", {
            "name": "test",
            "project_ids": [],
            "session_id": project_session_id
        })
        result = content[0].text

        assert "Error" in result or "Insufficient" in result
