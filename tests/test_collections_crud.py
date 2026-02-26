"""Tests for GlobalDB collection CRUD methods.

Tests all collection management functions:
- create_collection: Create a new collection with optional project assignments
- get_collection: Retrieve a collection with its metadata and projects
- list_collections: List all collections ordered by name
- get_collection_projects: Get projects in a collection
- add_project_to_collection: Add a project to an existing collection
- delete_collection: Delete a collection and clear project assignments
"""

import pytest
import tempfile
import uuid as uuid_module
from tessera.db import GlobalDB


@pytest.fixture
def global_db():
    """Create a temporary GlobalDB for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/global.db"
        db = GlobalDB(db_path=db_path)
        yield db
        db.close()


@pytest.fixture
def db_with_projects(global_db):
    """Create a GlobalDB with some test projects."""
    project1_id = global_db.register_project(
        path="/path/to/project1",
        name="Project 1",
        language="python"
    )
    project2_id = global_db.register_project(
        path="/path/to/project2",
        name="Project 2",
        language="typescript"
    )
    project3_id = global_db.register_project(
        path="/path/to/project3",
        name="Project 3",
        language="php"
    )
    return global_db, {
        "project1": project1_id,
        "project2": project2_id,
        "project3": project3_id
    }


class TestCreateCollection:
    """Test collection creation."""

    def test_create_collection_without_projects(self, global_db):
        """Create a collection without assigning projects."""
        collection_id = global_db.create_collection("My Collection")
        assert isinstance(collection_id, int)
        assert collection_id > 0

    def test_create_collection_with_projects(self, db_with_projects):
        """Create a collection and assign projects."""
        db, projects = db_with_projects
        collection_id = db.create_collection(
            "Team Collection",
            project_ids=[projects["project1"], projects["project2"]]
        )
        assert isinstance(collection_id, int)

        # Verify projects are assigned
        coll = db.get_collection(collection_id)
        assert len(coll["projects"]) == 2
        project_ids = {p["id"] for p in coll["projects"]}
        assert projects["project1"] in project_ids
        assert projects["project2"] in project_ids

    def test_create_collection_generates_scope_id(self, global_db):
        """Verify scope_id is generated on creation."""
        collection_id = global_db.create_collection("Scoped Collection")
        coll = global_db.get_collection(collection_id)
        assert coll["scope_id"] is not None
        # Verify it's a valid UUID format
        try:
            uuid_module.UUID(coll["scope_id"])
        except ValueError:
            pytest.fail("scope_id is not a valid UUID")

    def test_create_collection_duplicate_name_fails(self, global_db):
        """Creating collection with duplicate name should fail."""
        global_db.create_collection("Duplicate")
        with pytest.raises(Exception):  # SQLite UNIQUE constraint
            global_db.create_collection("Duplicate")


class TestGetCollection:
    """Test retrieving a collection."""

    def test_get_collection_by_id(self, global_db):
        """Retrieve a collection by ID."""
        collection_id = global_db.create_collection("Test Collection")
        coll = global_db.get_collection(collection_id)
        assert coll is not None
        assert coll["id"] == collection_id
        assert coll["name"] == "Test Collection"

    def test_get_collection_not_found(self, global_db):
        """Retrieving non-existent collection returns None."""
        coll = global_db.get_collection(999)
        assert coll is None

    def test_get_collection_returns_dict(self, global_db):
        """Collection dict has expected keys."""
        collection_id = global_db.create_collection("Complete Collection")
        coll = global_db.get_collection(collection_id)
        assert "id" in coll
        assert "name" in coll
        assert "scope_id" in coll
        assert "created_at" in coll
        assert "projects" in coll

    def test_get_collection_includes_projects(self, db_with_projects):
        """Collection includes assigned projects with correct fields."""
        db, projects = db_with_projects
        collection_id = db.create_collection(
            "Projects Collection",
            project_ids=[projects["project1"], projects["project2"]]
        )
        coll = db.get_collection(collection_id)
        assert len(coll["projects"]) == 2
        # Each project should have required fields
        for proj in coll["projects"]:
            assert "id" in proj
            assert "name" in proj
            assert "path" in proj
            assert "language" in proj


class TestListCollections:
    """Test listing collections."""

    def test_list_collections_empty(self, global_db):
        """Empty database returns empty list."""
        collections = global_db.list_collections()
        assert collections == []

    def test_list_collections_multiple(self, global_db):
        """List all collections."""
        global_db.create_collection("Alice")
        global_db.create_collection("Zebra")
        global_db.create_collection("Beta")

        collections = global_db.list_collections()
        assert len(collections) == 3

    def test_list_collections_ordered_by_name(self, global_db):
        """Collections are ordered by name."""
        global_db.create_collection("Zebra")
        global_db.create_collection("Alice")
        global_db.create_collection("Beta")

        collections = global_db.list_collections()
        names = [c["name"] for c in collections]
        assert names == ["Alice", "Beta", "Zebra"]

    def test_list_collections_includes_all_fields(self, global_db):
        """Each collection in list has required fields."""
        global_db.create_collection("Test")
        collections = global_db.list_collections()
        assert len(collections) > 0
        for coll in collections:
            assert "id" in coll
            assert "name" in coll
            assert "scope_id" in coll
            assert "created_at" in coll


class TestGetCollectionProjects:
    """Test retrieving projects in a collection."""

    def test_get_collection_projects_empty(self, global_db):
        """Collection with no projects returns empty list."""
        collection_id = global_db.create_collection("Empty Collection")
        projects = global_db.get_collection_projects(collection_id)
        assert projects == []

    def test_get_collection_projects_multiple(self, db_with_projects):
        """Get all projects in collection."""
        db, projects = db_with_projects
        collection_id = db.create_collection(
            "Projects Collection",
            project_ids=[projects["project1"], projects["project2"]]
        )
        proj_list = db.get_collection_projects(collection_id)
        assert len(proj_list) == 2

    def test_get_collection_projects_ordered_by_name(self, db_with_projects):
        """Projects are ordered by name."""
        db, projects = db_with_projects
        collection_id = db.create_collection(
            "Sorted Collection",
            project_ids=[projects["project3"], projects["project1"], projects["project2"]]
        )
        proj_list = db.get_collection_projects(collection_id)
        names = [p["name"] for p in proj_list]
        assert names == ["Project 1", "Project 2", "Project 3"]

    def test_get_collection_projects_has_required_fields(self, db_with_projects):
        """Each project has {id, name, path, language}."""
        db, projects = db_with_projects
        collection_id = db.create_collection(
            "Projects Collection",
            project_ids=[projects["project1"]]
        )
        proj_list = db.get_collection_projects(collection_id)
        assert len(proj_list) > 0
        for proj in proj_list:
            assert "id" in proj
            assert "name" in proj
            assert "path" in proj
            assert "language" in proj

    def test_get_collection_projects_nonexistent_collection(self, global_db):
        """Querying non-existent collection returns empty list."""
        projects = global_db.get_collection_projects(999)
        assert projects == []


class TestAddProjectToCollection:
    """Test adding projects to collections."""

    def test_add_project_to_collection(self, db_with_projects):
        """Add a project to a collection."""
        db, projects = db_with_projects
        collection_id = db.create_collection("Target Collection")

        # Initially empty
        proj_list = db.get_collection_projects(collection_id)
        assert len(proj_list) == 0

        # Add a project
        db.add_project_to_collection(collection_id, projects["project1"])

        # Now collection has the project
        proj_list = db.get_collection_projects(collection_id)
        assert len(proj_list) == 1
        assert proj_list[0]["id"] == projects["project1"]

    def test_add_multiple_projects_to_collection(self, db_with_projects):
        """Add multiple projects to a collection incrementally."""
        db, projects = db_with_projects
        collection_id = db.create_collection("Multi Collection")

        db.add_project_to_collection(collection_id, projects["project1"])
        db.add_project_to_collection(collection_id, projects["project2"])

        proj_list = db.get_collection_projects(collection_id)
        assert len(proj_list) == 2

    def test_add_project_invalid_collection_id(self, db_with_projects):
        """Adding to non-existent collection raises ValueError."""
        db, projects = db_with_projects
        with pytest.raises(ValueError):
            db.add_project_to_collection(999, projects["project1"])

    def test_add_project_invalid_project_id(self, global_db):
        """Adding non-existent project raises ValueError."""
        collection_id = global_db.create_collection("Test Collection")
        with pytest.raises(ValueError):
            global_db.add_project_to_collection(collection_id, 999)

    def test_add_project_updates_fk(self, db_with_projects):
        """Verify project's collection_id FK is updated."""
        db, projects = db_with_projects
        collection_id = db.create_collection("Target Collection")

        # Verify project not in collection initially
        proj = db.get_project(project_id=projects["project1"])
        assert proj["collection_id"] is None

        # Add to collection
        db.add_project_to_collection(collection_id, projects["project1"])

        # Verify FK is updated
        proj = db.get_project(project_id=projects["project1"])
        assert proj["collection_id"] == collection_id


class TestDeleteCollection:
    """Test deleting a collection."""

    def test_delete_collection(self, global_db):
        """Delete a collection."""
        collection_id = global_db.create_collection("To Delete")

        # Verify it exists
        coll = global_db.get_collection(collection_id)
        assert coll is not None

        # Delete it
        global_db.delete_collection(collection_id)

        # Verify it's gone
        coll = global_db.get_collection(collection_id)
        assert coll is None

    def test_delete_collection_clears_project_fk(self, db_with_projects):
        """Delete collection clears collection_id FK from projects."""
        db, projects = db_with_projects
        collection_id = db.create_collection(
            "To Delete",
            project_ids=[projects["project1"], projects["project2"]]
        )

        # Verify projects are in collection
        proj1 = db.get_project(project_id=projects["project1"])
        assert proj1["collection_id"] == collection_id

        # Delete collection
        db.delete_collection(collection_id)

        # Verify projects are no longer in collection
        proj1 = db.get_project(project_id=projects["project1"])
        assert proj1["collection_id"] is None
        proj2 = db.get_project(project_id=projects["project2"])
        assert proj2["collection_id"] is None

    def test_delete_collection_removes_list(self, global_db):
        """Deleted collection no longer appears in list."""
        collection_id = global_db.create_collection("To Delete")
        collections = global_db.list_collections()
        assert len(collections) == 1

        global_db.delete_collection(collection_id)
        collections = global_db.list_collections()
        assert len(collections) == 0

    def test_delete_nonexistent_collection(self, global_db):
        """Deleting non-existent collection should succeed silently."""
        # Should not raise an error
        global_db.delete_collection(999)
