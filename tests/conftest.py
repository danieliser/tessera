"""Test configuration and fixtures for CodeMem.

Phase 1: Basic fixtures for database, parser, and embedding client setup.
Will implement:
  - Temporary project directory fixture
  - In-memory SQLite database for testing
  - Mock embedding endpoint fixture
  - Test data generators (sample code files)
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_project_db(tmp_path):
    """Redirect ProjectDB storage to temp dir so tests don't pollute ~/.tessera/data/."""
    from tessera.db import ProjectDB
    old = ProjectDB.base_dir
    ProjectDB.base_dir = str(tmp_path / "tessera-data")
    yield
    ProjectDB.base_dir = old


@pytest.fixture
def test_project_dir(tmp_path):
    """Provide a temporary directory for test projects."""
    return tmp_path
