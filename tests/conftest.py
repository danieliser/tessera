"""Test configuration and fixtures for CodeMem.

Phase 1: Basic fixtures for database, parser, and embedding client setup.
Will implement:
  - Temporary project directory fixture
  - In-memory SQLite database for testing
  - Mock embedding endpoint fixture
  - Test data generators (sample code files)
"""

import pytest


@pytest.fixture
def test_project_dir(tmp_path):
    """Provide a temporary directory for test projects."""
    return tmp_path
