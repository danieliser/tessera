"""Tests for indexer.py module."""

import os
import pytest
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from tessera.indexer import IndexerPipeline, IndexStats
from tessera.chunker import Chunk


# Mock classes for db, parser, and embeddings
class MockSymbol:
    """Mock Symbol dataclass."""
    def __init__(self, name, kind, line, col, scope="", signature=""):
        self.name = name
        self.kind = kind
        self.line = line
        self.col = col
        self.scope = scope
        self.signature = signature


class MockReference:
    """Mock Reference dataclass."""
    def __init__(self, from_symbol, to_symbol, kind="calls", context="", line=0):
        self.from_symbol = from_symbol
        self.to_symbol = to_symbol
        self.kind = kind
        self.context = context
        self.line = line


class MockEdge:
    """Mock Edge dataclass."""
    def __init__(self, from_name, to_name, type="call", weight=1.0):
        self.from_name = from_name
        self.to_name = to_name
        self.type = type
        self.weight = weight


class MockConnection:
    """Mock SQLite connection for transaction context manager."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockProjectDB:
    """Mock ProjectDB for testing."""
    def __init__(self):
        self.files = {}  # file_id -> {path, hash, index_status, ...}
        self.symbols = {}  # symbol_id -> {name, kind, ...}
        self.chunks = {}  # chunk_id -> {content, start_line, ...}
        self.next_file_id = 1
        self.next_symbol_id = 1
        self.next_chunk_id = 1
        self._old_hash = None
        self.conn = MockConnection()  # Support transaction wrapping

    def upsert_file(self, project_id, path, language, file_hash):
        """Upsert a file record, return file_id."""
        for fid, f in self.files.items():
            if f['path'] == path:
                self._old_hash = f['hash']
                f['hash'] = file_hash
                return fid
        fid = self.next_file_id
        self.next_file_id += 1
        self._old_hash = None
        self.files[fid] = {
            'project_id': project_id,
            'path': path,
            'language': language,
            'hash': file_hash,
            'index_status': 'pending'
        }
        return fid

    def get_old_hash(self):
        """Get the old hash from the last upsert_file call."""
        return self._old_hash

    def get_file(self, file_id):
        """Get file record by ID."""
        return self.files.get(file_id)

    def insert_symbols(self, symbol_dicts):
        """Insert symbols, return list of symbol IDs."""
        ids = []
        for sym_dict in symbol_dicts:
            sid = self.next_symbol_id
            self.next_symbol_id += 1
            self.symbols[sid] = sym_dict
            ids.append(sid)
        return ids

    def insert_refs(self, ref_dicts):
        """Insert references."""
        pass

    def insert_edges(self, edge_dicts):
        """Insert edges."""
        pass

    def insert_chunks(self, chunk_dicts):
        """Insert chunks."""
        for chunk_dict in chunk_dicts:
            cid = self.next_chunk_id
            self.next_chunk_id += 1
            self.chunks[cid] = chunk_dict

    def update_file_status(self, file_id, status):
        """Update file index status."""
        if file_id in self.files:
            self.files[file_id]['index_status'] = status

    def clear_file_data(self, file_id):
        """Clear old data for a file."""
        pass

    def keyword_search(self, query, limit=10):
        """Mock keyword search."""
        return []

    def get_all_embeddings(self):
        """Mock get all embeddings."""
        return None


class MockGlobalDB:
    """Mock GlobalDB for testing."""
    def register_project(self, path, name, language):
        """Register a project, return project_id."""
        return 1

    def create_job(self, project_id):
        return 1

    def start_job(self, job_id):
        pass

    def complete_job(self, job_id):
        pass

    def fail_job(self, job_id, error):
        pass


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory."""
    return tmp_path


@pytest.fixture
def mock_project_db():
    """Create a mock ProjectDB."""
    return MockProjectDB()


@pytest.fixture
def mock_global_db():
    """Create a mock GlobalDB."""
    return MockGlobalDB()


class TestDiscoverFiles:
    """Test file discovery."""

    def test_discover_files_python(self, temp_project_dir):
        """Test discovering Python files."""
        # Create files
        (temp_project_dir / "module1.py").write_text("print('hello')")
        (temp_project_dir / "module2.py").write_text("print('world')")
        (temp_project_dir / "image.png").write_bytes(b"\x89PNG\r\n")

        pipeline = IndexerPipeline(str(temp_project_dir), languages=['python'])
        files = pipeline._discover_files()

        assert len(files) == 2
        assert any('module1.py' in f for f in files)
        assert any('module2.py' in f for f in files)
        assert not any('image.png' in f for f in files)

    def test_discover_files_multiple_languages(self, temp_project_dir):
        """Test discovering files in multiple languages."""
        (temp_project_dir / "file.py").write_text("print('hello')")
        (temp_project_dir / "file.ts").write_text("console.log('hello')")
        (temp_project_dir / "file.php").write_text("<?php echo 'hello';")

        pipeline = IndexerPipeline(
            str(temp_project_dir),
            languages=['python', 'typescript', 'php']
        )
        files = pipeline._discover_files()

        assert len(files) == 3

    def test_discover_files_skips_hidden_dirs(self, temp_project_dir):
        """Test that hidden directories are skipped."""
        (temp_project_dir / ".hidden").mkdir()
        (temp_project_dir / ".hidden" / "file.py").write_text("print('hidden')")
        (temp_project_dir / "visible.py").write_text("print('visible')")

        pipeline = IndexerPipeline(str(temp_project_dir), languages=['python'])
        files = pipeline._discover_files()

        assert len(files) == 1
        assert 'visible.py' in files[0]

    def test_discover_files_skips_vendor_dirs(self, temp_project_dir):
        """Test that vendor and node_modules are skipped."""
        (temp_project_dir / "node_modules").mkdir()
        (temp_project_dir / "node_modules" / "file.py").write_text("print('nodemod')")
        (temp_project_dir / "vendor").mkdir()
        (temp_project_dir / "vendor" / "file.py").write_text("print('vendor')")
        (temp_project_dir / "main.py").write_text("print('main')")

        pipeline = IndexerPipeline(str(temp_project_dir), languages=['python'])
        files = pipeline._discover_files()

        assert len(files) == 1
        assert 'main.py' in files[0]

    def test_discover_files_empty_dir(self, temp_project_dir):
        """Test discovering files in empty directory."""
        pipeline = IndexerPipeline(str(temp_project_dir), languages=['python'])
        files = pipeline._discover_files()

        assert len(files) == 0

    def test_discover_files_nested_dirs(self, temp_project_dir):
        """Test discovering files in nested directories."""
        (temp_project_dir / "src").mkdir()
        (temp_project_dir / "src" / "main.py").write_text("print('main')")
        (temp_project_dir / "src" / "utils").mkdir(parents=True)
        (temp_project_dir / "src" / "utils" / "helper.py").write_text("def help(): pass")

        pipeline = IndexerPipeline(str(temp_project_dir), languages=['python'])
        files = pipeline._discover_files()

        assert len(files) == 2


class TestFileHash:
    """Test file hashing."""

    def test_file_hash_consistency(self, temp_project_dir):
        """Test that file hash is consistent."""
        filepath = temp_project_dir / "test.py"
        filepath.write_text("print('hello')")

        pipeline = IndexerPipeline(str(temp_project_dir))
        hash1 = pipeline._file_hash(str(filepath))
        hash2 = pipeline._file_hash(str(filepath))

        assert hash1 == hash2

    def test_file_hash_differs_on_change(self, temp_project_dir):
        """Test that hash changes when file changes."""
        filepath = temp_project_dir / "test.py"
        filepath.write_text("print('hello')")

        pipeline = IndexerPipeline(str(temp_project_dir))
        hash1 = pipeline._file_hash(str(filepath))

        filepath.write_text("print('world')")
        hash2 = pipeline._file_hash(str(filepath))

        assert hash1 != hash2

    def test_file_hash_format(self, temp_project_dir):
        """Test that hash is SHA-256 format."""
        filepath = temp_project_dir / "test.py"
        filepath.write_text("test")

        pipeline = IndexerPipeline(str(temp_project_dir))
        hash_val = pipeline._file_hash(str(filepath))

        # SHA-256 produces 64 hex characters
        assert len(hash_val) == 64
        assert all(c in '0123456789abcdef' for c in hash_val)


class TestIndexFile:
    """Test single file indexing."""

    @patch('tessera.indexer.detect_language')
    @patch('tessera.indexer.parse_and_extract')
    @patch('tessera.indexer.chunk_with_cast')
    def test_index_single_python_file(
        self, mock_chunk, mock_parse, mock_detect, temp_project_dir, mock_project_db
    ):
        """Test indexing a single Python file."""
        # Setup mocks
        mock_detect.return_value = 'python'
        mock_parse.return_value = (
            [MockSymbol('greet', 'function', 1, 0)],  # symbols
            [MockReference('greet', None, 'definition')],  # references
            []  # edges
        )
        mock_chunk.return_value = [
            Chunk("def greet():\n    print('hi')", 1, 2, "function_definition")
        ]

        # Create file
        filepath = temp_project_dir / "greet.py"
        filepath.write_text("def greet():\n    print('hi')")

        # Index
        pipeline = IndexerPipeline(
            str(temp_project_dir),
            project_db=mock_project_db,
            languages=['python']
        )
        pipeline.project_id = 1

        result = pipeline.index_file(str(filepath))

        assert result['status'] == 'indexed'
        assert result['symbols'] == 1
        assert result['chunks'] == 1

    @patch('tessera.indexer.detect_language')
    @patch('tessera.indexer.parse_and_extract')
    @patch('tessera.indexer.chunk_with_cast')
    def test_index_single_php_file(
        self, mock_chunk, mock_parse, mock_detect, temp_project_dir, mock_project_db
    ):
        """Test indexing a single PHP file."""
        mock_detect.return_value = 'php'
        mock_parse.return_value = (
            [MockSymbol('greeting', 'function', 1, 0)],
            [],
            []
        )
        mock_chunk.return_value = [
            Chunk("<?php function greeting() {}", 1, 1, "function_definition")
        ]

        filepath = temp_project_dir / "greet.php"
        filepath.write_text("<?php function greeting() {}")

        pipeline = IndexerPipeline(
            str(temp_project_dir),
            project_db=mock_project_db,
            languages=['php']
        )
        pipeline.project_id = 1

        result = pipeline.index_file(str(filepath))

        assert result['status'] == 'indexed'
        assert result['symbols'] == 1

    @patch('tessera.indexer.detect_language')
    @patch('tessera.indexer.parse_and_extract')
    @patch('tessera.indexer.chunk_with_cast')
    def test_index_skips_unchanged_file(
        self, mock_chunk, mock_parse, mock_detect, temp_project_dir, mock_project_db
    ):
        """Test that unchanged files are skipped on re-index."""
        mock_detect.return_value = 'python'
        mock_parse.return_value = ([], [], [])
        mock_chunk.return_value = []

        filepath = temp_project_dir / "file.py"
        filepath.write_text("x = 1")

        pipeline = IndexerPipeline(
            str(temp_project_dir),
            project_db=mock_project_db,
            languages=['python']
        )
        pipeline.project_id = 1

        # First index
        result1 = pipeline.index_file(str(filepath))
        assert result1['status'] == 'indexed'

        # Second index with same content
        result2 = pipeline.index_file(str(filepath))
        assert result2['status'] == 'skipped'
        assert result2['reason'] == 'unchanged'

    @patch('tessera.indexer.detect_language')
    @patch('tessera.indexer.parse_and_extract')
    @patch('tessera.indexer.chunk_with_cast')
    def test_index_reindexes_changed_file(
        self, mock_chunk, mock_parse, mock_detect, temp_project_dir, mock_project_db
    ):
        """Test that changed files are re-indexed."""
        mock_detect.return_value = 'python'
        mock_parse.return_value = (
            [MockSymbol('var', 'variable', 1, 0)],
            [],
            []
        )
        mock_chunk.return_value = [Chunk("x = 1", 1, 1, "block")]

        filepath = temp_project_dir / "file.py"
        filepath.write_text("x = 1")

        pipeline = IndexerPipeline(
            str(temp_project_dir),
            project_db=mock_project_db,
            languages=['python']
        )
        pipeline.project_id = 1

        # First index
        result1 = pipeline.index_file(str(filepath))
        assert result1['status'] == 'indexed'

        # Modify file
        filepath.write_text("x = 2")

        # Re-index
        result2 = pipeline.index_file(str(filepath))
        assert result2['status'] == 'indexed'

    @patch('tessera.indexer.detect_language')
    def test_index_skips_unsupported_language(self, mock_detect, temp_project_dir, mock_project_db):
        """Test that unsupported languages are skipped."""
        mock_detect.return_value = None

        filepath = temp_project_dir / "file.unknown"
        filepath.write_text("unknown code")

        pipeline = IndexerPipeline(str(temp_project_dir), project_db=mock_project_db)
        pipeline.project_id = 1

        result = pipeline.index_file(str(filepath))

        assert result['status'] == 'skipped'
        assert result['reason'] == 'unsupported language'

    @patch('tessera.indexer.detect_language')
    def test_index_handles_unreadable_file(self, mock_detect, temp_project_dir, mock_project_db):
        """Test graceful handling of unreadable files."""
        mock_detect.return_value = 'python'

        # Create a file path that doesn't exist
        filepath = temp_project_dir / "nonexistent.py"

        pipeline = IndexerPipeline(str(temp_project_dir), project_db=mock_project_db)
        pipeline.project_id = 1

        result = pipeline.index_file(str(filepath))

        assert result['status'] == 'failed'


class TestIndexProject:
    """Test project-wide indexing."""

    @patch('tessera.indexer.detect_language')
    @patch('tessera.indexer.parse_and_extract')
    @patch('tessera.indexer.chunk_with_cast')
    def test_index_project_multiple_files(
        self, mock_chunk, mock_parse, mock_detect, temp_project_dir, mock_project_db, mock_global_db
    ):
        """Test indexing an entire project."""
        mock_detect.return_value = 'python'
        mock_parse.return_value = (
            [MockSymbol('func', 'function', 1, 0)],
            [],
            []
        )
        mock_chunk.return_value = [Chunk("def func(): pass", 1, 1, "function_definition")]

        # Create multiple files
        (temp_project_dir / "file1.py").write_text("def func(): pass")
        (temp_project_dir / "file2.py").write_text("def func(): pass")
        (temp_project_dir / "image.png").write_bytes(b"\x89PNG\r\n")

        pipeline = IndexerPipeline(
            str(temp_project_dir),
            project_db=mock_project_db,
            global_db=mock_global_db,
            languages=['python']
        )

        stats = pipeline.index_project()

        assert stats.files_processed == 2
        assert stats.symbols_extracted == 2
        assert stats.chunks_created == 2
        assert stats.files_skipped == 0
        assert stats.files_failed == 0

    @patch('tessera.indexer.detect_language')
    @patch('tessera.indexer.parse_and_extract')
    @patch('tessera.indexer.chunk_with_cast')
    def test_index_project_with_embeddings(
        self, mock_chunk, mock_parse, mock_detect, temp_project_dir, mock_project_db, mock_global_db
    ):
        """Test indexing with embeddings enabled."""
        mock_detect.return_value = 'python'
        mock_parse.return_value = ([], [], [])
        mock_chunk.return_value = [Chunk("code", 1, 1, "block")]

        # Mock embedding client
        mock_embedding_client = Mock()
        mock_embedding_client.embed.return_value = [[0.1, 0.2, 0.3]]

        (temp_project_dir / "file.py").write_text("x = 1")

        pipeline = IndexerPipeline(
            str(temp_project_dir),
            project_db=mock_project_db,
            global_db=mock_global_db,
            embedding_client=mock_embedding_client,
            languages=['python']
        )

        stats = pipeline.index_project()

        assert stats.files_processed == 1
        assert stats.chunks_embedded == 1
        assert mock_embedding_client.embed.called

    def test_index_stats_accumulation(self, temp_project_dir, mock_project_db):
        """Test that IndexStats accumulates correctly."""
        pipeline = IndexerPipeline(str(temp_project_dir), project_db=mock_project_db)

        stats = IndexStats()
        stats.files_processed = 3
        stats.symbols_extracted = 10
        stats.chunks_created = 5

        assert stats.files_processed == 3
        assert stats.symbols_extracted == 10
        assert stats.chunks_created == 5


class TestRegister:
    """Test project registration."""

    def test_register_with_global_db(self, temp_project_dir, mock_global_db):
        """Test registering a project with GlobalDB."""
        pipeline = IndexerPipeline(
            str(temp_project_dir),
            global_db=mock_global_db
        )

        project_id = pipeline.register(name='test-project')

        assert project_id == 1
        assert pipeline.project_id == 1

    def test_register_without_global_db(self, temp_project_dir):
        """Test registering without GlobalDB defaults to id 1."""
        pipeline = IndexerPipeline(str(temp_project_dir))

        project_id = pipeline.register()

        assert project_id == 1
        assert pipeline.project_id == 1


class TestSearch:
    """Test search functionality."""

    @patch('tessera.indexer.hybrid_search')
    def test_search_without_embeddings(self, mock_hybrid, temp_project_dir, mock_project_db):
        """Test search without embeddings."""
        mock_hybrid.return_value = [
            {'id': 1, 'file_path': 'file.py', 'start_line': 1, 'end_line': 5}
        ]

        pipeline = IndexerPipeline(
            str(temp_project_dir),
            project_db=mock_project_db
        )

        results = pipeline.search('test query', limit=10)

        assert len(results) == 1
        assert results[0]['id'] == 1

    @patch('tessera.indexer.hybrid_search')
    def test_search_with_embeddings(self, mock_hybrid, temp_project_dir, mock_project_db):
        """Test search with embeddings enabled."""
        mock_hybrid.return_value = []

        # Mock embedding client
        mock_embedding_client = Mock()
        mock_embedding_client.embed_single.return_value = [0.1] * 768

        pipeline = IndexerPipeline(
            str(temp_project_dir),
            project_db=mock_project_db,
            embedding_client=mock_embedding_client
        )

        results = pipeline.search('query', limit=10)

        assert mock_embedding_client.embed_single.called


class TestParseFailurePreservesData:
    """Test that parse failure doesn't destroy existing symbol data."""

    def test_parse_failure_keeps_old_symbols(self, temp_project_dir):
        """If parse_and_extract raises, old data should still exist."""
        # Create a real file to index
        test_file = temp_project_dir / "module.py"
        test_file.write_text("def existing_func():\n    pass\n")

        pipeline = IndexerPipeline(str(temp_project_dir))
        pipeline.register()

        # First index: should succeed
        result = pipeline.index_file(str(test_file))
        assert result["status"] == "indexed"
        assert result["symbols"] > 0

        # Verify symbols exist
        syms = pipeline.project_db.lookup_symbols("existing_func")
        assert len(syms) > 0

        # Now update the file so hash changes (triggers re-index)
        test_file.write_text("def existing_func():\n    pass\n# changed\n")

        # Mock parse_and_extract to raise on the second call
        with patch('tessera.indexer.parse_and_extract', side_effect=RuntimeError("parse crash")):
            result = pipeline.index_file(str(test_file))
            assert result["status"] == "failed"

        # Old symbols should still exist (not cleared before failed parse)
        syms = pipeline.project_db.lookup_symbols("existing_func")
        assert len(syms) > 0

        pipeline.project_db.close()
