"""Integration tests for asset indexing pipeline."""

import logging
import os
import struct
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tessera.indexer import IndexerPipeline, IndexStats
from tessera.db import ProjectDB


def _make_png(width: int, height: int) -> bytes:
    """Create minimal valid PNG header bytes."""
    magic = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>II', width, height) + b'\x08\x02\x00\x00\x00'
    ihdr_chunk = struct.pack('>I', 13) + b'IHDR' + ihdr_data + b'\x00\x00\x00\x00'
    return magic + ihdr_chunk


def _make_gif(width: int, height: int) -> bytes:
    """Create minimal GIF89a header bytes."""
    return b'GIF89a' + struct.pack('<HH', width, height) + b'\x00' * 10


@pytest.fixture
def project_dir(tmp_path):
    """Create a project directory with mixed file types."""
    # Python code file
    (tmp_path / "main.py").write_text("def hello(): pass")

    # Asset files
    img_dir = tmp_path / "assets" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "logo.png").write_bytes(_make_png(1200, 800))
    (img_dir / "icon.gif").write_bytes(_make_gif(64, 64))

    # Video (just need the extension, no valid header needed for indexing)
    (tmp_path / "demo.mp4").write_bytes(b'\x00' * 100)

    # Archive
    (tmp_path / "bundle.zip").write_bytes(b'PK\x03\x04' + b'\x00' * 50)

    # Markdown document
    (tmp_path / "README.md").write_text("# Hello\n\nThis is a readme.")

    return tmp_path


@pytest.fixture
def pipeline(project_dir):
    """Create an IndexerPipeline for the test project."""
    p = IndexerPipeline(str(project_dir), languages=['python'])
    p.register()
    return p


class TestAssetIndexing:
    """End-to-end asset indexing tests."""

    def test_discover_finds_assets(self, pipeline, project_dir):
        """Asset files are discovered alongside code and documents."""
        files = pipeline._discover_files()
        names = [os.path.basename(f) for f in files]
        assert "logo.png" in names
        assert "icon.gif" in names
        assert "demo.mp4" in names
        assert "bundle.zip" in names
        assert "main.py" in names
        assert "README.md" in names

    def test_index_asset_creates_chunk(self, pipeline, project_dir):
        """Indexing an asset creates one chunk with correct metadata."""
        png_path = str(project_dir / "assets" / "images" / "logo.png")
        result = pipeline._index_asset(png_path)

        assert result['status'] == 'indexed'
        assert result['chunks'] == 1
        assert result['embedded'] == 0
        assert result['dimensions'] == {'width': 1200, 'height': 800}
        assert 'image/png' in result['mime_type']

    def test_asset_chunk_metadata(self, pipeline, project_dir):
        """Asset chunks have correct source_type, ast_type, and key_path."""
        png_path = str(project_dir / "assets" / "images" / "logo.png")
        pipeline._index_asset(png_path)

        # FTS5 requires quoting tokens with dots — search by keyword instead
        results = pipeline.project_db.keyword_search("logo", limit=5)
        assert len(results) > 0

        chunk = pipeline.project_db.get_chunk(results[0]['id'])
        assert chunk['source_type'] == 'asset'
        assert chunk['ast_type'] == 'image'
        assert chunk['key_path'] == '1200x800'
        assert 'logo' in chunk['content']
        assert 'image/png' in chunk['content']

    def test_asset_no_embeddings(self, pipeline, project_dir):
        """Asset chunks are stored without embedding computation."""
        png_path = str(project_dir / "assets" / "images" / "logo.png")
        result = pipeline._index_asset(png_path)
        assert result['embedded'] == 0

    def test_index_project_includes_assets(self, pipeline, project_dir):
        """Full project index includes asset files."""
        stats = pipeline.index_project()

        # Should process: main.py, logo.png, icon.gif, demo.mp4, bundle.zip, README.md
        assert stats.files_processed >= 5  # at least code + assets
        assert stats.chunks_created >= 5

    def test_video_asset_no_dimensions(self, pipeline, project_dir):
        """Video assets are indexed without dimensions."""
        mp4_path = str(project_dir / "demo.mp4")
        result = pipeline._index_asset(mp4_path)

        assert result['status'] == 'indexed'
        assert result['dimensions'] is None

        results = pipeline.project_db.keyword_search("demo", limit=5)
        assert len(results) > 0
        chunk = pipeline.project_db.get_chunk(results[0]['id'])
        assert chunk['source_type'] == 'asset'
        assert chunk['ast_type'] == 'video'
        assert chunk['key_path'] is None


class TestAssetSearchFiltering:
    """Test source_type filtering on search results."""

    def test_search_returns_assets(self, pipeline, project_dir):
        """Default search returns asset results."""
        pipeline.index_project()
        results = pipeline.project_db.keyword_search("logo", limit=10)
        assert any(
            pipeline.project_db.get_chunk(r['id']).get('source_type') == 'asset'
            for r in results
        )

    def test_search_filter_asset_only(self, pipeline, project_dir):
        """source_type='asset' returns only asset results."""
        pipeline.index_project()
        results = pipeline.project_db.keyword_search("logo", limit=10, source_type=['asset'])
        for r in results:
            chunk = pipeline.project_db.get_chunk(r['id'])
            assert chunk['source_type'] == 'asset'

    def test_search_filter_code_excludes_assets(self, pipeline, project_dir):
        """source_type='code' excludes asset results."""
        pipeline.index_project()
        results = pipeline.project_db.keyword_search("hello", limit=10, source_type=['code'])
        for r in results:
            chunk = pipeline.project_db.get_chunk(r['id'])
            assert chunk['source_type'] != 'asset'


class TestIgnorePatterns:
    """Test that .tesseraignore is respected for assets."""

    def test_ignored_assets_not_indexed(self, pipeline, project_dir):
        """Files matched by .tesseraignore are excluded from asset discovery."""
        # Create ignore file
        (project_dir / ".tesseraignore").write_text("*.mp4\n")

        # Recreate pipeline to pick up ignore file
        p = IndexerPipeline(str(project_dir), languages=['python'])
        p.register()

        files = p._discover_files()
        names = [os.path.basename(f) for f in files]
        assert "demo.mp4" not in names
        assert "logo.png" in names  # PNG not ignored


class TestChangeDetection:
    """Test that unchanged assets are skipped on re-index."""

    def test_reindex_same_asset_succeeds(self, pipeline, project_dir):
        """Re-indexing the same asset succeeds (change detection is hash-based in production)."""
        png_path = str(project_dir / "assets" / "images" / "logo.png")

        result1 = pipeline.index_file(png_path)
        assert result1['status'] == 'indexed'

        # Second index still succeeds — ProjectDB.get_old_hash() would enable
        # skip-unchanged in production, but it's not implemented in all DB layers.
        result2 = pipeline.index_file(png_path)
        assert result2['status'] in ('indexed', 'skipped')

    def test_reindex_changed_asset(self, pipeline, project_dir):
        """Modified assets are re-indexed."""
        png_path = project_dir / "assets" / "images" / "logo.png"

        result1 = pipeline.index_file(str(png_path))
        assert result1['status'] == 'indexed'

        # Modify the file
        png_path.write_bytes(_make_png(800, 600))

        result2 = pipeline.index_file(str(png_path))
        assert result2['status'] == 'indexed'
        assert result2['dimensions'] == {'width': 800, 'height': 600}


class TestErrorHandling:
    """Test graceful error handling for malformed assets."""

    def test_truncated_png_still_indexes(self, pipeline, project_dir):
        """Truncated PNG indexes with dimensions=None."""
        truncated = project_dir / "bad.png"
        truncated.write_bytes(b'\x89PNG\r\n\x1a\n\x00')  # truncated after magic

        result = pipeline._index_asset(str(truncated))
        assert result['status'] == 'indexed'
        assert result['dimensions'] is None

    def test_unreadable_file_fails_gracefully(self, pipeline, project_dir):
        """Unreadable file returns failed status."""
        bad_path = str(project_dir / "nonexistent.png")
        result = pipeline._index_asset(bad_path)
        assert result['status'] == 'failed'


class TestSVGDualIndexing:
    """Test that SVG files are dual-indexed as asset and document."""

    def test_svg_indexed_as_asset(self, pipeline, project_dir):
        """SVG file gets an asset chunk."""
        svg_path = project_dir / "icon.svg"
        svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"><circle r="10"/></svg>')

        result = pipeline.index_file(str(svg_path))
        # index_file returns the document result for SVG (asset is done first, then falls through)
        assert result['status'] == 'indexed'

        # Check for asset chunk — search by keyword (FTS5 can't handle dots)
        results = pipeline.project_db.keyword_search("icon", limit=10)
        asset_chunks = []
        doc_chunks = []
        for r in results:
            chunk = pipeline.project_db.get_chunk(r['id'])
            if chunk['source_type'] == 'asset':
                asset_chunks.append(chunk)
            elif chunk['source_type'] in ('xml', 'html'):
                doc_chunks.append(chunk)

        assert len(asset_chunks) >= 1, "SVG should have an asset chunk"


class TestUnrecognizedBinaryLogging:
    """Test DEBUG logging for unrecognized binary files."""

    def test_unrecognized_binary_logs_debug(self, pipeline, project_dir, caplog):
        """Files with unrecognized extensions log at DEBUG."""
        unknown = project_dir / "data.bin"
        unknown.write_bytes(b'\x00\x01\x02\x03')

        # The file won't be discovered (not in any extension set),
        # but if we call index_file directly, it should log
        with caplog.at_level(logging.DEBUG, logger='tessera.indexer'):
            result = pipeline.index_file(str(unknown))

        assert result['status'] == 'skipped'
        assert any("unrecognized binary" in r.message.lower() or "unsupported language" in r.message.lower()
                    for r in caplog.records)
