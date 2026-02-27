"""Tests for document extraction and chunking."""
import os
import json
import tempfile
import pytest
import yaml
from tessera.document import (
    DocumentExtractionError,
    DocumentChunk,
    chunk_markdown,
    chunk_yaml,
    chunk_json,
)


class TestChunkMarkdown:
    def test_splits_by_headers(self):
        md = "# Title\nIntro\n## Section 1\nContent 1\n## Section 2\nContent 2\n"
        chunks = chunk_markdown(md)
        assert len(chunks) >= 2
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.source_type == "markdown" for c in chunks)

    def test_tracks_line_numbers(self):
        md = "# Title\nLine2\nLine3\n## Next\nLine5\n"
        chunks = chunk_markdown(md)
        assert chunks[0].start_line >= 0

    def test_preserves_hierarchy(self):
        md = "# H1\nText\n## H2\nMore\n### H3\nDeep\n"
        chunks = chunk_markdown(md)
        # Should have parent_section tracking
        assert any(c.section_heading for c in chunks)

    def test_empty_input(self):
        chunks = chunk_markdown("")
        assert isinstance(chunks, list)


class TestChunkYaml:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_chunks_by_top_level_keys(self):
        content = {"server": {"port": 8080}, "database": {"url": "postgres://..."}}
        path = os.path.join(self.tmpdir, "test.yaml")
        with open(path, "w") as f:
            yaml.dump(content, f)
        chunks = chunk_yaml(path)
        assert len(chunks) >= 2
        assert all(c.source_type == "yaml" for c in chunks)
        keys = [c.key_path for c in chunks]
        assert "server" in keys
        assert "database" in keys

    def test_error_on_missing_file(self):
        with pytest.raises(DocumentExtractionError):
            chunk_yaml("/nonexistent/file.yaml")

    def test_malformed_yaml(self):
        path = os.path.join(self.tmpdir, "bad.yaml")
        with open(path, "w") as f:
            f.write(":\n  invalid: [\n")
        with pytest.raises(DocumentExtractionError):
            chunk_yaml(path)


class TestChunkJson:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_chunks_by_top_level_keys(self):
        content = {"api": {"version": "2.0"}, "routes": ["/a", "/b"]}
        path = os.path.join(self.tmpdir, "test.json")
        with open(path, "w") as f:
            json.dump(content, f)
        chunks = chunk_json(path)
        assert len(chunks) >= 2
        assert all(c.source_type == "json" for c in chunks)

    def test_error_on_missing_file(self):
        with pytest.raises(DocumentExtractionError):
            chunk_json("/nonexistent/file.json")
