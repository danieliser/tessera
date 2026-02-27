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
    chunk_plaintext,
    chunk_html,
    chunk_xml,
    chunk_text_file,
    strip_html,
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


class TestStripHtml:
    def test_strips_tags(self):
        html = "<h1>Title</h1><p>Hello <b>world</b></p>"
        text = strip_html(html)
        assert "Title" in text
        assert "Hello" in text
        assert "world" in text
        assert "<h1>" not in text

    def test_skips_script_and_style(self):
        html = "<p>Visible</p><script>var x=1;</script><style>.a{}</style><p>Also visible</p>"
        text = strip_html(html)
        assert "Visible" in text
        assert "Also visible" in text
        assert "var x" not in text
        assert ".a{}" not in text

    def test_empty_html(self):
        assert strip_html("") == ""


class TestChunkPlaintext:
    def test_basic_chunking(self):
        text = "\n".join(f"Line {i}" for i in range(100))
        chunks = chunk_plaintext(text, max_chunk_size=200)
        assert len(chunks) > 1
        assert all(c.source_type == "text" for c in chunks)

    def test_custom_source_type(self):
        chunks = chunk_plaintext("hello\nworld", source_type="csv")
        assert len(chunks) == 1
        assert chunks[0].source_type == "csv"

    def test_empty_input(self):
        assert chunk_plaintext("") == []
        assert chunk_plaintext("   \n  \n  ") == []

    def test_line_numbers(self):
        text = "\n".join(f"Line {i}" for i in range(50))
        chunks = chunk_plaintext(text, max_chunk_size=100)
        assert chunks[0].start_line == 0
        # Later chunks should have advancing line numbers
        if len(chunks) > 1:
            assert chunks[-1].start_line > 0


class TestChunkHtml:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_chunks_html_file(self):
        path = os.path.join(self.tmpdir, "page.html")
        with open(path, "w") as f:
            f.write("<html><body><h1>Title</h1><p>Content here</p></body></html>")
        chunks = chunk_html(path)
        assert len(chunks) >= 1
        assert all(c.source_type == "html" for c in chunks)
        assert any("Title" in c.content for c in chunks)

    def test_error_on_missing_file(self):
        with pytest.raises(DocumentExtractionError):
            chunk_html("/nonexistent/file.html")


class TestChunkXml:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_chunks_xml_file(self):
        path = os.path.join(self.tmpdir, "data.xml")
        with open(path, "w") as f:
            f.write("<root><item>Hello</item><item>World</item></root>")
        chunks = chunk_xml(path)
        assert len(chunks) >= 1
        assert all(c.source_type == "xml" for c in chunks)

    def test_error_on_missing_file(self):
        with pytest.raises(DocumentExtractionError):
            chunk_xml("/nonexistent/file.xml")


class TestChunkTextFile:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_chunks_txt_file(self):
        path = os.path.join(self.tmpdir, "readme.txt")
        with open(path, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")
        chunks = chunk_text_file(path, source_type="txt")
        assert len(chunks) >= 1
        assert all(c.source_type == "txt" for c in chunks)

    def test_chunks_csv_file(self):
        path = os.path.join(self.tmpdir, "data.csv")
        with open(path, "w") as f:
            f.write("name,age\nAlice,30\nBob,25\n")
        chunks = chunk_text_file(path, source_type="csv")
        assert len(chunks) >= 1
        assert any("Alice" in c.content for c in chunks)

    def test_error_on_missing_file(self):
        with pytest.raises(DocumentExtractionError):
            chunk_text_file("/nonexistent/file.txt")
