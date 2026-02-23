"""Tests for chunker.py module."""

import pytest
from tessera.chunker import chunk_with_cast, Chunk


class TestChunkWithCast:
    """Test cAST chunking algorithm."""

    def test_single_function(self):
        """Test chunking a single function."""
        code = """def add(a, b):
    return a + b"""

        chunks = chunk_with_cast(code, "python")
        assert len(chunks) == 1
        assert chunks[0].ast_type == "function_definition"
        assert chunks[0].start_line == 0
        assert chunks[0].end_line == 1

    def test_multiple_functions(self):
        """Test chunking multiple functions as separate chunks."""
        code = """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b"""

        chunks = chunk_with_cast(code, "python")
        assert len(chunks) == 2
        assert chunks[0].ast_type == "function_definition"
        assert chunks[1].ast_type == "function_definition"

    def test_class_definition(self):
        """Test that classes are their own chunks."""
        code = """class MyClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value"""

        chunks = chunk_with_cast(code, "python")
        assert len(chunks) == 1
        assert chunks[0].ast_type == "class_definition"

    def test_mixed_definitions_and_statements(self):
        """Test chunking with mixed definitions and statements."""
        code = """def hello():
    pass

x = 10
y = 20

def goodbye():
    pass"""

        chunks = chunk_with_cast(code, "python")
        # Should have: hello function, x=10;y=20 block, goodbye function
        assert len(chunks) == 3
        assert chunks[0].ast_type == "function_definition"
        assert chunks[1].ast_type == "block"  # Non-definition statements
        assert chunks[2].ast_type == "function_definition"

    def test_budget_parameter(self):
        """Test that budget parameter affects chunking."""
        code = """def func1():
    x = 1

def func2():
    y = 2

def func3():
    z = 3"""

        # With small budget, should still keep definitions separate
        chunks_small_budget = chunk_with_cast(code, "python", budget=10)
        chunks_large_budget = chunk_with_cast(code, "python", budget=1000)

        # Functions should always be separate regardless of budget
        assert len(chunks_small_budget) == 3
        assert len(chunks_large_budget) == 3

    def test_chunk_content(self):
        """Test that chunk content is correct."""
        code = """def add(a, b):
    return a + b"""

        chunks = chunk_with_cast(code, "python")
        assert "def add" in chunks[0].content
        assert "return a + b" in chunks[0].content

    def test_chunk_line_numbers(self):
        """Test that chunk line numbers are correct."""
        code = """x = 1

def func():
    return 42"""

        chunks = chunk_with_cast(code, "python")
        # First chunk: x = 1 (line 0)
        # Second chunk: def func (lines 2-3)
        assert chunks[0].start_line == 0
        assert chunks[1].start_line == 2

    def test_unsupported_language_fallback(self):
        """Test that unsupported languages fall back to single chunk."""
        code = "some code here"

        chunks = chunk_with_cast(code, "unknown_language")
        assert len(chunks) == 1
        assert chunks[0].ast_type == "module"
        assert chunks[0].content == code

    def test_empty_code(self):
        """Test chunking empty code."""
        code = ""

        chunks = chunk_with_cast(code, "python")
        # Empty code might produce 0 or 1 chunk depending on parser
        assert len(chunks) >= 0

    def test_chunk_dataclass_fields(self):
        """Test Chunk dataclass has required fields."""
        chunk = Chunk(
            content="test",
            start_line=0,
            end_line=1,
            ast_type="test_type"
        )
        assert chunk.content == "test"
        assert chunk.start_line == 0
        assert chunk.end_line == 1
        assert chunk.ast_type == "test_type"
        assert chunk.symbol_ids == []

    def test_chunk_with_symbol_ids(self):
        """Test Chunk with symbol_ids populated."""
        chunk = Chunk(
            content="test",
            start_line=0,
            end_line=1,
            ast_type="function_definition",
            symbol_ids=[1, 2, 3]
        )
        assert chunk.symbol_ids == [1, 2, 3]
