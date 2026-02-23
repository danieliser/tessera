"""Source code chunking strategy.

Phase 1: Design and implement chunking for semantic vector indexing.
Will implement:
  - AST-aware chunking (respect syntactic boundaries)
  - Sliding window approach with overlap for context
  - Metadata attachment (file, symbol scope, type)
  - Support for multiple chunk strategies (function-level, class-level, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional
from tree_sitter import Parser, Language


@dataclass
class Chunk:
    """A code chunk with AST metadata."""
    content: str
    start_line: int
    end_line: int
    ast_type: str  # 'function_def', 'class_def', 'module', 'comment', etc.
    symbol_ids: list[int] = field(default_factory=list)  # populated later during indexing


def _get_parser(language: str) -> tuple[Optional[Parser], Optional[Language]]:
    """Get tree-sitter parser and language for given language."""
    try:
        if language == "python":
            import tree_sitter_python
            py_lang_capsule = tree_sitter_python.language()
            py_lang = Language(py_lang_capsule)
        elif language in ("typescript", "javascript", "js", "ts"):
            import tree_sitter_javascript
            js_lang_capsule = tree_sitter_javascript.language()
            py_lang = Language(js_lang_capsule)
        elif language == "php":
            import tree_sitter_php
            php_lang_capsule = tree_sitter_php.language()
            py_lang = Language(php_lang_capsule)
        else:
            return None, None

        parser = Parser()
        parser.language = py_lang
        return parser, py_lang
    except Exception:
        return None, None


def _is_definition_node(node_type: str) -> bool:
    """Check if a node is a definition (function, class, etc.)."""
    definition_types = {
        "function_definition",
        "class_definition",
        "method_definition",
        "async_function_definition",
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    }
    return node_type in definition_types


def _count_non_whitespace(text: str) -> int:
    """Count non-whitespace characters."""
    return sum(1 for c in text if not c.isspace())


def _get_node_text(source: bytes, node) -> str:
    """Extract text content from a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def chunk_with_cast(source_code: str, language: str, budget: int = 512) -> list[Chunk]:
    """
    Code-Aware Structural (cAST) chunking.

    Parse with tree-sitter, greedily merge sibling AST nodes up to budget
    (measured by non-whitespace character count).

    Algorithm:
    1. Parse source with tree-sitter
    2. Walk top-level children of root node
    3. For definition nodes (functions, classes): each is its own chunk
    4. For non-definition nodes: merge siblings until budget exceeded
    5. If a single node exceeds budget, keep it as-is (don't split mid-definition)

    Args:
        source_code: Source code as string
        language: Language identifier (python, typescript, javascript, php)
        budget: Maximum non-whitespace chars per chunk (default 512)

    Returns:
        list of Chunks with content, line ranges, and AST type.
    """
    parser, lang = _get_parser(language)
    if parser is None or lang is None:
        # Fallback: return whole source as single chunk
        lines = source_code.split("\n")
        return [
            Chunk(
                content=source_code,
                start_line=0,
                end_line=len(lines) - 1,
                ast_type="module",
            )
        ]

    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    chunks = []
    current_chunk_nodes = []
    current_chunk_size = 0

    for child in root.children:
        node_type = child.type
        node_text = _get_node_text(source_bytes, child)
        node_size = _count_non_whitespace(node_text)

        # Definition nodes: always start a new chunk if current has content
        if _is_definition_node(node_type):
            # Flush current chunk if it has content
            if current_chunk_nodes:
                chunks.append(
                    _make_chunk(source_bytes, current_chunk_nodes, "block")
                )
                current_chunk_nodes = []
                current_chunk_size = 0

            # Add definition as its own chunk
            chunks.append(
                _make_chunk(source_bytes, [child], node_type)
            )
        else:
            # Non-definition: try to merge with current chunk
            if current_chunk_size + node_size <= budget:
                current_chunk_nodes.append(child)
                current_chunk_size += node_size
            else:
                # Budget exceeded
                if current_chunk_nodes:
                    chunks.append(
                        _make_chunk(source_bytes, current_chunk_nodes, "block")
                    )

                # Start new chunk with this node
                current_chunk_nodes = [child]
                current_chunk_size = node_size

    # Flush remaining chunk
    if current_chunk_nodes:
        chunks.append(
            _make_chunk(source_bytes, current_chunk_nodes, "block")
        )

    return chunks


def _make_chunk(source_bytes: bytes, nodes, ast_type: str) -> Chunk:
    """Create a Chunk from a list of tree-sitter nodes."""
    if not nodes:
        return Chunk("", 0, 0, "empty")

    first_node = nodes[0]
    last_node = nodes[-1]

    # Get line numbers (0-indexed start, inclusive end)
    start_line = first_node.start_point[0]
    end_line = last_node.end_point[0]

    # Extract text
    start_byte = first_node.start_byte
    end_byte = last_node.end_byte
    content = source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")

    return Chunk(
        content=content,
        start_line=start_line,
        end_line=end_line,
        ast_type=ast_type,
    )


if __name__ == "__main__":
    # Test chunking with a small Python snippet
    test_code = '''def hello():
    """A simple greeting function."""
    print("Hello, world!")

class MyClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value

# Some comments
x = 10
y = 20
z = x + y
'''

    chunks = chunk_with_cast(test_code, "python", budget=256)

    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Type: {chunk.ast_type}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        content_preview = chunk.content[:80].replace("\n", "\\n") if len(chunk.content) > 80 else chunk.content.replace("\n", "\\n")
        print(f"Content: {content_preview}...")
