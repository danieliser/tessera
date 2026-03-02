"""Tree-sitter-based source code parser for multi-language projects.

Provides language detection, AST parsing, symbol extraction, and reference
resolution for building code dependency graphs.
"""

from tessera.parser._core import (
    _grammar_cache,
    _load_language,
    build_edges,
    detect_language,
    extract_references,
    extract_symbols,
    parse_and_extract,
    parse_file,
)
from tessera.parser._patterns import Edge, Reference, Symbol

__all__ = [
    "Symbol",
    "Reference",
    "Edge",
    "detect_language",
    "parse_file",
    "extract_symbols",
    "extract_references",
    "build_edges",
    "parse_and_extract",
    "_load_language",
    "_grammar_cache",
]
