"""Data classes and shared utilities for the parser package."""

from dataclasses import dataclass

import tree_sitter


@dataclass
class Symbol:
    """A symbol definition in source code (function, class, variable, etc.)."""

    name: str
    kind: str  # 'function', 'class', 'method', 'variable', 'constant', 'import'
    line: int
    col: int
    end_line: int = 0
    scope: str = ""  # parent class/namespace name
    signature: str = ""
    file_path: str = ""


@dataclass
class Reference:
    """A reference from one symbol to another (call, import, extends, etc.)."""

    from_symbol: str  # name of the calling symbol
    to_symbol: str  # name of the called symbol
    kind: str  # 'calls', 'imports', 'extends', 'implements', 'hooks_into'
    line: int
    context: str = ""  # surrounding code snippet


@dataclass
class Edge:
    """A graph edge representing a dependency relationship."""

    from_name: str
    to_name: str
    type: str  # 'calls', 'imports', 'extends', 'hooks_into'
    weight: float = 1.0


def find_child_by_type(
    node: tree_sitter.Node, node_type: str
) -> tree_sitter.Node | None:
    """Find the first child node of a given type."""
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def extract_first_string_arg(
    args_node: tree_sitter.Node, source_code: str
) -> str | None:
    """Extract the first string argument from a function call arguments node."""
    for child in args_node.children:
        if child.type == "argument":
            for subchild in child.children:
                if subchild.type in ("string", "encapsed_string"):
                    for content_child in subchild.children:
                        if content_child.type == "string_content":
                            return content_child.text.decode("utf-8")
                    text = subchild.text.decode("utf-8")
                    text = text.strip("'\"")
                    return text
        elif child.type in ("string", "encapsed_string", "heredoc", "string_literal"):
            for content_child in child.children:
                if content_child.type == "string_content":
                    return content_child.text.decode("utf-8")
            text = child.text.decode("utf-8")
            text = text.strip("'\"")
            return text
    return None
