"""Shared AST helper functions for language extractors."""

import tree_sitter

from tessera.parser._patterns import find_child_by_type


def node_signature(node: tree_sitter.Node) -> str:
    """Extract the first line of a node's source as its signature."""
    text = node.text.decode("utf-8") if node.text else ""
    return text.split("\n")[0].rstrip()


def extract_first_string_arg_generic(
    call_node: tree_sitter.Node, source_code: str, language: str
) -> str | None:
    """Extract the first string literal argument from a call node.

    Works across languages by trying common AST patterns for string args.
    """
    # Find the arguments node
    args_node = None
    for child in call_node.children:
        if child.type in ("arguments", "argument_list", "formal_parameters"):
            args_node = child
            break

    if not args_node:
        return None

    # Walk arguments looking for a string literal
    for child in args_node.children:
        result = _extract_string_value(child)
        if result is not None:
            return result
        # PHP wraps in "argument" nodes
        if child.type == "argument":
            for subchild in child.children:
                result = _extract_string_value(subchild)
                if result is not None:
                    return result
    return None


def _extract_string_value(node: tree_sitter.Node) -> str | None:
    """Extract string value from a string literal node."""
    string_types = ("string", "string_literal", "encapsed_string", "template_string")
    if node.type not in string_types:
        return None

    # Try to find string_content child (tree-sitter separates quotes from content)
    for child in node.children:
        if child.type in ("string_content", "string_fragment"):
            return child.text.decode("utf-8")

    # Fallback: strip quotes from raw text
    text = node.text.decode("utf-8")
    return text.strip("'\"`")
