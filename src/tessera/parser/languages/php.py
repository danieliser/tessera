"""PHP language extractor with WordPress hook support."""

import re

import tree_sitter

from tessera.parser._base import LanguageExtractor
from tessera.parser._helpers import node_signature, extract_first_string_arg_generic
from tessera.parser._patterns import Reference, Symbol, find_child_by_type


class PHPExtractor(LanguageExtractor):
    """PHP symbol and reference extractor with WordPress hook support.

    Detects:
    - Symbols: functions, classes, methods, traits, interfaces, namespaces
    - References: calls, extends, implements, type hints
    - Events: WordPress hooks (add_action, add_filter, do_action, apply_filters)
    """

    language = "php"
    extensions = (".php",)
    grammar_module = "tree_sitter_php"
    grammar_func = "language_php"

    # WordPress hook function mappings
    EVENT_REGISTERS = {
        "add_action": "registers_on",
        "add_filter": "registers_on",
    }
    EVENT_FIRES = {
        "do_action": "fires",
        "apply_filters": "fires",
    }

    def extract_symbols(self, tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
        """Extract PHP symbol definitions (functions, classes, methods, traits, interfaces)."""
        symbols = []
        current_namespace = ""

        def walk(node, scope=""):
            nonlocal current_namespace

            if node.type == "namespace_definition":
                # Track current namespace — persists for sibling declarations
                name_node = find_child_by_type(node, "namespace_name")
                if name_node:
                    current_namespace = name_node.text.decode("utf-8")
                # Walk namespace body (braced style)
                for child in node.children:
                    walk(child, scope)
                return

            if node.type == "function_definition":
                # Function definition
                name_node = find_child_by_type(node, "name")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    if current_namespace and not scope:
                        name = f"{current_namespace}\\{name}"
                    sym = Symbol(
                        name=name,
                        kind="function",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "class_declaration":
                # Class declaration
                name_node = find_child_by_type(node, "name")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    qualified = f"{current_namespace}\\{name}" if current_namespace else name
                    sym = Symbol(
                        name=qualified,
                        kind="class",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    )
                    symbols.append(sym)
                    # Walk class body
                    for child in node.children:
                        walk(child, scope=qualified)
                    return

            elif node.type == "method_declaration":
                # Method in class
                name_node = find_child_by_type(node, "name")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="method",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "namespace_use_declaration":
                # Use statement (import)
                sym = Symbol(
                    name="",
                    kind="import",
                    line=node.start_point[0] + 1,
                    col=node.start_point[1],
                    end_line=node.end_point[0] + 1,
                )
                symbols.append(sym)

            elif node.type == "trait_declaration":
                name_node = find_child_by_type(node, "name")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    qualified = f"{current_namespace}\\{name}" if current_namespace else name
                    sym = Symbol(
                        name=qualified,
                        kind="trait",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=qualified)
                    return

            elif node.type == "interface_declaration":
                name_node = find_child_by_type(node, "name")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    qualified = f"{current_namespace}\\{name}" if current_namespace else name
                    sym = Symbol(
                        name=qualified,
                        kind="interface",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=qualified)
                    return

            for child in node.children:
                walk(child, scope)

        walk(tree.root_node)
        return symbols

    def extract_references(
        self, tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] | None = None
    ) -> list[Reference]:
        """Extract PHP references (calls, extends, implements, hooks, type hints)."""
        references = []
        chain_nodes = set()  # Track nodes already processed as part of a chain

        def walk(node, current_function=""):
            # Skip nodes already processed as part of a chain
            if id(node) in chain_nodes:
                for child in node.children:
                    walk(child, current_function)
                return

            # Handle method calls ($obj->method())
            if node.type == "member_call_expression":
                # Try to extract the full chain first
                chain_names = _extract_call_chain(node)
                if chain_names:
                    # Mark inner calls as processed to avoid double-counting
                    for child in node.children:
                        if child.type == "member_call_expression":
                            chain_nodes.add(id(child))

                    # Create references for all methods in the chain
                    for method_name in chain_names:
                        ref = Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=method_name,
                            kind="calls",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)
                else:
                    # Fallback: extract method name from this level
                    method_node = find_child_by_type(node, "name")
                    if method_node:
                        func_name = method_node.text.decode("utf-8")
                        ref = Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=func_name,
                            kind="calls",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)

            # Handle function calls
            elif node.type == "function_call_expression":
                # Get the function name
                func_node = None
                for child in node.children:
                    if child.type == "name" or child.type == "qualified_name":
                        func_node = child
                        break

                if func_node:
                    func_name = func_node.text.decode("utf-8")

                    # Check for WordPress hooks
                    if func_name in self.EVENT_REGISTERS or func_name in self.EVENT_FIRES:
                        # Extract the first string argument as the hook name
                        args_node = find_child_by_type(node, "arguments")
                        if args_node:
                            hook_name = extract_first_string_arg_generic(node, source_code, "php")
                            if hook_name:
                                kind = (
                                    self.EVENT_REGISTERS.get(func_name)
                                    or self.EVENT_FIRES.get(func_name)
                                )
                                ref = Reference(
                                    from_symbol=current_function or "<module>",
                                    to_symbol=hook_name,
                                    kind=kind,
                                    line=node.start_point[0] + 1,
                                )
                                references.append(ref)
                    else:
                        # Regular function call
                        ref = Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=func_name,
                            kind="calls",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)

            # Handle constructor calls: new Foo()
            elif node.type == "object_creation_expression":
                for child in node.children:
                    if child.type in ("name", "qualified_name"):
                        class_name = child.text.decode("utf-8")
                        ref = Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=class_name,
                            kind="calls",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)
                        break

            # Handle class declarations with inheritance
            elif node.type == "class_declaration":
                name_node = find_child_by_type(node, "name")
                class_name = (
                    name_node.text.decode("utf-8")
                    if name_node
                    else ""
                )
                for child in node.children:
                    if child.type == "base_clause":
                        parent_node = find_child_by_type(child, "name")
                        if parent_node:
                            parent_name = parent_node.text.decode("utf-8")
                            ref = Reference(
                                from_symbol=class_name,
                                to_symbol=parent_name,
                                kind="extends",
                                line=node.start_point[0] + 1,
                            )
                            references.append(ref)
                    elif child.type == "class_interface_clause":
                        # class Foo implements Bar, Baz
                        for iface_child in child.children:
                            if iface_child.type == "name" or iface_child.type == "qualified_name":
                                references.append(Reference(
                                    from_symbol=class_name,
                                    to_symbol=iface_child.text.decode("utf-8"),
                                    kind="implements",
                                    line=iface_child.start_point[0] + 1,
                                ))

                # Walk class body (property types, method types)
                for child in node.children:
                    walk(child, current_function=class_name)
                return

            # Handle interface declarations
            elif node.type == "interface_declaration":
                name_node = find_child_by_type(node, "name")
                iface_name = (
                    name_node.text.decode("utf-8")
                    if name_node
                    else ""
                )
                # Walk interface body for method signatures
                for child in node.children:
                    walk(child, current_function=iface_name)
                return

            # Track current function for references + extract type refs
            elif node.type == "function_definition":
                name_node = find_child_by_type(node, "name")
                func_name = (
                    name_node.text.decode("utf-8")
                    if name_node
                    else ""
                )
                # Extract parameter types and return type
                for child in node.children:
                    if child.type == "formal_parameters":
                        _extract_php_param_types(child, func_name, references)
                    elif child.type in (
                        "named_type", "optional_type", "union_type",
                        "intersection_type", "primitive_type",
                    ):
                        # Return type
                        _collect_php_type_references(child, func_name, references)
                # Walk function body
                for child in node.children:
                    walk(child, current_function=func_name)
                return

            # Method declarations (in classes/interfaces)
            elif node.type == "method_declaration":
                name_node = find_child_by_type(node, "name")
                method_name = (
                    name_node.text.decode("utf-8")
                    if name_node
                    else ""
                )
                # Extract parameter types and return type
                for child in node.children:
                    if child.type == "formal_parameters":
                        _extract_php_param_types(child, method_name, references)
                    elif child.type in (
                        "named_type", "optional_type", "union_type",
                        "intersection_type", "primitive_type",
                    ):
                        # Return type
                        _collect_php_type_references(child, method_name, references)
                # Walk method body
                for child in node.children:
                    walk(child, current_function=method_name)
                return

            # Property declarations (class properties with types)
            elif node.type == "property_declaration":
                for child in node.children:
                    if child.type in (
                        "named_type", "optional_type", "union_type",
                        "intersection_type",
                    ):
                        _collect_php_type_references(
                            child, current_function or "<module>", references,
                        )
                        break

            for child in node.children:
                walk(child, current_function)

        walk(tree.root_node)

        # Second pass: extract types from PHPDoc comments
        _extract_comments_pass(tree.root_node, references, "phpdoc")

        return references


def _extract_call_chain(node: tree_sitter.Node) -> list[str]:
    """Extract all method names in a call chain.

    For a.b().c().d(), returns ['d', 'c', 'b'] (outermost to innermost).
    For a simple obj.method(), returns ['method'].

    Args:
        node: A member_call_expression node that may be part of a chain

    Returns:
        List of method names in the chain (outermost to innermost)
    """
    chain = []

    # PHP: member_call_expression → may have nested member_call_expression
    # Extract method name from this level
    method_node = find_child_by_type(node, "name")
    if method_node:
        chain.append(method_node.text.decode("utf-8"))

    # Check if this is part of a larger chain
    # Look for a child that's another member_call_expression or variable
    for child in node.children:
        if child.type == "member_call_expression":
            # Recurse
            chain.extend(_extract_call_chain(child))

    return chain


def _collect_php_type_references(
    node,
    from_symbol: str,
    references: list,
) -> None:
    """Extract type references from a PHP type node (named_type, optional_type, union_type).

    Skips primitive_type nodes (string, int, bool, float, void, etc.) since
    tree-sitter already separates them from named_type.
    """
    if node.type == "named_type":
        # Simple name or qualified_name inside
        for child in node.children:
            if child.type == "name":
                references.append(Reference(
                    from_symbol=from_symbol,
                    to_symbol=child.text.decode("utf-8"),
                    kind="type_reference",
                    line=child.start_point[0] + 1,
                ))
                return
            if child.type == "qualified_name":
                # Full qualified name including namespace
                references.append(Reference(
                    from_symbol=from_symbol,
                    to_symbol=child.text.decode("utf-8"),
                    kind="type_reference",
                    line=child.start_point[0] + 1,
                ))
                return
        return

    if node.type == "optional_type":
        # ?Type — recurse into children (skips the '?' token)
        for child in node.children:
            _collect_php_type_references(child, from_symbol, references)
        return

    if node.type == "union_type":
        # Type1|Type2 — recurse into each member
        for child in node.children:
            _collect_php_type_references(child, from_symbol, references)
        return

    if node.type == "intersection_type":
        for child in node.children:
            _collect_php_type_references(child, from_symbol, references)
        return


def _extract_php_param_types(
    params_node,
    from_symbol: str,
    references: list,
) -> None:
    """Extract type references from PHP formal_parameters node."""
    for child in params_node.children:
        if child.type == "simple_parameter":
            for param_child in child.children:
                if param_child.type in (
                    "named_type", "optional_type", "union_type",
                    "intersection_type",
                ):
                    _collect_php_type_references(
                        param_child, from_symbol, references,
                    )


# PHPDoc extraction for type hints in comments
_DOCBLOCK_PRIMITIVES = frozenset({
    "string", "number", "boolean", "undefined", "null", "void", "never",
    "any", "unknown", "object", "symbol", "bigint",  # JS/TS
    "int", "integer", "float", "double", "bool", "array", "callable",
    "iterable", "mixed", "self", "static", "parent", "true", "false",
    "resource",  # PHP
})

# PHPDoc: @param Type $name, @return Type, @var Type, @throws Type
# Handles: Type, ?Type, \Ns\Type, Type|Other, Type[]
_PHPDOC_TYPE_RE = re.compile(
    r"@(?:param|return|returns|var|throws|property(?:-read|-write)?)\s+"
    r"((?:\??\\?[\w\\]+(?:\[\])*(?:\|\??\\?[\w\\]+(?:\[\])*)*))"
)


def _extract_docblock_types(comment_text: str, style: str) -> list[str]:
    """Extract type names from a docblock comment."""
    types = []
    if style == "phpdoc":
        for match in _PHPDOC_TYPE_RE.finditer(comment_text):
            type_expr = match.group(1)
            for part in type_expr.split("|"):
                part = part.strip().lstrip("?")
                # Strip array suffix
                part = part.rstrip("[]")
                if part and part.lower() not in _DOCBLOCK_PRIMITIVES:
                    types.append(part)
    return types


def _extract_docblock_refs(
    comment_node,
    from_symbol: str,
    references: list,
    style: str,
) -> None:
    """Extract type references from a doc comment AST node."""
    text = comment_node.text.decode("utf-8")
    if not text.startswith("/**"):
        return
    type_names = _extract_docblock_types(text, style)
    line = comment_node.start_point[0] + 1
    for name in type_names:
        references.append(Reference(
            from_symbol=from_symbol,
            to_symbol=name,
            kind="type_reference",
            line=line,
        ))


def _get_declaration_name(node) -> str | None:
    """Extract the name from a declaration node."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return child.text.decode("utf-8")
        if child.type in (
            "function_declaration", "class_declaration",
            "interface_declaration", "method_declaration",
        ):
            return _get_declaration_name(child)
    return None


def _extract_comments_pass(
    root_node,
    references: list,
    style: str,
) -> None:
    """Walk the AST and extract type references from doc comments.

    Attributes refs to the next sibling declaration if present,
    otherwise to '<module>'.
    """
    # Determine which node types are declarations
    _DECL_TYPES = {
        "function_declaration", "function_definition", "method_definition",
        "class_declaration", "interface_declaration",
        "method_declaration", "property_declaration",
    }

    def _walk_for_comments(node, current_function=""):
        children = node.children
        for i, child in enumerate(children):
            if child.type == "comment":
                text = child.text.decode("utf-8")
                if not text.startswith("/**"):
                    continue

                # Find what this comment annotates
                from_symbol = current_function or "<module>"

                # Look at the next non-comment sibling
                for j in range(i + 1, len(children)):
                    sibling = children[j]
                    if sibling.type == "comment":
                        continue
                    if sibling.type in _DECL_TYPES:
                        # Find name of the declaration
                        name = _get_declaration_name(sibling)
                        if name:
                            from_symbol = name
                    break

                _extract_docblock_refs(child, from_symbol, references, style)

            elif child.type in (
                "class_declaration", "class_definition",
                "interface_declaration",
            ):
                # Enter class scope
                name = _get_declaration_name(child)
                _walk_for_comments(child, name or current_function)
            elif child.type in ("declaration_list", "class_body", "statement_block") or child.type == "program":
                _walk_for_comments(child, current_function)

    _walk_for_comments(root_node)
