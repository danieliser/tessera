"""Go language extractor for symbols, references, and graph edges."""

import tree_sitter

from tessera.parser._base import LanguageExtractor
from tessera.parser._helpers import node_signature
from tessera.parser._patterns import Reference, Symbol, find_child_by_type


class GoExtractor(LanguageExtractor):
    """Extract Go symbols and references.

    Detects:
    - Symbols: functions, methods (with receiver scope), structs, interfaces,
      type aliases, consts, vars, imports
    - References: calls, struct embedding (extends), interface assertions (implements),
      type references in params/returns/receivers
    - Graph edges: containment via method→receiver scope, call/extends/implements edges
    """

    language = "go"
    extensions = (".go",)
    grammar_module = "tree_sitter_go"
    grammar_func = "language"

    # Go has no standard event/hook system
    EVENT_REGISTERS: dict[str, str] = {}
    EVENT_FIRES: dict[str, str] = {}

    def extract_symbols(self, tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
        """Extract Go symbol definitions."""
        symbols = []

        def walk(node, scope=""):
            if node.type == "function_declaration":
                name_node = find_child_by_type(node, "identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    symbols.append(Symbol(
                        name=name,
                        kind="function",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    ))
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "method_declaration":
                # Method name is field_identifier, not identifier
                name_node = find_child_by_type(node, "field_identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    # Receiver type becomes the scope → generates containment edge
                    receiver_type = _extract_receiver_type(node)
                    method_scope = receiver_type or scope
                    symbols.append(Symbol(
                        name=name,
                        kind="method",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=method_scope,
                        signature=node_signature(node),
                    ))
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "type_declaration":
                # type_declaration contains type_spec children
                for child in node.children:
                    if child.type == "type_spec":
                        _extract_type_spec(child, scope, symbols)
                return

            elif node.type == "const_declaration":
                for child in node.children:
                    if child.type == "const_spec":
                        _extract_const_or_var(child, "variable", scope, symbols)
                    elif child.type == "const_spec_list":
                        for spec in child.children:
                            if spec.type == "const_spec":
                                _extract_const_or_var(spec, "variable", scope, symbols)
                return

            elif node.type == "var_declaration":
                for child in node.children:
                    if child.type == "var_spec":
                        _extract_const_or_var(child, "variable", scope, symbols)
                    elif child.type == "var_spec_list":
                        for spec in child.children:
                            if spec.type == "var_spec":
                                _extract_const_or_var(spec, "variable", scope, symbols)
                return

            elif node.type == "import_declaration":
                symbols.append(Symbol(
                    name="",
                    kind="import",
                    line=node.start_point[0] + 1,
                    col=node.start_point[1],
                    end_line=node.end_point[0] + 1,
                ))
                return

            for child in node.children:
                walk(child, scope)

        walk(tree.root_node)
        return symbols

    def extract_references(
        self, tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] | None = None
    ) -> list[Reference]:
        """Extract Go references (calls, embedding, interface assertions, types)."""
        references: list[Reference] = []

        def walk(node, current_function=""):
            # Function calls: call_expression
            if node.type == "call_expression":
                _extract_call_ref(node, current_function, references)

            # Interface assertion: var _ Interface = (*Type)(nil)
            elif node.type == "var_declaration":
                _extract_interface_assertion(node, references)

            # Method declaration: receiver type reference
            elif node.type == "method_declaration":
                name_node = find_child_by_type(node, "field_identifier")
                method_name = name_node.text.decode("utf-8") if name_node else ""
                receiver_type = _extract_receiver_type(node)
                if receiver_type:
                    references.append(Reference(
                        from_symbol=method_name or current_function or "<module>",
                        to_symbol=receiver_type,
                        kind="type_reference",
                        line=node.start_point[0] + 1,
                    ))
                # Extract param/return type refs
                _extract_func_type_refs(node, method_name or current_function, references)
                # Walk body
                for child in node.children:
                    walk(child, current_function=method_name or current_function)
                return

            # Function declaration: extract param/return type refs
            elif node.type == "function_declaration":
                name_node = find_child_by_type(node, "identifier")
                func_name = name_node.text.decode("utf-8") if name_node else ""
                _extract_func_type_refs(node, func_name or current_function, references)
                for child in node.children:
                    walk(child, current_function=func_name or current_function)
                return

            # Type declaration: extract type refs from struct fields, interface methods, etc.
            elif node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        _extract_type_spec_refs(child, references)
                # Still walk children for nested refs
                for child in node.children:
                    walk(child, current_function)
                return

            for child in node.children:
                walk(child, current_function)

        walk(tree.root_node)
        return references


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

def _extract_receiver_type(method_node: tree_sitter.Node) -> str | None:
    """Extract receiver type from method_declaration, stripping pointer."""
    # First parameter_list is the receiver
    for child in method_node.children:
        if child.type == "parameter_list":
            for param in child.children:
                if param.type == "parameter_declaration":
                    for param_child in param.children:
                        if param_child.type == "pointer_type":
                            type_id = find_child_by_type(param_child, "type_identifier")
                            if type_id:
                                return type_id.text.decode("utf-8")
                        elif param_child.type == "type_identifier":
                            return param_child.text.decode("utf-8")
            # Only check the first parameter_list (receiver), not params
            return None
    return None


def _extract_type_spec(
    spec_node: tree_sitter.Node, scope: str, symbols: list[Symbol]
) -> None:
    """Extract a type_spec into the symbol list."""
    name_node = find_child_by_type(spec_node, "type_identifier")
    if not name_node:
        return
    name = name_node.text.decode("utf-8")

    # Determine kind from the type node
    kind = "type_alias"
    for child in spec_node.children:
        if child.type == "struct_type":
            kind = "class"
            break
        elif child.type == "interface_type":
            kind = "interface"
            break

    symbols.append(Symbol(
        name=name,
        kind=kind,
        line=spec_node.start_point[0] + 1,
        col=spec_node.start_point[1],
        end_line=spec_node.end_point[0] + 1,
        scope=scope,
        signature=node_signature(spec_node),
    ))


def _extract_const_or_var(
    spec_node: tree_sitter.Node, kind: str, scope: str, symbols: list[Symbol]
) -> None:
    """Extract a const_spec or var_spec into the symbol list."""
    name_node = find_child_by_type(spec_node, "identifier")
    if not name_node:
        return
    name = name_node.text.decode("utf-8")
    # Skip blank identifier
    if name == "_":
        return
    symbols.append(Symbol(
        name=name,
        kind=kind,
        line=spec_node.start_point[0] + 1,
        col=spec_node.start_point[1],
        end_line=spec_node.end_point[0] + 1,
        scope=scope,
    ))


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

_GO_BUILTIN_TYPES = frozenset({
    "string", "int", "int8", "int16", "int32", "int64",
    "uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
    "float32", "float64", "complex64", "complex128",
    "bool", "byte", "rune", "error", "any", "comparable",
})


def _extract_call_ref(
    call_node: tree_sitter.Node, current_function: str, references: list[Reference]
) -> None:
    """Extract call references from a call_expression."""
    func_node = call_node.children[0] if call_node.children else None
    if not func_node:
        return

    from_sym = current_function or "<module>"

    if func_node.type == "identifier":
        # Simple call: foo()
        name = func_node.text.decode("utf-8")
        # Skip builtins like make, len, append, etc.
        if name not in ("make", "len", "cap", "append", "copy", "delete",
                        "close", "panic", "recover", "print", "println",
                        "new", "complex", "real", "imag"):
            references.append(Reference(
                from_symbol=from_sym,
                to_symbol=name,
                kind="calls",
                line=call_node.start_point[0] + 1,
            ))

    elif func_node.type == "selector_expression":
        # pkg.Func() or obj.Method()
        field_node = find_child_by_type(func_node, "field_identifier")
        if field_node:
            name = field_node.text.decode("utf-8")
            references.append(Reference(
                from_symbol=from_sym,
                to_symbol=name,
                kind="calls",
                line=call_node.start_point[0] + 1,
            ))

    elif func_node.type == "parenthesized_expression":
        # Type conversion: (*Type)(val) — skip, not a real call
        pass


def _extract_embedding_ref(
    field_node: tree_sitter.Node, current_struct: str, references: list[Reference]
) -> None:
    """Extract struct embedding as extends reference.

    Embedding = field_declaration with type_identifier but no field_identifier.
    """
    has_field_name = False
    type_name = None

    for child in field_node.children:
        if child.type == "field_identifier":
            has_field_name = True
            break
        elif child.type == "type_identifier":
            type_name = child.text.decode("utf-8")
        elif child.type == "pointer_type":
            inner = find_child_by_type(child, "type_identifier")
            if inner:
                type_name = inner.text.decode("utf-8")
        elif child.type == "qualified_type":
            # pkg.Type embedding
            tid = find_child_by_type(child, "type_identifier")
            if tid:
                type_name = tid.text.decode("utf-8")

    if not has_field_name and type_name:
        references.append(Reference(
            from_symbol=current_struct or "<module>",
            to_symbol=type_name,
            kind="extends",
            line=field_node.start_point[0] + 1,
        ))


def _extract_interface_assertion(
    var_node: tree_sitter.Node, references: list[Reference]
) -> None:
    """Detect `var _ Interface = (*Type)(nil)` compile-time interface checks."""
    for child in var_node.children:
        if child.type not in ("var_spec", "var_spec_list"):
            continue
        specs = [child] if child.type == "var_spec" else [
            c for c in child.children if c.type == "var_spec"
        ]
        for spec in specs:
            id_node = find_child_by_type(spec, "identifier")
            if not id_node or id_node.text.decode("utf-8") != "_":
                continue
            # Find the interface type
            iface_node = find_child_by_type(spec, "type_identifier")
            if not iface_node:
                continue
            iface_name = iface_node.text.decode("utf-8")

            # Find the concrete type in the expression: (*Type)(nil)
            expr_list = find_child_by_type(spec, "expression_list")
            if not expr_list:
                continue
            concrete = _find_concrete_type_in_assertion(expr_list)
            if concrete:
                references.append(Reference(
                    from_symbol=concrete,
                    to_symbol=iface_name,
                    kind="implements",
                    line=spec.start_point[0] + 1,
                ))


def _find_concrete_type_in_assertion(node: tree_sitter.Node) -> str | None:
    """Find the concrete type in (*Type)(nil) or Type{} assertion patterns."""
    # Look for call_expression → parenthesized_expression → unary_expression → identifier
    for child in node.children:
        if child.type == "call_expression":
            paren = find_child_by_type(child, "parenthesized_expression")
            if paren:
                # (*Type) pattern
                unary = find_child_by_type(paren, "unary_expression")
                if unary:
                    type_id = find_child_by_type(unary, "identifier")
                    if type_id:
                        return type_id.text.decode("utf-8")
                # (Type) pattern (no pointer)
                type_id = find_child_by_type(paren, "identifier")
                if type_id:
                    return type_id.text.decode("utf-8")
        # Recurse
        result = _find_concrete_type_in_assertion(child)
        if result:
            return result
    return None


def _extract_func_type_refs(
    func_node: tree_sitter.Node, from_symbol: str, references: list[Reference]
) -> None:
    """Extract type references from function/method parameters and return types."""
    found_first_param_list = False
    for child in func_node.children:
        if child.type == "parameter_list":
            if not found_first_param_list and func_node.type == "method_declaration":
                # Skip receiver parameter list for methods
                found_first_param_list = True
                continue
            found_first_param_list = True
            _collect_type_refs_from_params(child, from_symbol, references)
        elif child.type in ("type_identifier", "pointer_type", "qualified_type",
                            "array_type", "slice_type", "map_type", "channel_type",
                            "function_type"):
            # Return type
            _collect_type_ref(child, from_symbol, references)


def _collect_type_refs_from_params(
    params_node: tree_sitter.Node, from_symbol: str, references: list[Reference]
) -> None:
    """Extract type references from a parameter_list."""
    for child in params_node.children:
        if child.type == "parameter_declaration":
            for param_child in child.children:
                if param_child.type in ("type_identifier", "pointer_type", "qualified_type",
                                        "array_type", "slice_type", "map_type",
                                        "channel_type", "function_type", "interface_type",
                                        "struct_type"):
                    _collect_type_ref(param_child, from_symbol, references)
        elif child.type == "variadic_parameter_declaration":
            for param_child in child.children:
                if param_child.type in ("type_identifier", "pointer_type", "qualified_type"):
                    _collect_type_ref(param_child, from_symbol, references)


def _collect_type_ref(
    type_node: tree_sitter.Node, from_symbol: str, references: list[Reference]
) -> None:
    """Collect a single type reference, skipping builtins."""
    if type_node.type == "type_identifier":
        name = type_node.text.decode("utf-8")
        if name not in _GO_BUILTIN_TYPES:
            references.append(Reference(
                from_symbol=from_symbol,
                to_symbol=name,
                kind="type_reference",
                line=type_node.start_point[0] + 1,
            ))
    elif type_node.type == "pointer_type":
        inner = find_child_by_type(type_node, "type_identifier")
        if inner:
            _collect_type_ref(inner, from_symbol, references)
    elif type_node.type == "qualified_type":
        # pkg.Type — reference the Type part
        tid = find_child_by_type(type_node, "type_identifier")
        if tid:
            _collect_type_ref(tid, from_symbol, references)
    elif type_node.type in ("array_type", "slice_type", "channel_type"):
        # Recurse into element type
        for child in type_node.children:
            if child.type in ("type_identifier", "pointer_type", "qualified_type",
                              "array_type", "slice_type", "map_type"):
                _collect_type_ref(child, from_symbol, references)
    elif type_node.type == "map_type":
        # Both key and value types
        for child in type_node.children:
            if child.type in ("type_identifier", "pointer_type", "qualified_type",
                              "array_type", "slice_type", "map_type"):
                _collect_type_ref(child, from_symbol, references)


def _extract_type_spec_refs(
    spec_node: tree_sitter.Node, references: list[Reference]
) -> None:
    """Extract references from within a type_spec (struct fields, interface embeds, etc.)."""
    name_node = find_child_by_type(spec_node, "type_identifier")
    type_name = name_node.text.decode("utf-8") if name_node else "<module>"

    for child in spec_node.children:
        if child.type == "struct_type":
            # Walk field declarations for type refs and embeddings
            field_list = find_child_by_type(child, "field_declaration_list")
            if field_list:
                for field in field_list.children:
                    if field.type == "field_declaration":
                        _extract_embedding_ref(field, type_name, references)
                        # Named fields: extract type ref
                        has_field_name = any(
                            c.type == "field_identifier" for c in field.children
                        )
                        if has_field_name:
                            for fc in field.children:
                                if fc.type in ("type_identifier", "pointer_type",
                                               "qualified_type", "array_type",
                                               "slice_type", "map_type"):
                                    _collect_type_ref(fc, type_name, references)
