"""Swift language extractor for symbols, references, and graph edges."""

import tree_sitter

from tessera.parser._base import LanguageExtractor
from tessera.parser._helpers import node_signature
from tessera.parser._patterns import Reference, Symbol, find_child_by_type


class SwiftExtractor(LanguageExtractor):
    """Extract Swift symbols and references.

    Detects:
    - Symbols: functions, methods, classes, structs, protocols, enums,
      extensions (as scope containers), properties, imports
    - References: calls, inheritance (extends), protocol conformance (implements),
      type references in params/returns
    - Graph edges: containment via method scope, call/extends/implements edges

    tree-sitter-swift uses `class_declaration` for class, struct, extension, and enum.
    Differentiation is by children:
    - `enum_class_body` → enum
    - Leading `user_type` before body → extension
    - `type_identifier` → class or struct (no further distinction in AST)
    """

    language = "swift"
    extensions = (".swift",)
    grammar_module = "tree_sitter_swift"
    grammar_func = "language"

    # Swift has no standard event/hook system
    EVENT_REGISTERS: dict[str, str] = {}
    EVENT_FIRES: dict[str, str] = {}

    def extract_symbols(self, tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
        """Extract Swift symbol definitions."""
        symbols = []

        def walk(node, scope=""):
            if node.type == "function_declaration":
                name_node = find_child_by_type(node, "simple_identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    kind = "method" if scope else "function"
                    symbols.append(Symbol(
                        name=name,
                        kind=kind,
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    ))
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "protocol_declaration":
                name_node = find_child_by_type(node, "type_identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    symbols.append(Symbol(
                        name=name,
                        kind="interface",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    ))
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "class_declaration":
                info = _classify_class_declaration(node)
                if info["is_extension"]:
                    # Extensions don't create new symbols — they add to an existing type.
                    # Walk body with the extended type as scope.
                    ext_scope = info["extended_type"] or scope
                    for child in node.children:
                        walk(child, scope=ext_scope)
                    return
                else:
                    name = info["name"] or ""
                    if name:
                        symbols.append(Symbol(
                            name=name,
                            kind=info["kind"],
                            line=node.start_point[0] + 1,
                            col=node.start_point[1],
                            end_line=node.end_point[0] + 1,
                            scope=scope,
                            signature=node_signature(node),
                        ))
                    for child in node.children:
                        walk(child, scope=name or scope)
                    return

            elif node.type == "protocol_function_declaration":
                # Method signatures inside protocols
                name_node = find_child_by_type(node, "simple_identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    symbols.append(Symbol(
                        name=name,
                        kind="method",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                    ))

            elif node.type == "import_declaration":
                symbols.append(Symbol(
                    name="",
                    kind="import",
                    line=node.start_point[0] + 1,
                    col=node.start_point[1],
                    end_line=node.end_point[0] + 1,
                ))
                return

            elif node.type == "property_declaration":
                # Top-level let/var or class properties
                name = _extract_property_name(node)
                if name:
                    symbols.append(Symbol(
                        name=name,
                        kind="variable",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                    ))

            for child in node.children:
                walk(child, scope)

        walk(tree.root_node)
        return symbols

    def extract_references(
        self, tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] | None = None
    ) -> list[Reference]:
        """Extract Swift references (calls, inheritance, conformance, types)."""
        references: list[Reference] = []

        def walk(node, current_function=""):
            # Function calls
            if node.type == "call_expression":
                _extract_call_ref(node, current_function, references)

            # Class/struct/extension declarations: inheritance + conformance
            elif node.type == "class_declaration":
                info = _classify_class_declaration(node)
                if info["is_extension"]:
                    # Extension conformance: extension Foo: Bar → implements
                    ext_type = info["extended_type"] or ""
                    for proto in info["conformances"]:
                        references.append(Reference(
                            from_symbol=ext_type or current_function or "<module>",
                            to_symbol=proto,
                            kind="implements",
                            line=node.start_point[0] + 1,
                        ))
                    for child in node.children:
                        walk(child, current_function=ext_type or current_function)
                    return
                else:
                    name = info["name"] or ""
                    # First conformance on a class is extends (superclass), rest are implements
                    for i, parent in enumerate(info["conformances"]):
                        kind = "extends" if i == 0 and not info["is_enum"] else "implements"
                        references.append(Reference(
                            from_symbol=name or current_function or "<module>",
                            to_symbol=parent,
                            kind=kind,
                            line=node.start_point[0] + 1,
                        ))
                    for child in node.children:
                        walk(child, current_function=name or current_function)
                    return

            # Protocol declarations with inheritance
            elif node.type == "protocol_declaration":
                name_node = find_child_by_type(node, "type_identifier")
                proto_name = name_node.text.decode("utf-8") if name_node else ""
                for child in node.children:
                    if child.type == "inheritance_specifier":
                        parent = _extract_type_from_inheritance(child)
                        if parent:
                            references.append(Reference(
                                from_symbol=proto_name or current_function or "<module>",
                                to_symbol=parent,
                                kind="extends",
                                line=child.start_point[0] + 1,
                            ))
                for child in node.children:
                    walk(child, current_function=proto_name or current_function)
                return

            # Function declarations: extract param/return type refs
            elif node.type == "function_declaration":
                name_node = find_child_by_type(node, "simple_identifier")
                func_name = name_node.text.decode("utf-8") if name_node else ""
                _extract_func_type_refs(node, func_name or current_function, references)
                for child in node.children:
                    walk(child, current_function=func_name or current_function)
                return

            for child in node.children:
                walk(child, current_function)

        walk(tree.root_node)
        return references


# ---------------------------------------------------------------------------
# Swift-specific helpers
# ---------------------------------------------------------------------------

_SWIFT_BUILTIN_TYPES = frozenset({
    "String", "Int", "Double", "Float", "Bool", "Void",
    "Int8", "Int16", "Int32", "Int64",
    "UInt", "UInt8", "UInt16", "UInt32", "UInt64",
    "Character", "Any", "AnyObject", "Never", "Optional",
    "Error", "Codable", "Hashable", "Equatable", "Comparable",
    "Identifiable", "Sendable",
})


def _classify_class_declaration(node: tree_sitter.Node) -> dict:
    """Classify a class_declaration as class/struct/enum/extension.

    Returns dict with: name, kind, is_extension, is_enum, extended_type, conformances
    """
    result = {
        "name": None,
        "kind": "class",
        "is_extension": False,
        "is_enum": False,
        "extended_type": None,
        "conformances": [],
    }

    # Check for enum body
    for child in node.children:
        if child.type == "enum_class_body":
            result["is_enum"] = True
            break

    # Look at children structure to detect extension vs class/struct
    # Extension pattern: class_declaration → user_type (extended type) → inheritance_specifier → class_body
    # Class pattern: class_declaration → type_identifier (name) → inheritance_specifier → class_body
    children = [c for c in node.children if c.is_named]

    # Find type_identifier (class/struct name) or user_type (extension target)
    type_id = find_child_by_type(node, "type_identifier")
    user_type = find_child_by_type(node, "user_type")

    if type_id:
        result["name"] = type_id.text.decode("utf-8")
        result["kind"] = "class"  # enum already set above
    elif user_type and not type_id:
        # Extension: the user_type is the extended type
        result["is_extension"] = True
        inner_type = find_child_by_type(user_type, "type_identifier")
        if inner_type:
            result["extended_type"] = inner_type.text.decode("utf-8")

    # Collect conformances/inheritance
    for child in node.children:
        if child.type == "inheritance_specifier":
            parent = _extract_type_from_inheritance(child)
            if parent:
                result["conformances"].append(parent)

    return result


def _extract_type_from_inheritance(node: tree_sitter.Node) -> str | None:
    """Extract type name from an inheritance_specifier node."""
    user_type = find_child_by_type(node, "user_type")
    if user_type:
        type_id = find_child_by_type(user_type, "type_identifier")
        if type_id:
            return type_id.text.decode("utf-8")
    return None


def _extract_property_name(node: tree_sitter.Node) -> str | None:
    """Extract the property name from a property_declaration."""
    # Look for pattern → simple_identifier
    pattern = find_child_by_type(node, "pattern")
    if pattern:
        name_node = find_child_by_type(pattern, "simple_identifier")
        if name_node:
            return name_node.text.decode("utf-8")
    return None


def _extract_call_ref(
    call_node: tree_sitter.Node, current_function: str, references: list[Reference]
) -> None:
    """Extract call references from a call_expression."""
    from_sym = current_function or "<module>"

    # Direct call: simple_identifier(...)
    name_node = find_child_by_type(call_node, "simple_identifier")
    if name_node:
        name = name_node.text.decode("utf-8")
        references.append(Reference(
            from_symbol=from_sym,
            to_symbol=name,
            kind="calls",
            line=call_node.start_point[0] + 1,
        ))
        return

    # Navigation call: obj.method(...)
    nav_node = find_child_by_type(call_node, "navigation_expression")
    if nav_node:
        nav_suffix = find_child_by_type(nav_node, "navigation_suffix")
        if nav_suffix:
            method_name = find_child_by_type(nav_suffix, "simple_identifier")
            if method_name:
                references.append(Reference(
                    from_symbol=from_sym,
                    to_symbol=method_name.text.decode("utf-8"),
                    kind="calls",
                    line=call_node.start_point[0] + 1,
                ))


def _extract_func_type_refs(
    func_node: tree_sitter.Node, from_symbol: str, references: list[Reference]
) -> None:
    """Extract type references from function parameters and return type."""
    for child in func_node.children:
        if child.type == "parameter":
            # Parameter type
            for param_child in child.children:
                if param_child.type == "user_type":
                    _collect_type_ref(param_child, from_symbol, references, func_node.start_point[0] + 1)
                elif param_child.type == "array_type":
                    _collect_array_type_ref(param_child, from_symbol, references, func_node.start_point[0] + 1)
                elif param_child.type == "optional_type":
                    _collect_optional_type_ref(param_child, from_symbol, references, func_node.start_point[0] + 1)
        elif child.type == "user_type":
            # Return type
            _collect_type_ref(child, from_symbol, references, func_node.start_point[0] + 1)
        elif child.type == "array_type":
            _collect_type_ref(child, from_symbol, references, func_node.start_point[0] + 1)
        elif child.type == "optional_type":
            _collect_optional_type_ref(child, from_symbol, references, func_node.start_point[0] + 1)


def _collect_type_ref(
    type_node: tree_sitter.Node, from_symbol: str, references: list[Reference], line: int
) -> None:
    """Collect a type reference from a user_type or array_type node."""
    if type_node.type == "user_type":
        type_id = find_child_by_type(type_node, "type_identifier")
        if type_id:
            name = type_id.text.decode("utf-8")
            if name not in _SWIFT_BUILTIN_TYPES:
                references.append(Reference(
                    from_symbol=from_symbol,
                    to_symbol=name,
                    kind="type_reference",
                    line=line,
                ))
    elif type_node.type == "array_type":
        _collect_array_type_ref(type_node, from_symbol, references, line)


def _collect_array_type_ref(
    array_node: tree_sitter.Node, from_symbol: str, references: list[Reference], line: int
) -> None:
    """Extract type from [Type] array syntax."""
    user_type = find_child_by_type(array_node, "user_type")
    if user_type:
        _collect_type_ref(user_type, from_symbol, references, line)


def _collect_optional_type_ref(
    opt_node: tree_sitter.Node, from_symbol: str, references: list[Reference], line: int
) -> None:
    """Extract type from Type? optional syntax."""
    user_type = find_child_by_type(opt_node, "user_type")
    if user_type:
        _collect_type_ref(user_type, from_symbol, references, line)
