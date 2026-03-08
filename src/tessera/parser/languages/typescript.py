"""TypeScript and JavaScript language extractor."""

import tree_sitter

from tessera.parser._base import LanguageExtractor
from tessera.parser._patterns import Reference, Symbol, find_child_by_type
from tessera.parser._references import (
    _extract_call_chain,
    _collect_type_references,
    _extract_comments_pass,
)
from tessera.parser._extractors import _node_signature
from tessera.parser._helpers import extract_first_string_arg_generic


class TypeScriptExtractor(LanguageExtractor):
    """Extractor for TypeScript source files."""

    language = "typescript"
    extensions = (".ts", ".tsx")
    grammar_module = "tree_sitter_typescript"
    grammar_func = "language_typescript"

    # Event pattern registry: method name → edge kind
    EVENT_REGISTERS = {
        "on": "registers_on",
        "once": "registers_on",
        "addListener": "registers_on",
        "addEventListener": "registers_on",
    }

    EVENT_FIRES = {
        "emit": "fires",
        "dispatchEvent": "fires",
        "trigger": "fires",
    }

    def extract_symbols(self, tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
        """Extract TypeScript symbol definitions."""
        symbols = []

        def walk(node, scope=""):
            if node.type == "function_declaration":
                # Function declaration
                name_node = find_child_by_type(node, "identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="function",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "class_declaration":
                # Class declaration — TS grammar uses type_identifier, JS uses identifier
                name_node = (
                    find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "identifier")
                )
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="class",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    # Walk class body
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "interface_declaration":
                # TypeScript interface
                name_node = find_child_by_type(node, "type_identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="interface",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "enum_declaration":
                # TypeScript enum
                name_node = find_child_by_type(node, "identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="enum",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    return

            elif node.type == "type_alias_declaration":
                # TypeScript type alias
                name_node = find_child_by_type(node, "type_identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="type",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    return

            elif node.type == "method_definition":
                # Method in class (also covers method_signature in interface body)
                name_node = find_child_by_type(node, "property_identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="method",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "method_signature":
                # Interface method signature
                name_node = find_child_by_type(node, "property_identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="method",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    return

            elif node.type == "variable_declarator":
                # Arrow function or const function expression
                name_node = find_child_by_type(node, "identifier")
                init = (
                    find_child_by_type(node, "arrow_function")
                    or find_child_by_type(node, "function_expression")
                )
                if name_node and init:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="function",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "pair":
                # Object property function: { click_open: function() {} }
                # or arrow: { handler: (e) => {} }
                key_node = find_child_by_type(node, "property_identifier")
                val = (
                    find_child_by_type(node, "function_expression")
                    or find_child_by_type(node, "arrow_function")
                )
                if key_node and val:
                    name = key_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="function",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=_node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "assignment_expression":
                # Property assignment: window.PUM.foo = function() {}
                # or: exports.foo = function() {}
                right = (
                    find_child_by_type(node, "function_expression")
                    or find_child_by_type(node, "arrow_function")
                )
                if right:
                    left = node.children[0] if node.children else None
                    name = None
                    if left and left.type == "member_expression":
                        # Extract rightmost property: window.PUM.foo -> "foo"
                        prop = find_child_by_type(left, "property_identifier")
                        if prop:
                            name = prop.text.decode("utf-8")
                    elif left and left.type == "identifier":
                        name = left.text.decode("utf-8")
                    if name:
                        sym = Symbol(
                            name=name,
                            kind="function",
                            line=node.start_point[0] + 1,
                            col=node.start_point[1],
                            end_line=node.end_point[0] + 1,
                            scope=scope,
                            signature=_node_signature(node),
                        )
                        symbols.append(sym)
                        for child in node.children:
                            walk(child, scope=name)
                        return

            elif node.type in (
                "import_statement",
                "import_specifier",
                "import_clause",
            ):
                # Imports - record as a single import symbol for now
                if node.type == "import_statement":
                    sym = Symbol(
                        name="",
                        kind="import",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                    )
                    symbols.append(sym)

            for child in node.children:
                walk(child, scope)

        walk(tree.root_node)
        return symbols

    def extract_references(
        self, tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] | None = None
    ) -> list[Reference]:
        """Extract TypeScript references (calls, types, extends, events)."""
        references = []
        chain_nodes = set()  # Track nodes already processed as part of a chain

        def walk(node, current_function=""):
            # Skip nodes already processed as part of a chain
            if id(node) in chain_nodes:
                for child in node.children:
                    walk(child, current_function)
                return

            # Handle function calls (including event detection)
            if node.type == "call_expression":
                func_node = node.children[0] if node.children else None
                if func_node:
                    func_name = None

                    if func_node.type == "identifier":
                        func_name = func_node.text.decode("utf-8")
                    elif func_node.type == "member_expression":
                        # obj.method() or obj.method().chain()
                        # First, check if this is an event function (on, emit, etc.)
                        prop = find_child_by_type(func_node, "property_identifier")
                        if prop:
                            method_name = prop.text.decode("utf-8")

                            # Check if it's an event register or event fire pattern
                            all_event_funcs = {**self.EVENT_REGISTERS, **self.EVENT_FIRES}
                            if method_name in all_event_funcs:
                                # Extract event name from first string argument
                                event_name = extract_first_string_arg_generic(node, source_code, self.language)
                                if event_name:
                                    references.append(
                                        Reference(
                                            from_symbol=current_function or "<module>",
                                            to_symbol=event_name,
                                            kind=all_event_funcs[method_name],
                                            line=node.start_point[0] + 1,
                                        )
                                    )
                                # Don't process as regular call
                                func_name = None
                            else:
                                # Regular method chain — try to extract the full chain
                                chain_names = _extract_call_chain(node, "typescript")
                                if chain_names:
                                    # Mark inner call nodes as processed to avoid double-counting
                                    obj = func_node.children[0] if func_node.children else None
                                    current = obj
                                    while current and current.type == "call_expression":
                                        inner_func = current.children[0] if current.children else None
                                        if inner_func and inner_func.type == "member_expression":
                                            chain_nodes.add(id(current))
                                            obj = inner_func.children[0] if inner_func.children else None
                                            current = obj
                                        else:
                                            break

                                    # Create references for all methods in the chain
                                    for method_name_chain in chain_names:
                                        ref = Reference(
                                            from_symbol=current_function or "<module>",
                                            to_symbol=method_name_chain,
                                            kind="calls",
                                            line=node.start_point[0] + 1,
                                        )
                                        references.append(ref)
                                    func_name = None  # Don't process again below

                    if func_name:
                        ref = Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=func_name,
                            kind="calls",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)

            # Handle constructor calls: new Foo()
            elif node.type == "new_expression":
                for child in node.children:
                    if child.type == "identifier":
                        class_name = child.text.decode("utf-8")
                        ref = Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=class_name,
                            kind="calls",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)
                        break

            # Handle class inheritance (extends)
            elif node.type == "class_declaration":
                name_node = (
                    find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "identifier")
                )
                class_name = (
                    name_node.text.decode("utf-8")
                    if name_node
                    else ""
                )
                # Look for class_heritage which contains extends/implements
                heritage_node = find_child_by_type(node, "class_heritage")
                if heritage_node:
                    for child in heritage_node.children:
                        if child.type == "extends_clause":
                            # extends ParentClass — parent name is identifier child
                            for ec_child in child.children:
                                if ec_child.type in ("identifier", "type_identifier"):
                                    parent_name = ec_child.text.decode("utf-8")
                                    ref = Reference(
                                        from_symbol=class_name,
                                        to_symbol=parent_name,
                                        kind="extends",
                                        line=node.start_point[0] + 1,
                                    )
                                    references.append(ref)
                        elif child.type == "implements_clause":
                            # class Foo implements Bar, Baz
                            for impl_child in child.children:
                                if impl_child.type in ("identifier", "type_identifier"):
                                    iface_name = impl_child.text.decode("utf-8")
                                    ref = Reference(
                                        from_symbol=class_name,
                                        to_symbol=iface_name,
                                        kind="implements",
                                        line=node.start_point[0] + 1,
                                    )
                                    references.append(ref)
                        elif child.type == "identifier":
                            # JS grammar fallback: heritage has bare identifiers
                            parent_name = child.text.decode("utf-8")
                            ref = Reference(
                                from_symbol=class_name,
                                to_symbol=parent_name,
                                kind="extends",
                                line=node.start_point[0] + 1,
                            )
                            references.append(ref)

                # Walk class body
                for child in node.children:
                    walk(child, current_function=class_name)
                return

            # Track current function/method for references
            elif node.type in ("function_declaration", "method_definition"):
                name_node = find_child_by_type(node, "identifier")
                if not name_node:
                    name_node = find_child_by_type(node, "property_identifier")
                func_name = (
                    name_node.text.decode("utf-8")
                    if name_node
                    else ""
                )
                # Walk function body
                for child in node.children:
                    walk(child, current_function=func_name)
                return

            # Type annotations (variable types, param types, return types, generics)
            elif node.type == "type_annotation":
                _collect_type_references(node, current_function or "<module>", references)
                return

            # Type alias declarations: type Foo = Bar & Baz
            elif node.type == "type_alias_declaration":
                decl_name = None
                for child in node.children:
                    if child.type == "type_identifier" and decl_name is None:
                        decl_name = child.text.decode("utf-8")
                    elif child.type not in ("type", "=", "type_identifier", "type_parameters"):
                        _collect_type_references(
                            child, decl_name or "<module>", references,
                            declaration_name=decl_name,
                        )
                return

            # as/satisfies expressions
            elif node.type in ("as_expression", "satisfies_expression"):
                for child in node.children:
                    if child.type in ("type_identifier", "generic_type", "union_type",
                                      "intersection_type", "nested_type_identifier"):
                        _collect_type_references(child, current_function or "<module>", references)
                    elif child.type not in ("as", "satisfies"):
                        walk(child, current_function)
                return

            # Type predicate annotations: x is Foo
            elif node.type in ("type_predicate_annotation", "type_predicate"):
                _collect_type_references(node, current_function or "<module>", references)
                return

            # Interface declarations — walk extends_type_clause for generic extends
            elif node.type == "interface_declaration":
                for child in node.children:
                    if child.type == "extends_type_clause":
                        _collect_type_references(child, current_function or "<module>", references)
                    else:
                        walk(child, current_function)
                return

            # Generic type parameter constraints and defaults
            elif node.type == "type_parameter":
                for child in node.children:
                    if child.type == "constraint" or child.type == "default_type":
                        _collect_type_references(child, current_function or "<module>", references)
                return

            for child in node.children:
                walk(child, current_function)

        walk(tree.root_node)

        # Second pass: extract types from JSDoc comments
        _extract_comments_pass(tree.root_node, references, "jsdoc")

        return references


class JavaScriptExtractor(TypeScriptExtractor):
    """Extractor for JavaScript source files.

    Uses the same extraction logic as TypeScript but with JavaScript grammar.
    """

    language = "javascript"
    extensions = (".js", ".jsx")
    grammar_module = "tree_sitter_javascript"
    grammar_func = "language"
