"""Generic fallback language extractor for unsupported languages."""

import tree_sitter

from tessera.parser._base import LanguageExtractor
from tessera.parser._helpers import node_signature
from tessera.parser._patterns import Reference, Symbol, find_child_by_type


class GenericExtractor(LanguageExtractor):
    """Generic extractor for unsupported languages using common AST patterns.

    Looks for common node types across C-family and other languages:
    - Functions: function_definition, function_declaration, function_item
    - Classes: class_definition, class_declaration, struct_item, struct_declaration
    - Methods: method_definition, method_declaration
    - Types: interface_declaration, enum_declaration, enum_item, trait_item
    - Imports: import_statement, import_declaration, use_declaration
    """

    language = "_generic"
    extensions = ()  # Never auto-matched by extension
    grammar_module = ""
    grammar_func = ""

    EVENT_REGISTERS = {}
    EVENT_FIRES = {}

    def extract_symbols(self, tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
        """Extract symbols using common AST node types."""
        symbols = []
        current_class = None

        def walk(node, scope=""):
            nonlocal current_class

            # Function/method definitions
            if node.type in (
                "function_definition",
                "function_declaration",
                "function_item",
            ):
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
                )
                if name_node:
                    name = name_node.text.decode("utf-8")
                    kind = "method" if current_class else "function"
                    sym = Symbol(
                        name=name,
                        kind=kind,
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

            # Class definitions
            elif node.type in ("class_definition", "class_declaration"):
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
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
                        signature=node_signature(node),
                    )
                    symbols.append(sym)
                    old_class = current_class
                    current_class = name
                    for child in node.children:
                        walk(child, scope=name)
                    current_class = old_class
                    return

            # Struct definitions (treat as classes)
            elif node.type in ("struct_item", "struct_declaration"):
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
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
                        signature=node_signature(node),
                    )
                    symbols.append(sym)
                    for child in node.children:
                        walk(child, scope=name)
                    return

            # Method definitions (already covered by function_definition in class context,
            # but explicit for some languages)
            elif node.type in ("method_definition", "method_declaration"):
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
                )
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

            # Interface definitions
            elif node.type == "interface_declaration":
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
                )
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="interface",
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

            # Enum definitions
            elif node.type in ("enum_declaration", "enum_item"):
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
                )
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="enum",
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

            # Trait definitions
            elif node.type == "trait_item":
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
                )
                if name_node:
                    name = name_node.text.decode("utf-8")
                    sym = Symbol(
                        name=name,
                        kind="trait",
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

            # Import statements
            elif node.type in (
                "import_statement",
                "import_declaration",
                "use_declaration",
            ):
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
        self, tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
    ) -> list[Reference]:
        """Extract references using common AST node types.

        Looks for:
        - call_expression, identifier/member_expression → kind="calls"
        - new_expression, object_creation_expression → kind="calls"
        - Class nodes with superclass, base_clause, heritage children → kind="extends"
        """
        references = []

        def walk(node, current_function=""):
            # Handle function/method calls
            if node.type == "call_expression":
                func_node = node.children[0] if node.children else None
                if func_node:
                    func_name = None
                    if func_node.type == "identifier":
                        func_name = func_node.text.decode("utf-8")
                    elif func_node.type == "member_expression":
                        # obj.method() → extract "method" (rightmost identifier)
                        for child in func_node.children:
                            if child.type == "identifier":
                                func_name = child.text.decode("utf-8")
                    if func_name:
                        ref = Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=func_name,
                            kind="calls",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)

            # Handle constructor calls: new Foo()
            elif node.type in ("new_expression", "object_creation_expression"):
                for child in node.children:
                    if child.type in ("identifier", "type_identifier"):
                        class_name = child.text.decode("utf-8")
                        ref = Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=class_name,
                            kind="calls",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)
                        break

            # Handle class inheritance
            elif node.type in ("class_declaration", "class_definition"):
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
                )
                class_name = name_node.text.decode("utf-8") if name_node else ""
                # Look for superclass, base_clause, or class_heritage
                for child in node.children:
                    if child.type in ("superclass", "base_clause", "class_heritage"):
                        for subchild in child.children:
                            if subchild.type in ("identifier", "type_identifier"):
                                parent_name = subchild.text.decode("utf-8")
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
            elif node.type in (
                "function_declaration",
                "function_definition",
                "function_item",
                "method_definition",
                "method_declaration",
            ):
                name_node = (
                    find_child_by_type(node, "identifier")
                    or find_child_by_type(node, "type_identifier")
                    or find_child_by_type(node, "name")
                )
                func_name = name_node.text.decode("utf-8") if name_node else ""
                # Walk function body
                for child in node.children:
                    walk(child, current_function=func_name)
                return

            for child in node.children:
                walk(child, current_function)

        walk(tree.root_node)
        return references
