"""Language-specific symbol extractors."""

import tree_sitter

from tessera.parser._patterns import Symbol, find_child_by_type


def _extract_symbols_python(tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
    """Extract Python symbols."""
    symbols = []
    current_class = None

    def walk(node, scope=""):
        nonlocal current_class

        if node.type in ("function_definition", "async_function_definition"):
            # Extract function/method (sync or async)
            name_node = find_child_by_type(node, "identifier")
            if name_node:
                name = name_node.text.decode("utf-8")
                # Build signature
                params_node = find_child_by_type(node, "parameters")
                sig = f"{name}(" + (
                    params_node.text.decode("utf-8")[1:-1]
                    if params_node
                    else ""
                ) + ")"

                kind = "method" if current_class else "function"
                sym = Symbol(
                    name=name,
                    kind=kind,
                    line=node.start_point[0] + 1,
                    col=node.start_point[1],
                    scope=scope,
                    signature=sig,
                )
                symbols.append(sym)
                # Walk function body with function as scope for nested defs
                for child in node.children:
                    walk(child, scope=name)
                return

        elif node.type == "class_definition":
            # Extract class
            name_node = find_child_by_type(node, "identifier")
            if name_node:
                name = name_node.text.decode("utf-8")
                sym = Symbol(
                    name=name,
                    kind="class",
                    line=node.start_point[0] + 1,
                    col=node.start_point[1],
                )
                symbols.append(sym)
                # Walk class body with updated scope
                old_class = current_class
                current_class = name
                for child in node.children:
                    walk(child, scope=name)
                current_class = old_class
                return

        elif node.type in ("import_statement", "import_from_statement"):
            # Extract imports
            sym = Symbol(
                name="",
                kind="import",
                line=node.start_point[0] + 1,
                col=node.start_point[1],
            )
            symbols.append(sym)

        elif node.type == "decorated_definition":
            # Unwrap: walk the inner function/class definition directly
            for child in node.children:
                if child.type in ("function_definition", "async_function_definition", "class_definition"):
                    walk(child, scope)
            return

        for child in node.children:
            walk(child, scope)

    walk(tree.root_node)
    return symbols



def _extract_symbols_typescript(tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
    """Extract TypeScript/JavaScript symbols."""
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                        scope=scope,
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
                )
                symbols.append(sym)

        for child in node.children:
            walk(child, scope)

    walk(tree.root_node)
    return symbols




def _extract_symbols_php(tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
    """Extract PHP symbols."""
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
                )
                symbols.append(sym)
                for child in node.children:
                    walk(child, scope=qualified)
                return

        for child in node.children:
            walk(child, scope)

    walk(tree.root_node)
    return symbols




def _extract_symbols_generic(tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
    """Extract symbols using common AST node types that appear across C-family languages.

    Looks for: function_definition, function_declaration, function_item, class_definition,
    class_declaration, method_definition, method_declaration, interface_declaration,
    enum_declaration, enum_item, trait_item, import_statement, import_declaration,
    use_declaration, struct_item, struct_declaration.

    Args:
        tree: Parsed AST tree
        source_code: Source code string

    Returns:
        List of Symbol objects found in the source code
    """
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
                    scope=scope,
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
            )
            symbols.append(sym)

        for child in node.children:
            walk(child, scope)

    walk(tree.root_node)
    return symbols


