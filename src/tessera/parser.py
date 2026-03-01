"""Tree-sitter-based source code parser for PHP, TypeScript, and Python.

Provides language detection, AST parsing, symbol extraction, and reference
resolution for building code dependency graphs.
"""

from dataclasses import dataclass, field
from typing import Optional
import importlib
import re
import tree_sitter
from tree_sitter import Language, Parser


@dataclass
class Symbol:
    """A symbol definition in source code (function, class, variable, etc.)."""

    name: str
    kind: str  # 'function', 'class', 'method', 'variable', 'constant', 'import'
    line: int
    col: int
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


# Language initialization (lazy-loaded)
_grammar_cache: dict[str, Language] = {}

# Map of language name → (module_name, function_name)
# Most packages use language(), but some differ
_GRAMMAR_MODULES = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "php": ("tree_sitter_php", "language_php"),
}


def _load_language(language_name: str) -> Language:
    """Lazy-load a tree-sitter grammar. Caches after first load.

    For languages not in _GRAMMAR_MODULES, tries the convention:
    tree_sitter_{name}.language()

    Args:
        language_name: Language identifier (e.g., 'python', 'typescript')

    Returns:
        Language object for tree-sitter parsing

    Raises:
        ValueError: If grammar module cannot be imported or function not found
    """
    if language_name in _grammar_cache:
        return _grammar_cache[language_name]

    if language_name in _GRAMMAR_MODULES:
        module_name, func_name = _GRAMMAR_MODULES[language_name]
    else:
        # Convention: tree_sitter_{name}.language()
        module_name = f"tree_sitter_{language_name}"
        func_name = "language"

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        pip_name = module_name.replace("_", "-")
        raise ValueError(
            f"Grammar for '{language_name}' not installed. "
            f"Install with: pip install {pip_name}"
        )

    func = getattr(mod, func_name, None)
    if func is None:
        raise ValueError(
            f"Grammar module '{module_name}' has no '{func_name}()' function"
        )

    lang = Language(func())
    _grammar_cache[language_name] = lang
    return lang


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension.

    Args:
        file_path: File path or filename

    Returns:
        Language identifier ('python', 'typescript', 'javascript', 'php', 'go', 'rust',
        'java', 'csharp', 'ruby', 'swift', 'kotlin', 'c', 'cpp') or None if cannot be detected
    """
    if file_path.endswith(".py"):
        return "python"
    elif file_path.endswith((".ts", ".tsx")):
        return "typescript"
    elif file_path.endswith((".js", ".jsx")):
        return "javascript"
    elif file_path.endswith(".php"):
        return "php"
    elif file_path.endswith(".go"):
        return "go"
    elif file_path.endswith(".rs"):
        return "rust"
    elif file_path.endswith(".java"):
        return "java"
    elif file_path.endswith(".cs"):
        return "csharp"
    elif file_path.endswith(".rb"):
        return "ruby"
    elif file_path.endswith(".swift"):
        return "swift"
    elif file_path.endswith((".kt", ".kts")):
        return "kotlin"
    elif file_path.endswith((".c", ".h")):
        return "c"
    elif file_path.endswith((".cpp", ".cc", ".cxx", ".hpp")):
        return "cpp"
    return None


def parse_file(source_code: str, language: str) -> tree_sitter.Tree:
    """Parse source code with tree-sitter and return the AST tree.

    Args:
        source_code: Source code string
        language: Language identifier ('python', 'typescript', 'php', etc.)

    Returns:
        tree_sitter.Tree: The parsed AST

    Raises:
        ValueError: If language grammar cannot be loaded
    """
    lang_obj = _load_language(language)
    parser = Parser()
    parser.language = lang_obj
    tree = parser.parse(bytes(source_code, "utf-8"))
    return tree


def extract_symbols(source_code: str, language: str) -> list[Symbol]:
    """Extract all symbol definitions from source code.

    Args:
        source_code: Source code string
        language: Language identifier ('python', 'typescript', 'php', etc.)

    Returns:
        List of Symbol objects found in the source code
    """
    tree = parse_file(source_code, language)
    symbols = []

    if language == "python":
        symbols = _extract_symbols_python(tree, source_code)
    elif language in ("typescript", "javascript"):
        symbols = _extract_symbols_typescript(tree, source_code)
    elif language == "php":
        symbols = _extract_symbols_php(tree, source_code)
    else:
        # Fallback to generic extractor for unsupported languages
        symbols = _extract_symbols_generic(tree, source_code)

    return symbols


def extract_references(
    source_code: str, language: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract references (calls, imports, extends) from source code.

    Args:
        source_code: Source code string
        language: Language identifier ('python', 'typescript', 'php', etc.)
        known_symbols: Optional list of known symbol names to match against

    Returns:
        List of Reference objects found in the source code
    """
    tree = parse_file(source_code, language)
    references = []

    if language == "python":
        references = _extract_references_python(tree, source_code, known_symbols)
    elif language in ("typescript", "javascript"):
        references = _extract_references_typescript(tree, source_code, known_symbols)
    elif language == "php":
        references = _extract_references_php(tree, source_code, known_symbols)
    else:
        # Fallback to generic extractor for unsupported languages
        references = _extract_references_generic(tree, source_code, known_symbols)

    return references


def build_edges(symbols: list[Symbol], references: list[Reference]) -> list[Edge]:
    """Build graph edges from symbols and references.

    Args:
        symbols: List of extracted symbols
        references: List of extracted references

    Returns:
        List of Edge objects representing the dependency graph
    """
    edges = []
    seen = set()  # (from_name, to_name, type) for deduplication
    symbol_names = {s.name for s in symbols}

    for ref in references:
        # Only create edges for references that resolve to known symbols
        # or for special cases like hooks which may not be defined in the file
        if ref.to_symbol in symbol_names or ref.kind == "hooks_into":
            key = (ref.from_symbol, ref.to_symbol, ref.kind)
            if key not in seen:
                seen.add(key)
                edges.append(Edge(
                    from_name=ref.from_symbol, to_name=ref.to_symbol, type=ref.kind
                ))

    # Containment edges from scope relationships (parent → child)
    for sym in symbols:
        if sym.scope and sym.scope in symbol_names:
            key = (sym.scope, sym.name, "contains")
            if key not in seen:
                seen.add(key)
                edges.append(Edge(
                    from_name=sym.scope, to_name=sym.name, type="contains"
                ))

    return edges


def parse_and_extract(
    file_path: str, source_code: str
) -> tuple[list[Symbol], list[Reference], list[Edge]]:
    """Convenience function: detect language, extract everything.

    Args:
        file_path: File path (used to detect language)
        source_code: Source code string

    Returns:
        Tuple of (symbols, references, edges)

    Raises:
        ValueError: If language cannot be detected from file_path
    """
    language = detect_language(file_path)
    if language is None:
        raise ValueError(
            f"Cannot detect language for file: {file_path}. "
            f"Supported extensions: .py, .ts, .tsx, .js, .jsx, .php"
        )

    symbols = extract_symbols(source_code, language)
    references = extract_references(source_code, language)
    edges = build_edges(symbols, references)

    return symbols, references, edges


# Python extraction helpers
def _extract_symbols_python(tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
    """Extract Python symbols."""
    symbols = []
    current_class = None

    def walk(node, scope=""):
        nonlocal current_class

        if node.type in ("function_definition", "async_function_definition"):
            # Extract function/method (sync or async)
            name_node = _find_child_by_type(node, "identifier")
            if name_node:
                name = name_node.text.decode("utf-8")
                # Build signature
                params_node = _find_child_by_type(node, "parameters")
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
            name_node = _find_child_by_type(node, "identifier")
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
            name_node = _find_child_by_type(node, "identifier")
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
                _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "identifier")
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
            name_node = _find_child_by_type(node, "type_identifier")
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
            name_node = _find_child_by_type(node, "identifier")
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
            name_node = _find_child_by_type(node, "type_identifier")
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
            name_node = _find_child_by_type(node, "property_identifier")
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
            name_node = _find_child_by_type(node, "property_identifier")
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
            name_node = _find_child_by_type(node, "identifier")
            init = (
                _find_child_by_type(node, "arrow_function")
                or _find_child_by_type(node, "function_expression")
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
            key_node = _find_child_by_type(node, "property_identifier")
            val = (
                _find_child_by_type(node, "function_expression")
                or _find_child_by_type(node, "arrow_function")
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
                _find_child_by_type(node, "function_expression")
                or _find_child_by_type(node, "arrow_function")
            )
            if right:
                left = node.children[0] if node.children else None
                name = None
                if left and left.type == "member_expression":
                    # Extract rightmost property: window.PUM.foo -> "foo"
                    prop = _find_child_by_type(left, "property_identifier")
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
            name_node = _find_child_by_type(node, "namespace_name")
            if name_node:
                current_namespace = name_node.text.decode("utf-8")
            # Walk namespace body (braced style)
            for child in node.children:
                walk(child, scope)
            return

        if node.type == "function_definition":
            # Function definition
            name_node = _find_child_by_type(node, "name")
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
            name_node = _find_child_by_type(node, "name")
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
            name_node = _find_child_by_type(node, "name")
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
            name_node = _find_child_by_type(node, "name")
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
            name_node = _find_child_by_type(node, "name")
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


# Call chain extraction helper
def _extract_call_chain(node: tree_sitter.Node, language: str) -> list[str]:
    """Extract all method names in a call chain.

    For a.b().c().d(), returns ['d', 'c', 'b'] (outermost to innermost).
    For a simple obj.method(), returns ['method'].
    For a direct func(), returns [].

    Args:
        node: A call/call_expression node that may be part of a chain
        language: Language identifier ('python', 'typescript', 'php')

    Returns:
        List of method names in the chain (outermost to innermost)
    """
    chain = []

    if language == "python":
        # Python: call → attribute → identifier (method name)
        # Attribute node has structure: [object, ".", identifier_or_attribute]
        # Check if this call's func_node is an attribute
        func_node = node.children[0] if node.children else None
        if func_node and func_node.type == "attribute":
            # Extract the RIGHTMOST identifier from attribute (it's the method name)
            # Attribute children are typically: [object, ".", method_name]
            # So we look for the last identifier
            last_identifier = None
            for child in func_node.children:
                if child.type == "identifier":
                    last_identifier = child.text.decode("utf-8")

            if last_identifier:
                chain.append(last_identifier)

            # Check if attribute's object is itself a call (chain detection)
            obj_node = func_node.children[0] if func_node.children else None
            if obj_node and obj_node.type == "call":
                # Recurse into the inner call
                chain.extend(_extract_call_chain(obj_node, language))

    elif language in ("typescript", "javascript"):
        # TypeScript/JS: call_expression → member_expression → property_identifier
        func_node = node.children[0] if node.children else None
        if func_node and func_node.type == "member_expression":
            # Extract property_identifier
            prop = _find_child_by_type(func_node, "property_identifier")
            if prop:
                chain.append(prop.text.decode("utf-8"))

            # Check if member_expression's object is itself a call_expression
            obj_node = func_node.children[0] if func_node.children else None
            if obj_node and obj_node.type == "call_expression":
                # Recurse into the inner call
                chain.extend(_extract_call_chain(obj_node, language))

    elif language == "php":
        # PHP: member_call_expression → may have nested member_call_expression
        # Extract method name from this level
        method_node = _find_child_by_type(node, "name")
        if method_node:
            chain.append(method_node.text.decode("utf-8"))

        # Check if this is part of a larger chain
        # In PHP, member_call_expression structure can have a chained call
        # Look for a child that's another member_call_expression or variable
        for child in node.children:
            if child.type == "member_call_expression":
                # Recurse
                chain.extend(_extract_call_chain(child, language))

    return chain




# Generic fallback extractors for unsupported languages
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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


def _extract_references_generic(
    tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract references using common AST node types.

    Looks for:
    - call_expression, identifier/member_expression → kind="calls"
    - new_expression, object_creation_expression → kind="calls"
    - Class nodes with superclass, base_clause, heritage children → kind="extends"

    Args:
        tree: Parsed AST tree
        source_code: Source code string
        known_symbols: Optional list of known symbol names to match against

    Returns:
        List of Reference objects found in the source code
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
            )
            class_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
            )
            func_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
            # Walk function body
            for child in node.children:
                walk(child, current_function=func_name)
            return

        for child in node.children:
            walk(child, current_function)

    walk(tree.root_node)
    return references


# Generic fallback extractors for unsupported languages
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
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


def _extract_references_generic(
    tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract references using common AST node types.

    Looks for:
    - call_expression, identifier/member_expression → kind="calls"
    - new_expression, object_creation_expression → kind="calls"
    - Class nodes with superclass, base_clause, heritage children → kind="extends"

    Args:
        tree: Parsed AST tree
        source_code: Source code string
        known_symbols: Optional list of known symbol names to match against

    Returns:
        List of Reference objects found in the source code
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
            )
            class_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
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
                _find_child_by_type(node, "identifier")
                or _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "name")
            )
            func_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
            # Walk function body
            for child in node.children:
                walk(child, current_function=func_name)
            return

        for child in node.children:
            walk(child, current_function)

    walk(tree.root_node)
    return references


# Reference extraction helpers
def _extract_references_python(
    tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract Python references (calls, imports, extends)."""
    references = []
    chain_nodes = set()  # Track nodes already processed as part of a chain

    def walk(node, current_function=""):
        # Skip nodes already processed as part of a chain
        if id(node) in chain_nodes:
            for child in node.children:
                walk(child, current_function)
            return

        # Handle function/method calls
        if node.type == "call":
            func_node = node.children[0] if node.children else None
            if func_node:
                func_name = None
                if func_node.type == "identifier":
                    func_name = func_node.text.decode("utf-8")
                elif func_node.type == "attribute":
                    # obj.method() or obj.method().chain()
                    # Try to extract the full chain
                    chain_names = _extract_call_chain(node, "python")
                    if chain_names:
                        # Mark ONLY inner call nodes that have attribute functions as processed
                        # (to avoid double-counting intermediate chained calls, but NOT simple function calls)
                        obj = func_node.children[0] if func_node.children else None
                        current = obj
                        while current and current.type == "call":
                            # Only mark this call if its function is an attribute (part of chain)
                            inner_func = current.children[0] if current.children else None
                            if inner_func and inner_func.type == "attribute":
                                chain_nodes.add(id(current))
                                obj = inner_func.children[0] if inner_func.children else None
                                current = obj
                            else:
                                # Hit a simple function call, don't mark it - let it be processed normally
                                break

                        # Create references for all methods in the chain
                        for method_name in chain_names:
                            ref = Reference(
                                from_symbol=current_function or "<module>",
                                to_symbol=method_name,
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

        # Handle class inheritance
        elif node.type == "class_definition":
            name_node = _find_child_by_type(node, "identifier")
            class_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
            # Look for inheritance in arguments
            args_node = _find_child_by_type(node, "argument_list")
            if args_node:
                for child in args_node.children:
                    if child.type == "identifier":
                        parent_name = child.text.decode("utf-8")
                        ref = Reference(
                            from_symbol=class_name,
                            to_symbol=parent_name,
                            kind="extends",
                            line=node.start_point[0] + 1,
                        )
                        references.append(ref)

            # Walk class body to find calls within
            for child in node.children:
                if child.type == "block":
                    for subchild in child.children:
                        walk(subchild, current_function=class_name)
            return

        # Track current function for references
        elif node.type in ("function_definition", "async_function_definition"):
            name_node = _find_child_by_type(node, "identifier")
            func_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
            # Walk function body
            for child in node.children:
                walk(child, current_function=func_name)
            return

        elif node.type == "decorated_definition":
            # Unwrap: walk the inner function/class definition directly
            for child in node.children:
                if child.type in ("function_definition", "async_function_definition", "class_definition"):
                    walk(child, current_function)
            return

        for child in node.children:
            walk(child, current_function)

    walk(tree.root_node)
    return references


# JSDoc/PHPDoc type primitives to skip
_DOCBLOCK_PRIMITIVES = frozenset({
    "string", "number", "boolean", "undefined", "null", "void", "never",
    "any", "unknown", "object", "symbol", "bigint",  # JS/TS
    "int", "integer", "float", "double", "bool", "array", "callable",
    "iterable", "mixed", "self", "static", "parent", "true", "false",
    "resource",  # PHP
})

# JSDoc: @param {Type} name, @returns {Type}, @type {Type}, @typedef {Type}
_JSDOC_TYPE_RE = re.compile(
    r"@(?:param|returns?|type|typedef|throws?|callback)\s+\{([^}]+)\}"
)

# PHPDoc: @param Type $name, @return Type, @var Type, @throws Type
# Handles: Type, ?Type, \Ns\Type, Type|Other, Type[]
_PHPDOC_TYPE_RE = re.compile(
    r"@(?:param|return|returns|var|throws|property(?:-read|-write)?)\s+"
    r"((?:\??\\?[\w\\]+(?:\[\])*(?:\|\??\\?[\w\\]+(?:\[\])*)*))"
)


def _extract_docblock_types(comment_text: str, style: str) -> list[str]:
    """Extract type names from a doc comment string.

    Args:
        comment_text: The raw comment text (including delimiters)
        style: 'jsdoc' for {Type} syntax, 'phpdoc' for Type $var syntax

    Returns:
        List of individual type name strings (primitives excluded)
    """
    types = []
    if style == "jsdoc":
        for match in _JSDOC_TYPE_RE.finditer(comment_text):
            type_expr = match.group(1)
            # Strip nullable prefix
            type_expr = type_expr.lstrip("?!")
            # Split on union/intersection
            for part in re.split(r"[|&]", type_expr):
                part = part.strip()
                if not part:
                    continue
                # Handle generic: Array<Item> → extract Array and Item
                generic_match = re.match(r"([\w.]+)\s*<(.+)>", part)
                if generic_match:
                    base = generic_match.group(1)
                    if base.lower() not in _DOCBLOCK_PRIMITIVES:
                        types.append(base)
                    inner = generic_match.group(2)
                    for inner_part in re.split(r"[,|&]", inner):
                        inner_part = inner_part.strip()
                        if inner_part and inner_part.lower() not in _DOCBLOCK_PRIMITIVES:
                            types.append(inner_part)
                else:
                    # Strip array suffix
                    part = part.rstrip("[]")
                    if part and part.lower() not in _DOCBLOCK_PRIMITIVES:
                        types.append(part)
    elif style == "phpdoc":
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


def _collect_type_references(
    node,
    from_symbol: str,
    references: list,
    declaration_name: str = None,
) -> None:
    """Recursively collect type_identifier nodes from a type-context subtree.

    Args:
        node: The type-context root node to walk
        from_symbol: The enclosing symbol (function/class name or "<module>")
        references: List to append Reference objects to
        declaration_name: Name being declared (to exclude self-references)
    """
    if node.type == "type_identifier":
        text = node.text.decode("utf-8")
        # Skip declaration name (avoid self-reference in type aliases)
        if text == declaration_name:
            return
        # Skip single uppercase letters (likely generic type parameters)
        if len(text) == 1 and text.isupper():
            return
        references.append(Reference(
            from_symbol=from_symbol,
            to_symbol=text,
            kind="type_reference",
            line=node.start_point[0] + 1,
        ))
        return

    if node.type == "nested_type_identifier":
        # Qualified name like Namespace.Foo — extract full name
        parts = []
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                parts.append(child.text.decode("utf-8"))
        if parts:
            full_name = ".".join(parts)
            if full_name != declaration_name:
                references.append(Reference(
                    from_symbol=from_symbol,
                    to_symbol=full_name,
                    kind="type_reference",
                    line=node.start_point[0] + 1,
                ))
        return  # Don't recurse — already handled

    # Recurse into children
    for child in node.children:
        _collect_type_references(child, from_symbol, references, declaration_name)


def _extract_references_typescript(
    tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract TypeScript/JavaScript references."""
    references = []
    chain_nodes = set()  # Track nodes already processed as part of a chain

    def walk(node, current_function=""):
        # Skip nodes already processed as part of a chain
        if id(node) in chain_nodes:
            for child in node.children:
                walk(child, current_function)
            return

        # Handle function calls
        if node.type == "call_expression":
            func_node = node.children[0] if node.children else None
            if func_node:
                func_name = None
                if func_node.type == "identifier":
                    func_name = func_node.text.decode("utf-8")
                elif func_node.type == "member_expression":
                    # obj.method() or obj.method().chain()
                    # Try to extract the full chain
                    chain_names = _extract_call_chain(node, "typescript")
                    if chain_names:
                        # Mark ONLY inner call nodes that have member_expression functions as processed
                        # (to avoid double-counting intermediate chained calls, but NOT simple function calls)
                        obj = func_node.children[0] if func_node.children else None
                        current = obj
                        while current and current.type == "call_expression":
                            # Only mark this call if its function is a member_expression (part of chain)
                            inner_func = current.children[0] if current.children else None
                            if inner_func and inner_func.type == "member_expression":
                                chain_nodes.add(id(current))
                                obj = inner_func.children[0] if inner_func.children else None
                                current = obj
                            else:
                                # Hit a simple function call, don't mark it - let it be processed normally
                                break

                        # Create references for all methods in the chain
                        for method_name in chain_names:
                            ref = Reference(
                                from_symbol=current_function or "<module>",
                                to_symbol=method_name,
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
                _find_child_by_type(node, "type_identifier")
                or _find_child_by_type(node, "identifier")
            )
            class_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
            # Look for class_heritage which contains extends/implements
            heritage_node = _find_child_by_type(node, "class_heritage")
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
            name_node = _find_child_by_type(node, "identifier")
            if not name_node:
                name_node = _find_child_by_type(node, "property_identifier")
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
                if child.type == "constraint":
                    _collect_type_references(child, current_function or "<module>", references)
                elif child.type == "default_type":
                    _collect_type_references(child, current_function or "<module>", references)
            return

        for child in node.children:
            walk(child, current_function)

    walk(tree.root_node)

    # Second pass: extract types from JSDoc comments
    _extract_comments_pass(tree.root_node, references, "jsdoc")

    return references


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
        "class_declaration", "interface_declaration", "type_alias_declaration",
        "lexical_declaration", "variable_declaration", "export_statement",
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
            elif child.type in ("declaration_list", "class_body", "statement_block"):
                _walk_for_comments(child, current_function)
            elif child.type == "program":
                _walk_for_comments(child, current_function)

    _walk_for_comments(root_node)


def _get_declaration_name(node) -> Optional[str]:
    """Extract the name from a declaration node."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return child.text.decode("utf-8")
        # TS export: export function foo() {}
        if child.type in (
            "function_declaration", "class_declaration",
            "interface_declaration", "type_alias_declaration",
            "lexical_declaration",
        ):
            return _get_declaration_name(child)
    return None


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


def _extract_references_php(
    tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract PHP references (calls, imports, extends, WordPress hooks)."""
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
            chain_names = _extract_call_chain(node, "php")
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
                method_node = _find_child_by_type(node, "name")
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

                # Check for WordPress hooks (add_action, add_filter, do_action, apply_filters)
                if func_name in (
                    "add_action",
                    "add_filter",
                    "do_action",
                    "apply_filters",
                ):
                    # Extract the first string argument as the hook name
                    args_node = _find_child_by_type(node, "arguments")
                    if args_node:
                        hook_name = _extract_first_string_arg(args_node, source_code)
                        if hook_name:
                            ref = Reference(
                                from_symbol=current_function or "<module>",
                                to_symbol=hook_name,
                                kind="hooks_into",
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
            name_node = _find_child_by_type(node, "name")
            class_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
            for child in node.children:
                if child.type == "base_clause":
                    parent_node = _find_child_by_type(child, "name")
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
                        if iface_child.type == "name":
                            references.append(Reference(
                                from_symbol=class_name,
                                to_symbol=iface_child.text.decode("utf-8"),
                                kind="implements",
                                line=iface_child.start_point[0] + 1,
                            ))
                        elif iface_child.type == "qualified_name":
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
            name_node = _find_child_by_type(node, "name")
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
            name_node = _find_child_by_type(node, "name")
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
            name_node = _find_child_by_type(node, "name")
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


# Helper functions
def _find_child_by_type(
    node: tree_sitter.Node, node_type: str
) -> Optional[tree_sitter.Node]:
    """Find the first child node of a given type."""
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def _extract_first_string_arg(
    args_node: tree_sitter.Node, source_code: str
) -> Optional[str]:
    """Extract the first string argument from a function call arguments node."""
    for child in args_node.children:
        if child.type == "argument":
            # In PHP, the argument may contain a string or encapsed_string
            for subchild in child.children:
                if subchild.type in ("string", "encapsed_string"):
                    # Look for string_content inside
                    for content_child in subchild.children:
                        if content_child.type == "string_content":
                            return content_child.text.decode("utf-8")
                    # If no string_content, try to get the whole text and strip quotes
                    text = subchild.text.decode("utf-8")
                    text = text.strip("'\"")
                    return text
        elif child.type in ("string", "encapsed_string", "heredoc", "string_literal"):
            # Direct string child (for older or simpler cases)
            # Look for string_content inside
            for content_child in child.children:
                if content_child.type == "string_content":
                    return content_child.text.decode("utf-8")
            # If no string_content, try to get the whole text and strip quotes
            text = child.text.decode("utf-8")
            text = text.strip("'\"")
            return text
    return None


if __name__ == "__main__":
    # Test parsing with small snippets
    print("Testing PHP parsing:")
    php_code = """<?php
function greet($name) {
    return "Hello " . $name;
}

class Calculator {
    public function add($a, $b) {
        return $a + $b;
    }
}

add_action('wp_loaded', 'greet');
"""
    symbols, refs, edges = parse_and_extract("test.php", php_code)
    print(f"  Symbols: {len(symbols)}")
    for sym in symbols:
        print(f"    - {sym.name} ({sym.kind})")
    print(f"  References: {len(refs)}")
    for ref in refs:
        print(f"    - {ref.from_symbol} -> {ref.to_symbol} ({ref.kind})")

    print("\nTesting TypeScript parsing:")
    ts_code = """
function process(data: any) {
    return transform(data);
}

function transform(d: any) {
    return d;
}

class Animal {
    name: string;
}

class Dog extends Animal {
    bark() {}
}
"""
    symbols, refs, edges = parse_and_extract("test.ts", ts_code)
    print(f"  Symbols: {len(symbols)}")
    for sym in symbols:
        print(f"    - {sym.name} ({sym.kind})")
    print(f"  References: {len(refs)}")
    for ref in refs:
        print(f"    - {ref.from_symbol} -> {ref.to_symbol} ({ref.kind})")

    print("\nTesting Python parsing:")
    py_code = """
def helper():
    pass

def main():
    helper()

class Base:
    pass

class Derived(Base):
    def process(self):
        helper()
"""
    symbols, refs, edges = parse_and_extract("test.py", py_code)
    print(f"  Symbols: {len(symbols)}")
    for sym in symbols:
        print(f"    - {sym.name} ({sym.kind})" + (f" in {sym.scope}" if sym.scope else ""))
    print(f"  References: {len(refs)}")
    for ref in refs:
        print(f"    - {ref.from_symbol} -> {ref.to_symbol} ({ref.kind})")
