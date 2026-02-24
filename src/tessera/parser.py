"""Tree-sitter-based source code parser for PHP, TypeScript, and Python.

Provides language detection, AST parsing, symbol extraction, and reference
resolution for building code dependency graphs.
"""

from dataclasses import dataclass, field
from typing import Optional
import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_php


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


# Language initialization
_LANGUAGES = {
    "python": Language(tree_sitter_python.language()),
    "typescript": Language(tree_sitter_javascript.language()),
    "php": Language(tree_sitter_php.language_php()),
}


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension.

    Args:
        file_path: File path or filename

    Returns:
        'php', 'typescript', 'python', or None if language cannot be detected
    """
    if file_path.endswith(".py"):
        return "python"
    elif file_path.endswith((".ts", ".tsx")):
        return "typescript"
    elif file_path.endswith((".js", ".jsx")):
        return "typescript"  # JS uses TypeScript grammar
    elif file_path.endswith(".php"):
        return "php"
    return None


def parse_file(source_code: str, language: str) -> tree_sitter.Tree:
    """Parse source code with tree-sitter and return the AST tree.

    Args:
        source_code: Source code string
        language: Language identifier ('python', 'typescript', 'php')

    Returns:
        tree_sitter.Tree: The parsed AST

    Raises:
        ValueError: If language is not supported
    """
    if language not in _LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    parser = Parser()
    parser.language = _LANGUAGES[language]
    tree = parser.parse(bytes(source_code, "utf-8"))
    return tree


def extract_symbols(source_code: str, language: str) -> list[Symbol]:
    """Extract all symbol definitions from source code.

    Args:
        source_code: Source code string
        language: Language identifier ('python', 'typescript', 'php')

    Returns:
        List of Symbol objects found in the source code
    """
    tree = parse_file(source_code, language)
    symbols = []

    if language == "python":
        symbols = _extract_symbols_python(tree, source_code)
    elif language == "typescript":
        symbols = _extract_symbols_typescript(tree, source_code)
    elif language == "php":
        symbols = _extract_symbols_php(tree, source_code)

    return symbols


def extract_references(
    source_code: str, language: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract references (calls, imports, extends) from source code.

    Args:
        source_code: Source code string
        language: Language identifier ('python', 'typescript', 'php')
        known_symbols: Optional list of known symbol names to match against

    Returns:
        List of Reference objects found in the source code
    """
    tree = parse_file(source_code, language)
    references = []

    if language == "python":
        references = _extract_references_python(tree, source_code, known_symbols)
    elif language == "typescript":
        references = _extract_references_typescript(tree, source_code, known_symbols)
    elif language == "php":
        references = _extract_references_php(tree, source_code, known_symbols)

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
            # Class declaration
            name_node = _find_child_by_type(node, "identifier")
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

        elif node.type == "method_definition":
            # Method in class
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

        elif node.type == "variable_declarator":
            # Arrow function or const function
            name_node = _find_child_by_type(node, "identifier")
            init = _find_child_by_type(node, "arrow_function")
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

        for child in node.children:
            walk(child, scope)

    walk(tree.root_node)
    return symbols


# Reference extraction helpers
def _extract_references_python(
    tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract Python references (calls, imports, extends)."""
    references = []

    def walk(node, current_function=""):
        # Handle function/method calls
        if node.type == "call":
            func_node = node.children[0] if node.children else None
            if func_node:
                func_name = None
                if func_node.type == "identifier":
                    func_name = func_node.text.decode("utf-8")
                elif func_node.type == "attribute":
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

        for child in node.children:
            walk(child, current_function)

    walk(tree.root_node)
    return references


def _extract_references_typescript(
    tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract TypeScript/JavaScript references."""
    references = []

    def walk(node, current_function=""):
        # Handle function calls
        if node.type == "call_expression":
            func_node = node.children[0] if node.children else None
            if func_node:
                func_name = None
                if func_node.type == "identifier":
                    func_name = func_node.text.decode("utf-8")
                elif func_node.type == "member_expression":
                    # obj.method() → extract "method" (property_identifier)
                    prop = _find_child_by_type(func_node, "property_identifier")
                    if prop:
                        func_name = prop.text.decode("utf-8")
                if func_name:
                    ref = Reference(
                        from_symbol=current_function or "<module>",
                        to_symbol=func_name,
                        kind="calls",
                        line=node.start_point[0] + 1,
                    )
                    references.append(ref)

        # Handle class inheritance (extends)
        elif node.type == "class_declaration":
            name_node = _find_child_by_type(node, "identifier")
            class_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
            # Look for class_heritage which contains extends/implements
            heritage_node = _find_child_by_type(node, "class_heritage")
            if heritage_node:
                for child in heritage_node.children:
                    if child.type == "identifier":
                        parent_name = child.text.decode("utf-8")
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
                            if impl_child.type == "identifier":
                                iface_name = impl_child.text.decode("utf-8")
                                ref = Reference(
                                    from_symbol=class_name,
                                    to_symbol=iface_name,
                                    kind="implements",
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

        for child in node.children:
            walk(child, current_function)

    walk(tree.root_node)
    return references


def _extract_references_php(
    tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] = None
) -> list[Reference]:
    """Extract PHP references (calls, imports, extends, WordPress hooks)."""
    references = []

    def walk(node, current_function=""):
        # Handle method calls ($obj->method())
        if node.type == "member_call_expression":
            # Extract method name from $obj->method()
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

        # Handle class declarations with inheritance
        elif node.type == "class_declaration":
            name_node = _find_child_by_type(node, "name")
            class_name = (
                name_node.text.decode("utf-8")
                if name_node
                else ""
            )
            # Look for base class
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

            # Walk class body
            for child in node.children:
                walk(child, current_function=class_name)
            return

        # Track current function for references
        elif node.type == "function_definition":
            name_node = _find_child_by_type(node, "name")
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
