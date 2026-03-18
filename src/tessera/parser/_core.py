"""Core parser functionality: grammar loading, language detection, and dispatch."""

import importlib

import tree_sitter
from tree_sitter import Language, Parser

from tessera.parser._patterns import Edge, Reference, Symbol
from tessera.parser.languages import (
    detect_language_from_ext,
    get_all_extractors,
    get_extractor,
)

# Keep imports for backward compatibility - code may import these directly
from tessera.parser._extractors import (
    _extract_symbols_generic,
    _extract_symbols_php,
    _extract_symbols_python,
    _extract_symbols_typescript,
)
from tessera.parser._references import (
    _extract_references_generic,
    _extract_references_php,
    _extract_references_python,
    _extract_references_typescript,
)

# Language initialization (lazy-loaded)
_grammar_cache: dict[str, Language] = {}

# Map of language name → (module_name, function_name)
# Most packages use language(), but some differ
_GRAMMAR_MODULES = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "php": ("tree_sitter_php", "language_php"),
    "go": ("tree_sitter_go", "language"),
    "swift": ("tree_sitter_swift", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
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
        ) from None

    func = getattr(mod, func_name, None)
    if func is None:
        raise ValueError(
            f"Grammar module '{module_name}' has no '{func_name}()' function"
        )

    lang = Language(func())
    _grammar_cache[language_name] = lang
    return lang


def detect_language(file_path: str) -> str | None:
    """Detect language from file extension.

    Args:
        file_path: File path or filename

    Returns:
        Language identifier ('python', 'typescript', 'javascript', 'php', 'go', 'rust',
        'java', 'csharp', 'ruby', 'swift', 'kotlin', 'c', 'cpp') or None if cannot be detected
    """
    # Try auto-discovery registry first
    lang = detect_language_from_ext(file_path)
    if lang:
        return lang

    # Fallback to hardcoded extension map for languages without extractors
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

    # Try plugin registry first for new language extractors
    all_extractors = get_all_extractors()
    if language in all_extractors:
        return all_extractors[language].extract_symbols(tree, source_code)

    # Fallback to old functions for backward compatibility
    if language == "python":
        return _extract_symbols_python(tree, source_code)
    elif language in ("typescript", "javascript"):
        return _extract_symbols_typescript(tree, source_code)
    elif language == "php":
        return _extract_symbols_php(tree, source_code)

    # Fallback to generic for anything else
    return _extract_symbols_generic(tree, source_code)


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

    # Try plugin registry first for new language extractors
    all_extractors = get_all_extractors()
    if language in all_extractors:
        return all_extractors[language].extract_references(tree, source_code, known_symbols)

    # Fallback to old functions for backward compatibility
    if language == "python":
        return _extract_references_python(tree, source_code, known_symbols)
    elif language in ("typescript", "javascript"):
        return _extract_references_typescript(tree, source_code, known_symbols)
    elif language == "php":
        return _extract_references_php(tree, source_code, known_symbols)

    # Fallback to generic for anything else
    return _extract_references_generic(tree, source_code, known_symbols)


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
        # or for special cases like hooks/events which may not be defined in the file
        if ref.to_symbol in symbol_names or ref.kind in (
            "hooks_into",
            "registers_on",
            "fires",
        ):
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
