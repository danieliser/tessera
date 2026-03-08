"""Abstract base class for language extractors."""

from abc import ABC, abstractmethod

import tree_sitter

from tessera.parser._patterns import Reference, Symbol


class LanguageExtractor(ABC):
    """Base class for per-language symbol and reference extractors.

    To add a new language, create a module in parser/languages/ with a class
    that subclasses LanguageExtractor and implements extract_symbols() and
    extract_references(). The module is auto-discovered at import time.

    Event patterns (hooks, signals, pub/sub) are declared via class-level dicts:

        # Functions that REGISTER a listener on an event (first string arg = event name)
        EVENT_REGISTERS: dict[str, str]  # func_name → edge kind
        # For WordPress hooks, use "hooks_into" for backward compatibility
        # For new systems, use "registers_on"

        # Functions that FIRE/EMIT an event (first string arg = event name)
        EVENT_FIRES: dict[str, str]  # func_name → edge kind
        # For WordPress hooks, use "hooks_into" for backward compatibility
        # For new systems, use "fires"

    The base class provides extract_events() which uses these dicts for automatic
    event edge extraction. Override extract_events() for languages where event
    patterns need custom AST handling beyond simple function-name + first-string-arg.
    """

    # --- Subclass must define these ---
    language: str                      # e.g., "python"
    extensions: tuple[str, ...]        # e.g., (".py",)
    grammar_module: str                # e.g., "tree_sitter_python"
    grammar_func: str = "language"     # e.g., "language" or "language_php"

    # --- Event pattern registry (override in subclass) ---
    EVENT_REGISTERS: dict[str, str] = {}
    EVENT_FIRES: dict[str, str] = {}

    @abstractmethod
    def extract_symbols(self, tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
        """Extract symbol definitions from parsed AST."""
        ...

    @abstractmethod
    def extract_references(
        self, tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] | None = None
    ) -> list[Reference]:
        """Extract references (calls, imports, extends, events) from parsed AST.

        Implementations MUST call self.extract_events() and include its results,
        or handle event extraction inline using self.EVENT_REGISTERS / self.EVENT_FIRES.
        """
        ...

    def extract_events(self, tree: tree_sitter.Tree, source_code: str) -> list[Reference]:
        """Default event extraction using EVENT_REGISTERS and EVENT_FIRES dicts.

        Walks the AST looking for function calls where the function name matches
        a key in EVENT_REGISTERS or EVENT_FIRES. Extracts the first string argument
        as the event/hook name.

        Override this for languages where event patterns need custom handling.
        """
        if not self.EVENT_REGISTERS and not self.EVENT_FIRES:
            return []

        from tessera.parser._helpers import extract_first_string_arg_generic

        all_event_funcs = {**self.EVENT_REGISTERS, **self.EVENT_FIRES}
        references = []

        self._walk_for_events(tree.root_node, all_event_funcs, references, source_code)
        return references

    def _walk_for_events(
        self,
        node: tree_sitter.Node,
        event_funcs: dict[str, str],
        references: list[Reference],
        source_code: str,
        current_function: str = "",
    ) -> None:
        """Walk AST and extract event references. Override for custom call node types."""
        # Default handles common call node types
        call_types = self._get_call_node_types()

        if node.type in call_types:
            func_name = self._get_func_name_from_call(node)
            if func_name and func_name in event_funcs:
                from tessera.parser._helpers import extract_first_string_arg_generic

                event_name = extract_first_string_arg_generic(node, source_code, self.language)
                if event_name:
                    references.append(
                        Reference(
                            from_symbol=current_function or "<module>",
                            to_symbol=event_name,
                            kind=event_funcs[func_name],
                            line=node.start_point[0] + 1,
                        )
                    )

        # Track current function scope
        new_scope = self._get_function_scope(node) or current_function

        for child in node.children:
            self._walk_for_events(child, event_funcs, references, source_code, new_scope)

    def _get_call_node_types(self) -> set[str]:
        """AST node types that represent function/method calls. Override per language."""
        return {
            "call_expression",
            "call",
            "function_call_expression",
            "member_call_expression",
        }

    def _get_func_name_from_call(self, node: tree_sitter.Node) -> str | None:
        """Extract function name from a call node. Override for language-specific AST shapes."""
        from tessera.parser._patterns import find_child_by_type

        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
            if child.type == "name":
                return child.text.decode("utf-8")
            if child.type == "member_expression":
                prop = find_child_by_type(child, "property_identifier")
                if prop:
                    return prop.text.decode("utf-8")
            if child.type == "attribute":
                # Python: obj.method -> last identifier
                last_id = None
                for sub in child.children:
                    if sub.type == "identifier":
                        last_id = sub.text.decode("utf-8")
                return last_id
        return None

    def _get_function_scope(self, node: tree_sitter.Node) -> str | None:
        """If node is a function/method definition, return its name. Else None."""
        func_types = {
            "function_definition",
            "function_declaration",
            "function_item",
            "method_definition",
            "method_declaration",
            "async_function_definition",
        }
        if node.type not in func_types:
            return None
        from tessera.parser._patterns import find_child_by_type

        name_node = (
            find_child_by_type(node, "identifier")
            or find_child_by_type(node, "name")
            or find_child_by_type(node, "property_identifier")
        )
        return name_node.text.decode("utf-8") if name_node else None
