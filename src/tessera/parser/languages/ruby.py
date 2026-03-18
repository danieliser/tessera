"""Ruby language extractor for symbols, references, and graph edges."""

import tree_sitter

from tessera.parser._base import LanguageExtractor
from tessera.parser._helpers import node_signature
from tessera.parser._patterns import Reference, Symbol, find_child_by_type


class RubyExtractor(LanguageExtractor):
    """Extract Ruby symbols and references.

    Detects:
    - Symbols: methods (function at top-level, method in class/module),
      classes, modules, constants, require/require_relative (imports)
    - References: calls, inheritance (extends), include/extend/prepend (implements),
      constant references
    - Graph edges: containment via class/module scope, call/extends/implements edges

    Ruby-specific:
    - `module` is treated as kind=class (modules are first-class in Ruby)
    - `include`/`extend`/`prepend` are calls that map to implements references
    - `require`/`require_relative` produce import symbols, not references
    """

    language = "ruby"
    extensions = (".rb",)
    grammar_module = "tree_sitter_ruby"
    grammar_func = "language"

    # Ruby has no standard event/hook system
    EVENT_REGISTERS: dict[str, str] = {}
    EVENT_FIRES: dict[str, str] = {}

    # Mixin methods that map to implements references
    _MIXIN_METHODS = frozenset({"include", "extend", "prepend"})

    def extract_symbols(self, tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
        """Extract Ruby symbol definitions."""
        symbols = []

        def walk(node, scope=""):
            if node.type == "method":
                name_node = find_child_by_type(node, "identifier")
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

            elif node.type == "singleton_method":
                # self.method_name
                name_node = find_child_by_type(node, "identifier")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    symbols.append(Symbol(
                        name=name,
                        kind="method",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    ))
                    for child in node.children:
                        walk(child, scope=name)
                    return

            elif node.type == "class":
                name_node = find_child_by_type(node, "constant")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    qualified = f"{scope}::{name}" if scope else name
                    symbols.append(Symbol(
                        name=qualified,
                        kind="class",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    ))
                    for child in node.children:
                        walk(child, scope=qualified)
                    return

            elif node.type == "module":
                name_node = find_child_by_type(node, "constant")
                if name_node:
                    name = name_node.text.decode("utf-8")
                    qualified = f"{scope}::{name}" if scope else name
                    symbols.append(Symbol(
                        name=qualified,
                        kind="class",  # Module as class
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        signature=node_signature(node),
                    ))
                    for child in node.children:
                        walk(child, scope=qualified)
                    return

            elif node.type == "call":
                # require/require_relative → import symbol
                func_name = _get_call_name(node)
                if func_name in ("require", "require_relative"):
                    symbols.append(Symbol(
                        name="",
                        kind="import",
                        line=node.start_point[0] + 1,
                        col=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                    ))
                    return

            elif node.type == "assignment":
                # CONSTANT = value (top-level or in class/module)
                const_node = find_child_by_type(node, "constant")
                if const_node:
                    name = const_node.text.decode("utf-8")
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
        """Extract Ruby references (calls, inheritance, mixins)."""
        references: list[Reference] = []

        def walk(node, current_function=""):
            if node.type == "call":
                _extract_call_ref(node, current_function, references, self._MIXIN_METHODS)

            elif node.type == "class":
                name_node = find_child_by_type(node, "constant")
                class_name = name_node.text.decode("utf-8") if name_node else ""
                # Check for superclass
                super_node = find_child_by_type(node, "superclass")
                if super_node:
                    parent_node = find_child_by_type(super_node, "constant")
                    if parent_node:
                        references.append(Reference(
                            from_symbol=class_name or current_function or "<module>",
                            to_symbol=parent_node.text.decode("utf-8"),
                            kind="extends",
                            line=node.start_point[0] + 1,
                        ))
                for child in node.children:
                    walk(child, current_function=class_name or current_function)
                return

            elif node.type == "module":
                name_node = find_child_by_type(node, "constant")
                mod_name = name_node.text.decode("utf-8") if name_node else ""
                for child in node.children:
                    walk(child, current_function=mod_name or current_function)
                return

            elif node.type == "method":
                name_node = find_child_by_type(node, "identifier")
                method_name = name_node.text.decode("utf-8") if name_node else ""
                for child in node.children:
                    walk(child, current_function=method_name or current_function)
                return

            for child in node.children:
                walk(child, current_function)

        walk(tree.root_node)
        return references


# ---------------------------------------------------------------------------
# Ruby-specific helpers
# ---------------------------------------------------------------------------

def _get_call_name(call_node: tree_sitter.Node) -> str | None:
    """Get the function/method name from a call node."""
    # Direct call: identifier(args) — identifier is a direct child
    for child in call_node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8")
    return None


def _get_call_method_name(call_node: tree_sitter.Node) -> str | None:
    """Get method name from object.method call pattern.

    In Ruby AST: call → [receiver, identifier(method_name), argument_list]
    The receiver can be a constant, identifier, or another call.
    The second identifier child is the method name (first may be receiver).
    """
    identifiers = [c for c in call_node.children if c.type == "identifier"]
    if len(identifiers) >= 2:
        return identifiers[1].text.decode("utf-8")
    elif len(identifiers) == 1:
        # Check if there's a constant or other receiver before the identifier
        has_receiver = any(
            c.type in ("constant", "call", "instance_variable", "self")
            for c in call_node.children
        )
        if has_receiver:
            return identifiers[0].text.decode("utf-8")
    return None


def _extract_call_ref(
    call_node: tree_sitter.Node,
    current_function: str,
    references: list[Reference],
    mixin_methods: frozenset,
) -> None:
    """Extract references from a Ruby call node."""
    from_sym = current_function or "<module>"

    # Get function name (first identifier child)
    func_name = _get_call_name(call_node)

    # Skip require/require_relative (handled as import symbols)
    if func_name in ("require", "require_relative"):
        return

    # include/extend/prepend → implements reference
    if func_name in mixin_methods:
        arg_list = find_child_by_type(call_node, "argument_list")
        if arg_list:
            for arg in arg_list.children:
                if arg.type == "constant":
                    references.append(Reference(
                        from_symbol=from_sym,
                        to_symbol=arg.text.decode("utf-8"),
                        kind="implements",
                        line=call_node.start_point[0] + 1,
                    ))
                elif arg.type == "scope_resolution":
                    # Module::Constant
                    const = find_child_by_type(arg, "constant")
                    if const:
                        references.append(Reference(
                            from_symbol=from_sym,
                            to_symbol=const.text.decode("utf-8"),
                            kind="implements",
                            line=call_node.start_point[0] + 1,
                        ))
        return

    # Regular method/function call
    # Check for receiver.method pattern
    method_name = _get_call_method_name(call_node)
    if method_name and method_name != func_name:
        # obj.method() call
        references.append(Reference(
            from_symbol=from_sym,
            to_symbol=method_name,
            kind="calls",
            line=call_node.start_point[0] + 1,
        ))
    elif func_name:
        # Simple function call or attr_reader/attr_writer (skip those)
        if func_name not in ("attr_reader", "attr_writer", "attr_accessor",
                              "private", "protected", "public",
                              "puts", "print", "p", "pp", "raise"):
            references.append(Reference(
                from_symbol=from_sym,
                to_symbol=func_name,
                kind="calls",
                line=call_node.start_point[0] + 1,
            ))

    # Check for Constant.method() pattern (e.g., HTTPClient.post)
    const_node = find_child_by_type(call_node, "constant")
    if const_node and method_name:
        # The constant is a receiver — already handled via method_name
        pass
    elif const_node and not method_name:
        # Bare constant reference in call context (e.g., Transaction.new)
        pass
