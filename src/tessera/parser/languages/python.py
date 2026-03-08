"""Python language extractor with signal/event edge detection."""

import tree_sitter

from tessera.parser._base import LanguageExtractor
from tessera.parser._helpers import node_signature
from tessera.parser._patterns import Reference, Symbol, find_child_by_type


class PythonExtractor(LanguageExtractor):
    """Extract Python symbols and references, including Django-style signal events.

    Event patterns (signals):
    - signal_obj.connect(handler) → registers_on
    - signal_obj.send(...) → fires

    Python signals are method calls on signal objects, not string-based like PHP/JS hooks.
    Example: user_created.connect(handle_user) where user_created is the signal name.
    """

    language = "python"
    extensions = (".py",)
    grammar_module = "tree_sitter_python"
    grammar_func = "language"

    # Python signal patterns: object.method(...)
    # The method name (connect, send) determines the kind
    # The object name becomes the event/signal name
    EVENT_REGISTERS = {
        "connect": "registers_on",
    }
    EVENT_FIRES = {
        "send": "fires",
        "send_robust": "fires",
    }

    def extract_symbols(self, tree: tree_sitter.Tree, source_code: str) -> list[Symbol]:
        """Extract Python symbol definitions (functions, classes, methods, imports)."""
        symbols = []
        current_class = None

        def walk(node, scope=""):
            nonlocal current_class

            if node.type in ("function_definition", "async_function_definition"):
                # Extract function/method (sync or async)
                name_node = find_child_by_type(node, "identifier")
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
                        end_line=node.end_point[0] + 1,
                        signature=node_signature(node),
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
                    end_line=node.end_point[0] + 1,
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

    def extract_references(
        self, tree: tree_sitter.Tree, source_code: str, known_symbols: list[str] | None = None
    ) -> list[Reference]:
        """Extract Python references (calls, imports, extends, events).

        Includes:
        - Regular function/method calls
        - Class inheritance
        - Signal registrations and emissions (via extract_events)
        """
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
                        chain_names = self._extract_call_chain(node)
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
                name_node = find_child_by_type(node, "identifier")
                class_name = (
                    name_node.text.decode("utf-8")
                    if name_node
                    else ""
                )
                # Look for inheritance in arguments
                args_node = find_child_by_type(node, "argument_list")
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
                name_node = find_child_by_type(node, "identifier")
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

        # Extract event references (signals)
        event_refs = self.extract_events(tree, source_code)
        references.extend(event_refs)

        return references

    def extract_events(self, tree: tree_sitter.Tree, source_code: str) -> list[Reference]:
        """Extract Python signal/event references.

        In Python, signals are method calls on signal objects:
        - user_created.connect(handler) → registers_on "user_created"
        - post_save.send(sender=User) → fires "post_save"

        The signal name is the object, the method (connect/send) determines the kind.
        """
        references = []
        all_event_methods = {**self.EVENT_REGISTERS, **self.EVENT_FIRES}

        def walk(node, current_function=""):
            # Look for calls with attributes (obj.method)
            if node.type == "call":
                func_node = node.children[0] if node.children else None
                if func_node and func_node.type == "attribute":
                    # Extract the method name (last identifier in attribute)
                    method_name = None
                    for child in func_node.children:
                        if child.type == "identifier":
                            method_name = child.text.decode("utf-8")

                    # Check if this method is an event operation
                    if method_name and method_name in all_event_methods:
                        # Extract the signal name (object of the attribute)
                        signal_name = self._extract_signal_name(func_node)
                        if signal_name:
                            kind = all_event_methods[method_name]
                            references.append(
                                Reference(
                                    from_symbol=current_function or "<module>",
                                    to_symbol=signal_name,
                                    kind=kind,
                                    line=node.start_point[0] + 1,
                                )
                            )

            # Track current function scope
            new_scope = self._get_function_scope(node) or current_function

            for child in node.children:
                walk(child, new_scope)

        walk(tree.root_node)
        return references

    def _extract_call_chain(self, call_node: tree_sitter.Node) -> list[str]:
        """Extract all method names in a call chain.

        For a.b().c().d(), returns ['d', 'c', 'b'] (outermost to innermost).
        For a simple obj.method(), returns ['method'].
        For a direct func(), returns [].
        """
        chain = []

        # Python: call → attribute → identifier (method name)
        func_node = call_node.children[0] if call_node.children else None
        if func_node and func_node.type == "attribute":
            # Extract the RIGHTMOST identifier from attribute (it's the method name)
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
                chain.extend(self._extract_call_chain(obj_node))

        return chain

    def _extract_signal_name(self, attribute_node: tree_sitter.Node) -> str | None:
        """Extract the signal/object name from an attribute node.

        For user_created.connect, returns "user_created".
        Handles nested attributes like obj.signal.connect by returning the
        first identifier before the last method name.
        """
        # Attribute structure: [object, ".", method_name]
        # We need the first child (object)
        if not attribute_node.children:
            return None

        obj_node = attribute_node.children[0]

        # If object is a simple identifier, return it
        if obj_node.type == "identifier":
            return obj_node.text.decode("utf-8")

        # If object is another attribute, recurse to get the base
        if obj_node.type == "attribute":
            # For nested attributes, return the rightmost identifier before the last method
            # e.g., for obj.attr.method in attr node, we want "attr"
            last_id = None
            for child in obj_node.children:
                if child.type == "identifier":
                    last_id = child.text.decode("utf-8")
            return last_id

        return None

    def _get_function_scope(self, node: tree_sitter.Node) -> str | None:
        """If node is a function/method definition, return its name. Else None."""
        func_types = {
            "function_definition",
            "async_function_definition",
        }
        if node.type not in func_types:
            return None

        name_node = find_child_by_type(node, "identifier")
        return name_node.text.decode("utf-8") if name_node else None
