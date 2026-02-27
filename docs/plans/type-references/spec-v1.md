# Spec: TypeScript Type Reference Extraction

**Version:** 1
**Date:** 2026-02-25

## Executive Summary

Add type reference extraction to Tessera's TypeScript parser. Currently `_extract_references_typescript` only captures function calls (`call_expression`), constructor calls (`new_expression`), and class inheritance (`extends`/`implements`). This leaves type annotations, generics, union/intersection types, type aliases, `as` casts, and `satisfies` expressions completely untracked — making `impact()` and `references()` useless for types and interfaces.

The fix adds a `_collect_type_references` helper that recursively walks type-context subtrees and collects all `type_identifier` nodes, creating `Reference` objects with `kind="type_reference"`. This integrates into the existing walk function with ~50-70 lines of new code.

## Implementation

### New Helper: `_collect_type_references`

```python
def _collect_type_references(
    node: tree_sitter.Node,
    from_symbol: str,
    references: list[Reference],
    declaration_name: str = None,
) -> None:
    """Recursively collect type_identifier nodes from a type context subtree.

    Args:
        node: The type-context root node to walk
        from_symbol: The enclosing symbol (function/class name or "<module>")
        references: List to append Reference objects to
        declaration_name: Name being declared (to exclude from references)
    """
```

**Logic:**
1. If `node.type == "type_identifier"`:
   - Get the text
   - Skip if it matches `declaration_name` (avoid self-references)
   - Skip single uppercase letters (`T`, `K`, `V`, etc.) — likely type parameters
   - Append `Reference(from_symbol, to_symbol=text, kind="type_reference", line=...)`
2. If `node.type == "nested_type_identifier"` (e.g., `Namespace.Foo`):
   - Extract the full qualified name
   - Create one reference for the full name
   - Skip recursing into children (already handled)
3. Otherwise: recurse into all children

**No primitive filtering needed** — tree-sitter uses `predefined_type` for builtins, not `type_identifier`.

### Integration Points in `_extract_references_typescript.walk()`

Add handlers for these node types in the existing `walk` function:

#### 1. Type Annotations (variable, parameter, return types)
```
type_annotation → recurse with _collect_type_references
```
Already present as children of `variable_declarator`, `required_parameter`, `function_declaration`, `method_definition`.

#### 2. Type Alias Declarations
```
type_alias_declaration → walk RHS (everything after `=`)
```
The `type_identifier` immediately after `type` keyword is the declaration name — pass as `declaration_name` to exclude.

#### 3. Interface Extends (already partially handled)
```
extends_type_clause → _collect_type_references
```
The existing `extends_clause` handler covers class extends but `extends_type_clause` (interface extends) may need the type collection helper for generic extends like `extends Array<Foo>`.

#### 4. As/Satisfies Expressions
```
as_expression → walk type child with _collect_type_references
satisfies_expression → walk type child with _collect_type_references
```

#### 5. Type Predicate
```
type_predicate_annotation → type_predicate → _collect_type_references
```

#### 6. Generic Type Parameters (constraints and defaults)
```
type_parameter with constraint → _collect_type_references on constraint
type_parameter with default → _collect_type_references on default
```

### Where NOT to Add Handlers

- `generic_type` — handled by recursing into `type_annotation` which contains it
- `union_type`, `intersection_type`, `conditional_type`, `tuple_type` — handled by recursive descent from their parent context
- `predefined_type` — builtin types, correctly excluded by only matching `type_identifier`

### Reference Kind

New kind: `type_reference`

Existing kinds remain unchanged:
- `calls` — function/method calls, constructor calls
- `extends` — class inheritance
- `implements` — interface implementation

### Node Type Walk Priority

In the `walk()` function, add handlers **before** the generic child recursion. The type annotation nodes appear as children of declarations we already handle (function_declaration, class_declaration), so the collector runs during the normal walk.

Specifically, add these checks in the `walk` function's elif chain:

```python
# Type annotations (variable types, param types, return types)
elif node.type == "type_annotation":
    _collect_type_references(node, current_function or "<module>", references)
    return  # Don't recurse further — collector handles children

# Type alias declarations: type Foo = Bar & Baz
elif node.type == "type_alias_declaration":
    decl_name = None
    for child in node.children:
        if child.type == "type_identifier" and decl_name is None:
            decl_name = child.text.decode("utf-8")
        elif child.type not in ("type", "=", "type_identifier", "type_parameters"):
            # This is the RHS — collect type references
            _collect_type_references(child, decl_name or "<module>", references, declaration_name=decl_name)
    return

# as/satisfies expressions
elif node.type in ("as_expression", "satisfies_expression"):
    # Walk children — the type part will hit type_annotation or type_identifier
    for child in node.children:
        if child.type == "type_identifier":
            _collect_type_references(child, current_function or "<module>", references)
        elif child.type not in ("as", "satisfies"):
            walk(child, current_function)
    return

# Type predicate annotations: x is Foo
elif node.type == "type_predicate_annotation":
    _collect_type_references(node, current_function or "<module>", references)
    return
```

## Test Plan

### Unit Tests (in `tests/test_parser.py`)

Add a new test class `TestTypeScriptTypeReferences`:

1. **test_variable_type_annotation** — `const a: Foo = ...` → ref to `Foo`
2. **test_parameter_type** — `function(x: Foo)` → ref to `Foo`
3. **test_return_type** — `function(): Foo` → ref to `Foo`
4. **test_generic_type** — `const a: Array<Foo>` → refs to `Array` and `Foo`
5. **test_union_type** — `type X = Foo | Bar` → refs to `Foo` and `Bar`
6. **test_intersection_type** — `type X = Foo & Bar` → refs to `Foo` and `Bar`
7. **test_as_expression** — `x as Foo` → ref to `Foo`
8. **test_satisfies_expression** — `x satisfies Foo` → ref to `Foo`
9. **test_type_predicate** — `x is Foo` → ref to `Foo`
10. **test_conditional_type** — `type X = Foo extends Bar ? Baz : Qux` → refs to `Foo`, `Bar`, `Baz`
11. **test_generic_constraint** — `<T extends Foo>` → ref to `Foo`
12. **test_interface_extends_generic** — `interface X extends Array<Foo>` → refs to `Array`, `Foo`
13. **test_nested_type_identifier** — `Namespace.Foo` → ref to `Namespace.Foo`
14. **test_no_primitive_refs** — `const a: string` → no type_reference refs
15. **test_no_self_ref_in_alias** — `type Foo = Foo[]` → ref to `Foo` (array element), NOT self-ref on declaration name
16. **test_skips_single_letter_type_params** — `<T>` → no ref to `T`
17. **test_existing_call_refs_preserved** — verify call_expression refs unchanged

### Integration Test

18. **test_real_repo_type_refs** — Index PM packages `core-data/src/call-to-actions/types/posttype.ts`, verify `CallToAction` has inbound type references from other files.

### Regression

19. Run full test suite — all 342+ existing tests must pass.

## Risks

| Risk | Mitigation |
|---|---|
| Over-counting (same type referenced multiple times on same line) | Acceptable — each annotation IS a separate usage |
| Type parameters (T, K) creating noise | Skip single uppercase letters |
| Performance on large files | Minimal — one extra node type check per AST node |
| Namespace types not matching symbol names | Use full qualified name `Namespace.Foo` |

## Open Questions

1. Should `import type { Foo }` create a type_reference? Currently imports are handled separately. Probably not — imports are already tracked.
2. Should class property type annotations (`private data: Foo`) use the class name or method name as `from_symbol`? Recommendation: class name, since properties are class-level.
