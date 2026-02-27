# Plan: TypeScript Type Reference Extraction
**Date:** 2026-02-25
**Tier:** Quick
**Status:** Approved

## Executive Summary

Add type reference extraction to Tessera's TypeScript parser (`_extract_references_typescript` in `parser.py`). Currently only function calls, constructor calls, and class inheritance are tracked. This adds tracking for type annotations, generics, union/intersection types, type aliases, `as`/`satisfies` casts, and type predicates — enabling meaningful `impact()` and `references()` results for types and interfaces.

## Specification

### New Helper: `_collect_type_references`

Recursively walks type-context subtrees, collecting all `type_identifier` nodes as `Reference(kind="type_reference")`. Filters:
- Skip declaration names (avoid self-references in type aliases)
- Skip single uppercase letters (likely generic type parameters)
- Only create refs for identifiers matching `known_symbols` (if provided)

No primitive filtering needed — tree-sitter uses `predefined_type` for builtins.

### Integration Points (4 new handlers in `walk()`)

1. **`type_annotation`** — variable types, param types, return types, generics
2. **`type_alias_declaration`** — `type X = Foo & Bar`, unions, intersections, conditionals
3. **`as_expression` / `satisfies_expression`** — type casts
4. **`type_predicate_annotation`** — `x is Foo` predicates

### Reference Kind

New: `type_reference` (distinct from `calls`, `extends`, `implements`)

### Scope Attribution

- Class property annotations: `from_symbol` = class name
- Module-level annotations: `from_symbol` = `<module>`
- Function parameter/return types: `from_symbol` = function name

## Key Decisions

| Decision | Rationale |
|---|---|
| Recursive `type_identifier` collection | Automatically handles nested generics, unions, intersections, tuples, conditionals — no per-node-type handler needed |
| No primitive filtering | Grammar separates `predefined_type` from `type_identifier` |
| `known_symbols` gating | Only create refs for types that match known exported symbols, consistent with existing call ref extraction |
| Class name for property `from_symbol` | Properties are class-level declarations |
| Skip single-letter type params | Reduces noise from generic parameters |

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Type params (T, K) noise | Low | Skip single uppercase letters |
| Namespace types not matching | Medium | Use full qualified name `Namespace.Foo` |
| Performance regression | Low | One extra node check per AST node, negligible |

## Test Plan

17 unit tests covering each type reference pattern + 1 integration test on real PM packages repo + full regression (342+ tests).

## Follow-up Items

- PHP type hint references (same pattern: `function foo(Bar $bar): Baz`)
- Re-export tracing for cross-file type resolution
- Generic type parameter tracking (which type params are bound to which concrete types)
