# Intake: TypeScript Type Reference Extraction

**Date:** 2026-02-25
**Tier:** Quick
**Project Type:** Technical (parser enhancement)

## The Right Question

**"What forms of symbol usage should Tessera track beyond function calls, and how should they map to the existing reference/edge model?"**

The TS reference extractor currently only captures `call_expression` nodes. It completely misses type usage — annotations, return types, generics, interface extends, type aliases, `as` casts. This means impact analysis for any type/interface change returns zero results.

## Adjacent Questions

1. Should PHP get the same treatment? (Yes, but out of scope for this plan — same pattern applies later)
2. Do we need new reference `kind` values? (Yes — "type_reference" distinct from "calls")

## Premise Validation

Tree-sitter's TypeScript grammar exposes all needed nodes:

| Usage Pattern | Node Type Path |
|---|---|
| `const a: Foo` | `type_annotation` → `type_identifier` |
| `function(): Bar` | `type_annotation` → `type_identifier` |
| `Array<Foo>` | `generic_type` → `type_identifier` |
| `interface Bar extends Foo` | `extends_type_clause` → `type_identifier` |
| `type Baz = Foo & {...}` | `intersection_type` / `union_type` → `type_identifier` |
| `{} as Bar` | `as_expression` → `type_identifier` |

All confirmed via AST dump. The `Reference` dataclass uses a free-text `kind` field, no schema migration needed.

## Constraints

- One file: `src/tessera/parser.py` (`_extract_references_typescript`)
- Must not regress existing call/extends/implements reference extraction
- Must filter out primitive types (string, number, boolean, void, etc.)
- New reference kind: `type_reference` (distinguishes from `calls`)

## Success Criteria

1. `references("CallToAction")` on PM packages returns type annotation usages
2. `impact("CallToActionSettings")` returns non-zero results for type dependencies
3. All 342+ existing tests still pass
4. New tests cover each type reference pattern

## Non-Goals

- Full type resolution / generic parameter tracking
- Re-export tracing
- PHP type hints (future, same pattern)
- Modifying DB schema
