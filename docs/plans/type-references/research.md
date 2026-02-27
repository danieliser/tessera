# Research: TypeScript Type Reference Extraction

**Date:** 2026-02-25
**Tier:** Quick

## Key Findings

### 1. Complete Node Type Catalog

All tree-sitter TS node types that contain type references, verified via AST dump:

| Category | Parent Node | Child with type name | Example |
|---|---|---|---|
| Variable annotation | `type_annotation` | `type_identifier` | `const a: Foo` |
| Return type | `type_annotation` | `type_identifier` | `function(): Foo` |
| Parameter type | `type_annotation` (in `required_parameter`) | `type_identifier` | `(x: Foo)` |
| Generic type | `generic_type` | `type_identifier` (+ nested) | `Array<Foo>` |
| Union type | `union_type` | `type_identifier` (multiple) | `Foo \| Bar` |
| Intersection type | `intersection_type` | `type_identifier` (multiple) | `Foo & Bar` |
| Conditional type | `conditional_type` | `type_identifier` (3 positions) | `Foo extends Bar ? Baz : never` |
| Interface extends | `extends_type_clause` | `type_identifier` | `interface B extends A` |
| Type alias RHS | `type_alias_declaration` | (walk the RHS subtree) | `type X = Foo` |
| Tuple type | `tuple_type` | `type_identifier` (multiple) | `[Foo, Bar]` |
| Index access | `lookup_type` | `type_identifier` | `Foo["key"]` |
| `as` expression | `as_expression` | `type_identifier` | `x as Foo` |
| `satisfies` | `satisfies_expression` | `type_identifier` | `x satisfies Foo` |
| Type predicate | `type_predicate` | `type_identifier` | `x is Foo` |
| Qualified name | `nested_type_identifier` | `identifier` + `type_identifier` | `NS.Foo` |
| Generic constraint | `type_parameter` → `constraint` | `type_identifier` | `T extends Foo` |
| Default type param | `type_parameter` → `default_type` | `type_identifier` | `T = Foo` |

### 2. Optimal Strategy: Recursive `type_identifier` Collection

Instead of handling each parent node type individually, the most robust approach is:

**Walk the AST and collect ALL `type_identifier` nodes**, then filter out:
1. Nodes that are the **name** of a declaration (first `type_identifier` child of `type_alias_declaration`, `interface_declaration`)
2. Primitive/builtin types

This is simpler, more complete, and automatically handles future grammar additions.

However, this approach loses context (we wouldn't know *how* the type is used). A hybrid approach is better:
- Use recursive `type_identifier` collection for the **value** side of type contexts
- Use specific parent node matching to set the reference `kind` appropriately

### 3. Primitive Type Filtering

Tree-sitter uses `predefined_type` for builtins: `string`, `number`, `boolean`, `void`, `never`, `any`, `unknown`, `object`, `symbol`, `bigint`, `undefined`, `null`.

These appear as `predefined_type` nodes, NOT `type_identifier`. So **no filtering needed** — the grammar already distinguishes them.

Additional identifiers to skip: single-letter type parameters (`T`, `K`, `V`, `U`, etc.) — these are generic parameters, not type references. Can be filtered by checking if the identifier is declared as a `type_parameter` in scope, or heuristically by single uppercase letter.

### 4. How Similar Tools Handle This

- **SCIP/Sourcegraph**: Uses the TypeScript compiler API (`ts.createProgram`) for full semantic resolution. Overkill for our use case.
- **rust-analyzer**: Walks HIR (high-level IR) nodes. Each type reference is resolved to a definition. We're doing the name-level equivalent with tree-sitter.
- **CodeQL**: Has `TypeExpr` and `TypeAccess` concepts that capture all syntactic type references. Our approach mirrors CodeQL's `TypeAccess` pattern.

### 5. Recommendation

**Single-pass `type_identifier` collector** integrated into the existing walk function:

1. When entering a type context (type_annotation, type_alias RHS, extends_type_clause, etc.), collect all `type_identifier` descendants
2. Exclude declaration names (the `type_identifier` that IS the declared name)
3. No primitive filtering needed (grammar handles it via `predefined_type`)
4. Skip single-letter identifiers as likely type parameters
5. Reference kind: `type_reference` for all type usages

This adds ~40-60 lines to the existing walk function.

## Sources

- [tree-sitter-typescript grammar](https://github.com/tree-sitter/tree-sitter-typescript)
- Direct AST verification via `parse_file()` on 15+ edge case patterns
