# Code Intelligence Guide

Tessera extracts structural information from code using tree-sitter and exposes it as a searchable symbol graph stored in SQLite. This guide covers the six code intelligence tools: `symbols`, `references`, `file_context`, `impact`, `cross_refs`, and `collection_map`.

Unlike text search, these tools understand code structure. They answer questions like "where is this defined," "who calls this," and "what breaks if I change this."

## Core Concepts

### Symbols

A **symbol** is a named, addressable code element: a function, class, method, import, or variable. Every symbol has:
- **Name** — the identifier (`ProjectDB`, `hybrid_search`, `Event`)
- **Kind** — the type: `function`, `class`, `method`, `import`, `variable`, `interface`, or `type_alias`
- **File path and line number** — where it's defined
- **Signature** — parameter list and return type (when available)
- **Scope** — the parent (class/namespace) if it's a method or nested symbol

Example symbol from PHP with namespace:
```json
{
  "id": 42,
  "name": "App\\Analytics\\Tracker",
  "kind": "class",
  "file": "src/Analytics/Tracker.php",
  "line": 7,
  "signature": null,
  "scope": null
}
```

### References

A **reference** is a directed edge from one symbol to another, indicating usage. Reference kinds include:
- **calls** — function/method invocation
- **extends** — class inheritance
- **implements** — interface implementation
- **type_ref** — type annotations (parameter types, return types)
- **import** — module import
- **attribute** — field/property access
- **hooks_into** — WordPress hook registration (PHP-specific)

Example reference (from test fixtures):
```json
{
  "from_symbol_name": "createUser",
  "to_symbol_name": "validate",
  "kind": "calls",
  "line": 25
}
```

### Edges

**Edges** are deduplicated references aggregated into a dependency graph. The system stores edge type, weight, and metadata to support impact analysis and ranking.

## symbols() — Finding Definitions

Use `symbols()` to locate where something is defined. Query by exact name, pattern (wildcards), kind, or language.

### Query patterns

**Find a specific function or class:**
```python
symbols(query="ProjectDB")
```

Returns all symbols named `ProjectDB`. Exact match takes priority; substring fallback applies if no exact match.

**List all classes:**
```python
symbols(query="*", kind="class")
```

Returns every class symbol in all accessible projects. Useful for understanding the object model.

**Find by pattern:**
```python
symbols(query="*search*")
```

Wildcard queries match symbols containing the substring. Returns `hybrid_search`, `normalize_bm25_score`, `search_refs`, etc.

**Filter by language:**
```python
symbols(query="*", kind="function", language="python")
```

Restricts results to Python function definitions only. Supported languages: `python`, `typescript`, `php`, `javascript`, `swift`.

**Find by prefix:**
```python
symbols(query="test_*")
```

Matches all symbols starting with `test_`. Useful for locating all tests related to a feature.

### Response format

```json
[
  {
    "id": 42,
    "name": "hybrid_search",
    "kind": "function",
    "file": "src/tessera/search.py",
    "line": 87,
    "col": 0,
    "signature": "(query: str, limit: int = 10) -> list[SearchResult]",
    "scope": null,
    "project_id": 1,
    "project_name": "tessera"
  }
]
```

**Key fields:**
- `id` — internal symbol ID (used by `references()` and `impact()`)
- `signature` — parameter list and return type
- `scope` — parent (class/namespace) if nested; `null` if file-level
- `project_id` / `project_name` — which project(s) define this symbol

### Practical use cases

- **"I want to find where this class is defined"** → `symbols(query="MyClass")`
- **"Show me all hooks registered in the plugin"** → `symbols(query="*", kind="import")` + filter for WordPress hooks
- **"Find all test files for a feature"** → `symbols(query="test_feature_*")`
- **"List all classes in the codebase"** → `symbols(query="*", kind="class")`

---

## references() — Finding Usages

Use `references()` to see what calls a symbol (callers) and what it depends on (outgoing references).

### Query pattern

**Find all references to a symbol:**
```python
references(symbol_name="handle_request")
```

Returns two lists:

1. **Callers** — Who calls this symbol? Returns symbols that reference `handle_request`.
2. **Outgoing** — What does this symbol call/depend on? Returns symbols that `handle_request` references.

**Filter by reference kind:**
```python
references(symbol_name="ProjectDB", kind="import")
```

Only returns reference edges of the specified kind. Kinds: `call`, `import`, `inherit`, `type_ref`, `attribute`, or `all` (default).

### Response format

```json
{
  "outgoing": [
    {
      "to_symbol": "sqlite3.Connection",
      "kind": "type_ref",
      "line": 45,
      "project_id": 1,
      "project_name": "tessera"
    }
  ],
  "callers": [
    {
      "name": "get_symbols",
      "kind": "call",
      "file_id": 3,
      "line": 102,
      "scope": null,
      "project_id": 1,
      "project_name": "tessera"
    }
  ]
}
```

**Key fields:**
- **Outgoing:** `to_symbol` (what this symbol uses), `kind` (type of reference)
- **Callers:** `name` (symbol calling us), `scope` (if it's a method, its class), `line` (where the call happens)

### Practical use cases

- **"Who calls this function?"** → `references(symbol_name="handle_request")` → check `callers`
- **"What does this function depend on?"** → `references(symbol_name="handle_request")` → check `outgoing`
- **"Find all places that instantiate this class"** → `references(symbol_name="ProjectDB", kind="call")`
- **"Which modules import this utility?"** → `references(symbol_name="normalize_path", kind="import")`

### Real example from test fixtures

Searching `references(symbol_name="validate")` on the TypeScript fixture returns:
- **Callers:** `createUser` calls `validate` (validation step in user creation)
- **Outgoing:** none (validate method doesn't call anything)

This tells you that changing `validate` affects `createUser` and nowhere else.

---

## file_context() — Understanding File Structure

Use `file_context()` to get a complete structural overview of a file without reading the full source.

### Query pattern

**Get the full structure of a file:**
```python
file_context(file_path="src/tessera/search.py")
```

Returns all symbols defined in the file and all references between them.

### Response format

```json
{
  "file": {
    "id": 5,
    "path": "src/tessera/search.py",
    "language": "python",
    "lines": 287,
    "project_id": 1,
    "project_name": "tessera"
  },
  "symbols": [
    {
      "id": 23,
      "name": "hybrid_search",
      "kind": "function",
      "line": 87,
      "col": 0,
      "scope": null,
      "signature": "(query: str, limit: int = 10) -> list[SearchResult]"
    },
    {
      "id": 24,
      "name": "normalize_bm25_score",
      "kind": "function",
      "line": 142,
      "col": 0,
      "scope": null,
      "signature": "(score: float) -> float"
    }
  ],
  "references": [
    {
      "from_symbol_id": 23,
      "to_symbol_id": 24,
      "kind": "calls",
      "line": 112
    }
  ]
}
```

**Key fields:**
- `file.language` — detected language (`python`, `typescript`, `php`, etc.)
- `file.lines` — total line count
- `symbols` — all symbols defined in this file, with position and signature
- `references` — internal call graph (all references within the file)

### Practical use cases

- **"I'm about to modify this file — what's in it?"** → `file_context("src/models.py")`
- **"Find all functions in this file"** → `file_context()` + filter for `kind == "function"`
- **"Map the call graph within this file"** → Check the `references` array
- **"Find all imports at the top of this file"** → `file_context()` + filter for `kind == "import"`

### Real example from test fixtures

On `php_sample.php`:
- **Symbols:** 8 total (2 classes with namespaces, 3 methods, 1 function, 1 import, 1 module pseudo-symbol)
- **References:** 10 edges (3 contains, 2 calls, 1 extends, 2 hook registrations, 2 type references)
- **Key insight:** The file defines `App\Analytics\Tracker` and `App\Analytics\ClickTracker` (extends Tracker), registers WordPress hooks via `add_action()`, and has internal calls to `flush()` and `dispatch()`.

---

## impact() — Change Blast Radius

Use `impact()` to understand what depends on a symbol and will break if you change it. Traces the dependency graph N levels deep.

### Query pattern

**Find all downstream dependents (default: 3 levels deep):**
```python
impact(symbol_name="ProjectDB")
```

Returns every symbol that directly or transitively depends on `ProjectDB`.

**Shallow check (direct callers only):**
```python
impact(symbol_name="ProjectDB", depth=1)
```

Faster query; shows only direct callers.

**Deep trace (full transitive closure):**
```python
impact(symbol_name="ProjectDB", depth=5)
```

Slower but comprehensive. Use when refactoring core infrastructure.

**Include or exclude type references:**
```python
impact(symbol_name="ProjectDB", include_types=False)
```

By default, type annotations (parameter types, return types) are included in the dependency graph. Set `include_types=False` to count only dynamic calls.

### Response format

```json
[
  {
    "id": 42,
    "name": "get_symbols",
    "kind": "function",
    "file": "src/tessera/search.py",
    "line": 87,
    "scope": null,
    "depth": 1,
    "ppr_relevance": 0.85,
    "project_id": 1,
    "project_name": "tessera"
  },
  {
    "id": 99,
    "name": "hybrid_search",
    "kind": "function",
    "file": "src/tessera/search.py",
    "line": 142,
    "scope": null,
    "depth": 2,
    "ppr_relevance": 0.32,
    "project_id": 1,
    "project_name": "tessera"
  }
]
```

**Key fields:**
- `depth` — how many hops from the queried symbol (1 = direct caller, 2 = transitive)
- `ppr_relevance` — PageRank score (if graph intelligence enabled). Higher = more central to the system. Used for sorting.

Results are sorted by `ppr_relevance` (most important first), giving you the highest-impact dependents at the top.

### Depth guide

| Depth | Scope | Use Case |
|-------|-------|----------|
| **1** | Direct callers | Quick "who immediately depends on this?" |
| **2-3** | Typical refactoring | Understand impact for safe renaming/interface changes |
| **5+** | Full transitive | System-level changes (moving core library, major version bump) |

### Practical use cases

- **"Before renaming `normalize_path`, who does it affect?"** → `impact(symbol_name="normalize_path", depth=2)`
- **"Is it safe to change this class's constructor?"** → `impact(symbol_name="ProjectDB", depth=1)` + check count
- **"If I remove this utility function, what breaks?"** → `impact(symbol_name="utility_func", depth=3)` → get full list
- **"Find all code that depends on this deprecated API"** → `impact(symbol_name="old_api", depth=5)`

### Real example from test fixtures

On the Python fixture (Event class hierarchy):

```
impact(symbol_name="Event", depth=1) →
  __init__ (direct: contains)
  serialize (direct: contains)

impact(symbol_name="Event", depth=2) →
  + ClickEvent (extends Event, depth=1)
  + ClickEvent.__init__ (contains, depth=2, calls Event.__init__)
  + distance_from_origin (depth=2, called by ClickEvent.__init__)
```

This tells you that changing `Event` affects its direct methods and subclasses, and transitively affects anything that instantiates `ClickEvent`.

---

## cross_refs() — Cross-Project Dependencies

Use `cross_refs()` to find references from one project to a symbol defined in another. Requires multiple projects indexed in the same Tessera instance.

### Query pattern

**Find all cross-project references:**
```python
cross_refs(symbol_name="SharedDatabase")
```

Returns which projects define `SharedDatabase` and which projects reference it.

### Response format

```json
{
  "symbol": "SharedDatabase",
  "definition_projects": {
    "1": {"id": 1, "name": "core-lib"}
  },
  "cross_refs": [
    {
      "from_project_id": 2,
      "from_project_name": "plugin-a",
      "to_project_id": 1,
      "to_project_name": "core-lib",
      "file": "src/integrations.php",
      "line": 45,
      "kind": "function"
    },
    {
      "from_project_id": 3,
      "from_project_name": "plugin-b",
      "to_project_id": 1,
      "to_project_name": "core-lib",
      "file": "src/handlers.php",
      "line": 78,
      "kind": "function"
    }
  ]
}
```

**Key fields:**
- `definition_projects` — which project(s) define this symbol
- `cross_refs` — list of all cross-project usages
- `from_project_name` / `to_project_name` — direction of the dependency

### Practical use cases

- **"Which plugins depend on this core library class?"** → `cross_refs(symbol_name="PluginAPI")`
- **"Can I safely deprecate this exported API?"** → Check `cross_refs()` to see adoption
- **"Find all integrations with a third-party library"** → `cross_refs(symbol_name="ExternalAPI")`
- **"Map the dependency graph between projects"** → Call `cross_refs()` on key symbols to visualize

---

## collection_map() — Bird's Eye View

Use `collection_map()` to get statistics on all projects in a collection and the dependencies between them.

### Query pattern

**Get overview of all projects:**
```python
collection_map()
```

Returns per-project symbol counts and cross-project dependency edges.

### Response format

```json
{
  "collection_id": 0,
  "projects": {
    "tessera": {"id": 1, "symbol_count": 142},
    "plugin-a": {"id": 2, "symbol_count": 87},
    "plugin-b": {"id": 3, "symbol_count": 56}
  },
  "edges": [
    {
      "from": "plugin-a",
      "to": "tessera",
      "cross_refs": 12,
      "symbols": ["ProjectDB", "hybrid_search", "SearchResult"]
    },
    {
      "from": "plugin-b",
      "to": "tessera",
      "cross_refs": 8,
      "symbols": ["ProjectDB", "SearchResult"]
    }
  ]
}
```

**Key fields:**
- `projects` — per-project metadata (ID, symbol count)
- `edges` — cross-project dependencies
  - `cross_refs` — number of references from A to B
  - `symbols` — which symbols from B are used by A

### Practical use cases

- **"What projects are in this collection?"** → `collection_map()` → check `projects`
- **"Which projects depend on the core library?"** → Filter `edges` for those pointing to the core
- **"Find all projects that reference a symbol"** → `collection_map()` → scan `symbols` array
- **"Is this project critical infrastructure?"** → Count incoming edges from `collection_map()`

---

## Workflow Examples

### 1. Safely Rename a Function

You want to rename `normalize_path()` throughout the codebase. Steps:

1. **Find the definition:**
   ```python
   symbols(query="normalize_path")
   ```
   Returns: file location, line number, signature.

2. **Check impact (direct callers):**
   ```python
   impact(symbol_name="normalize_path", depth=1)
   ```
   Returns: 3 direct callers. Review each one.

3. **Check transitive impact:**
   ```python
   impact(symbol_name="normalize_path", depth=2)
   ```
   Returns: 7 total affected symbols (direct + indirect). Ensures no surprise breakage.

4. **Rename in all locations:**
   Use `impact()` results to list every file needing changes.

5. **Verify:**
   ```python
   symbols(query="normalize_path")
   ```
   Should now return no results (or only references in comments).

### 2. Understand a New File

A colleague committed 300 lines. You need to understand what it does. Steps:

1. **Get structural overview:**
   ```python
   file_context(file_path="src/new_module.py")
   ```
   Returns: all functions, classes, imports, and internal call graph.

2. **Identify entry points:**
   Check for functions/classes with no callers.

3. **Trace key dependencies:**
   For each function, call:
   ```python
   references(symbol_name="main_function")
   ```
   See what it depends on and what calls it.

4. **Check impact on the rest of the codebase:**
   ```python
   impact(symbol_name="main_function", depth=1)
   ```
   See if this new code affects existing code.

### 3. Find All WordPress Hooks

You need to audit all hook registrations. Steps:

1. **Search by pattern:**
   ```python
   symbols(query="add_*", kind="import")
   ```
   Find `add_action`, `add_filter` imports.

2. **Trace hook calls:**
   ```python
   references(symbol_name="add_action", kind="call")
   ```
   Get all places where hooks are registered.

3. **Map the hook ecosystem:**
   For each hook, call:
   ```python
   references(symbol_name="some_hook_name", kind="hooks_into")
   ```
   See what functions listen to this hook.

### 4. Assess Coupling Between Projects

You're planning to split a monolith into microservices. Steps:

1. **Get dependency overview:**
   ```python
   collection_map()
   ```
   See which projects import from which.

2. **Drill into heaviest dependencies:**
   For the symbols listed in edges, call:
   ```python
   cross_refs(symbol_name="SharedClass")
   ```
   Understand exactly what data/APIs are shared.

3. **Quantify effort:**
   Count total `cross_refs` values in edges to estimate refactoring scope.

---

## Best Practices

### Use `depth` strategically

- Start with `depth=1` for quick checks.
- Increase to `depth=2` or `depth=3` for real refactoring.
- Only use `depth=5+` for system-level decisions.

### Combine tools, don't use them in isolation

**Bad:** Use only `symbols()` to find a function, then manually trace all usages.

**Better:** Use `symbols()` to find it, then `impact()` to see all dependents automatically.

**Best:** Use `file_context()` to understand its environment, then `impact()` to check safety, then `references()` to understand its dependencies.

### Validate with code review

Code intelligence tools are deterministic (based on tree-sitter parsing), but they can miss:
- Dynamic calls (`$fn()`, `apply_filters()` without literal hook name)
- Reflection-based code paths
- Eval'd code

Always review results by reading the actual code.

### Check language-specific behavior

Tessera handles language nuances:
- **PHP:** Tracks hook registration (`add_action`, `add_filter`) as `hooks_into` references
- **Python:** Distinguishes async functions from sync
- **TypeScript/JavaScript:** Supports classes, arrow functions, constructor parameters
- **Swift:** Class and protocol definitions

If a reference seems missing, check if it's language-specific dynamism that the parser can't resolve.

---

## Limitations

### What works

- Symbol definitions and direct references
- Inheritance chains (extends, implements)
- Type annotations (parameter and return types)
- Import statements
- Function/method calls with resolvable names

### What doesn't work

- **Dynamic calls:** `call_user_func($fn)`, `apply_filters("hook_$name")` — requires runtime analysis
- **Reflection:** Java/PHP reflection APIs — too complex for static analysis
- **String-based dispatch:** Rails `send()`, Django URL routing strings
- **Auto-aliasing:** Unused imports, aliased symbols

These show up as missing edges. If your code relies heavily on these patterns, supplement with text search and code review.

---

## Examples by Language

### Python

**Find all async functions:**
```python
symbols(query="*", kind="function", language="python")
# Filter results by checking if signature contains "async"
```

**Trace class inheritance:**
```python
references(symbol_name="BaseModel")
# Check "callers" for classes that extend BaseModel
```

### TypeScript / JavaScript

**Find all methods in a class:**
```python
file_context(file_path="src/User.ts")
# Check symbols array for kind="method" with scope="User"
```

**Understand what a constructor does:**
```python
references(symbol_name="User")
# "callers" shows code that instantiates User
```

### PHP

**Find all hooks registered:**
```python
symbols(query="*", kind="import")
# Filter for "add_action", "add_filter", "do_action"
```

**Audit a plugin:**
```python
file_context(file_path="plugin.php")
# Map all classes, functions, and hook registrations
```

**Track hook listeners:**
```python
references(symbol_name="save_post")
# Check "callers" for all functions listening to this hook
```

---

## Performance Notes

### Query speed (typical)

| Tool | Complexity | Time |
|------|-----------|------|
| `symbols()` | O(files) | <10ms |
| `references()` | O(edges) | <50ms |
| `file_context()` | O(symbols in file) | <5ms |
| `impact(depth=1)` | O(direct callers) | <50ms |
| `impact(depth=3)` | O(transitive closure) | <200ms |
| `cross_refs()` | O(all projects) | <100ms |
| `collection_map()` | O(projects × refs) | <500ms |

Speed depends on project size. Tessera indexes are in-process SQLite, so queries run locally.

### Large projects

For projects with >10K symbols:
- Avoid `symbols("*")` without filters — use `kind` and `language` to narrow scope
- Use `depth=1` or `depth=2` for `impact()` instead of full traversal
- Filter `collection_map()` by `collection_id` if using multiple collections

---

## See Also

- [README](/README.md) — overview of all MCP tools
- [Architecture](/docs/architecture.md) — how the symbol graph is built and stored
- [Integration Tests](/tests/integration/test_validation.py) — real examples of symbol extraction
