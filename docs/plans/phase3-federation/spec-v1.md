# Specification: Phase 3 — Collection Federation + Cross-Project References

**Version:** 1.0
**Date:** 2026-02-26
**Tier:** Implementation
**Status:** Ready for development

---

## Executive Summary

Phase 3 implements collection-scoped multi-project federation for Tessera via query-time cross-project reference resolution and parallel query execution. This spec enables agents to search, navigate, and map dependencies across related projects (e.g., WordPress plugin ecosystems) without pre-computed cross-project edge tables.

**Key decision:** Query-time resolution of cross-project references by matching symbol names across allowed ProjectDB instances. No new GlobalDB cross_project_edges table. Parallelized query execution via asyncio.gather() targets <100ms p95 latency at 5–10 projects.

**Implementation approach:** Minimal — ~800 LOC across 3 files (db.py, server.py, auth.py). Leverages existing multi-project query loop infrastructure. PHP WordPress hooks remain unchanged.

---

## Problem Statement

**Current state:** Phase 1–2 built per-project indexing and single-project query infrastructure. Collection metadata exists in GlobalDB (collections table), but:

- No CRUD methods to manage collections in GlobalDB
- No parallel query execution — all 5 query tools loop sequentially
- No tools to expose cross-project references (symbol defined in ProjectA, used in ProjectB)
- No tools to visualize inter-project dependency graphs
- Collection membership validation is implicit; scope gating for collection-level agents is fragile

**Pain:** Agents indexing plugin ecosystems (e.g., WP, Node.js) cannot reason about which plugins depend on which without manual symbol matching across projects. Sequential queries at 5+ projects add unnecessary latency.

**Who has it:** Plugin ecosystem maintainers, monorepo teams analyzing cross-module impact, dependency analysis tools.

**Severity:** Moderate — blocks full federation use case, but foundational infrastructure (multi-project query loop, scope model) already exists.

---

## Proposed Solution

### 1. Collection CRUD in GlobalDB

Add methods to manage collection membership and project metadata:

```python
def create_collection(self, name: str, project_ids: List[int] = None) -> int:
    """Create a new collection.

    Args:
        name: Unique collection name
        project_ids: Initial list of project IDs to add (optional)

    Returns:
        Collection ID
    """

def get_collection(self, collection_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a collection by ID.

    Returns:
        Dict with keys: {id, name, created_at, projects: [...]}
    """

def list_collections(self) -> List[Dict[str, Any]]:
    """List all collections."""

def get_collection_projects(self, collection_id: int) -> List[Dict[str, Any]]:
    """Get projects that are members of a collection.

    Returns:
        List of project dicts: [{id, name, path, language, ...}, ...]
    """

def add_project_to_collection(self, collection_id: int, project_id: int) -> None:
    """Add an existing project to a collection.

    Raises:
        ValueError if collection_id or project_id not found
    """

def delete_collection(self, collection_id: int) -> None:
    """Delete a collection. Does NOT delete projects."""
```

**Schema:** Use existing `projects.collection_id` foreign key (already present in schema). Remove `collections.projects_json` in migration (denormalized, fragile). Keep `collections` table with (id, name, scope_id, created_at).

### 2. Parallel Query Execution

Convert sequential loops in all 5 core query tools to use `asyncio.gather()`:

```python
# Before (sequential):
all_results = []
for pid, pname, db in dbs:
    results = db.keyword_search(query, limit)
    all_results.extend(results)

# After (parallel):
async def parallel_search():
    tasks = [
        asyncio.to_thread(db.keyword_search, query, limit)
        for pid, pname, db in dbs
    ]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    for results in results_list:
        if isinstance(results, Exception):
            logger.warning("Search on project failed: %s", results)
            continue
        all_results.extend(results)
    return all_results
```

**Applied to:** `search()`, `symbols()`, `references()`, `impact()` tools. (Note: `file_context()` has early-exit logic, keep sequential.)

**Error handling:** `return_exceptions=True` ensures one slow/failed DB doesn't block others. Log failures at WARNING level.

### 3. `cross_refs` MCP Tool

New tool to find cross-project references for a symbol:

```python
@mcp.tool()
async def cross_refs(symbol_name: str, session_id: str = "") -> str:
    """Find cross-project references for a symbol.

    Returns references where the symbol is defined in one project
    and referenced (called, imported, used) in another.

    Input:
        symbol_name: Name of symbol to find references for
        session_id: Optional session token

    Output: JSON object:
    {
        "symbol": "hook_name",
        "cross_refs": [
            {
                "from_project_id": 1,
                "from_project_name": "plugin-a",
                "to_project_id": 2,
                "to_project_name": "plugin-b",
                "references": [
                    {
                        "file": "hooks.php",
                        "line": 42,
                        "kind": "calls",
                        "context": "do_action('hook_name')"
                    }
                ]
            }
        ]
    }
    """
```

**Implementation approach:**

1. Validate session scope (project-level minimum)
2. Get allowed ProjectDBs via `_get_project_dbs(scope)`
3. For each project:
   - Query: Does this project define symbol_name? (symbol in symbols table)
   - If yes, store as "definition project"
4. For each other project (not definition project):
   - Query: Does this project reference symbol_name? (symbol_name in refs.to_symbol_name)
   - If yes, collect references with file, line, kind
5. Filter results: Only include entries where from_project != to_project
6. Format output with project metadata

**Acceptance criterion:** Cross-project references tested on WordPress hooks (e.g., WP plugin A calls hook defined in plugin B).

### 4. `collection_map` MCP Tool

New tool to visualize inter-project dependency graph:

```python
@mcp.tool()
async def collection_map(collection_id: int = 0, session_id: str = "") -> str:
    """Get inter-project dependency graph for a collection.

    If collection_id is 0, use all allowed projects in scope.

    Output: JSON adjacency list with edge counts:
    {
        "collection_id": 42,
        "projects": {
            "plugin-a": {"id": 1, "symbol_count": 150},
            "plugin-b": {"id": 2, "symbol_count": 89}
        },
        "edges": [
            {
                "from": "plugin-a",
                "to": "plugin-b",
                "cross_refs": 7,
                "symbols": ["hook_1", "hook_2"]
            }
        ]
    }
    """
```

**Implementation approach:**

1. Validate session and collection membership if collection_id given
2. Resolve collection_id → list of projects (or use scope.projects if id=0)
3. For each project pair (A → B):
   - Count symbols defined in A that are referenced in B
   - Collect symbol names
   - If count > 0, add edge
4. Return adjacency structure

**Acceptance criterion:** Tested on 3+ WordPress plugins; visualizes which plugins depend on which.

### 5. Collection Scope Enforcement

Harden scope gating to make collection membership explicit:

**In `_get_project_dbs(scope: Optional[ScopeInfo])`:**

```python
def _get_project_dbs(scope: Optional[ScopeInfo]) -> list[tuple[int, str, ProjectDB]]:
    if not _global_db:
        return []

    # Determine allowed project IDs based on scope level
    if _locked_project is not None:
        allowed_ids = [_locked_project]
    elif scope and scope.level == "collection" and scope.collections:
        # Collection scope: resolve collection IDs to project memberships
        all_allowed = set()
        for cid in scope.collections:
            try:
                collection_projects = _global_db.get_collection_projects(int(cid))
                all_allowed.update(p["id"] for p in collection_projects)
            except Exception as e:
                logger.warning("Failed to resolve collection %s: %s", cid, e)
        allowed_ids = list(all_allowed)
        if not allowed_ids:
            logger.warning("Collection scope resolved to 0 projects")
    elif scope and scope.level == "project" and scope.projects:
        allowed_ids = [int(p) for p in scope.projects]
    else:
        # Dev mode: all projects
        projects = _global_db.list_projects()
        allowed_ids = [p["id"] for p in projects]

    # Lazy-load and return
    result = []
    for pid in allowed_ids:
        # [existing lazy-load logic]
    return result
```

**Key change:** When scope.level == "collection", explicitly resolve collection_ids via `get_collection_projects()` instead of relying on pre-populated scope.projects.

### 6. MCP Tools: Collection Management

Add two admin-level tools for collection CRUD:

```python
@mcp.tool()
async def create_collection_tool(name: str, project_ids: list[int] = None, session_id: str = "") -> str:
    """Create a new collection (global scope only).

    Args:
        name: Collection name (must be unique)
        project_ids: List of project IDs to add (optional)
        session_id: Session token

    Returns:
        Collection dict with id, name, projects
    """

@mcp.tool()
async def add_to_collection_tool(collection_id: int, project_id: int, session_id: str = "") -> str:
    """Add a project to a collection (global scope only)."""

@mcp.tool()
async def list_collections_tool(session_id: str = "") -> str:
    """List all collections with member counts (global scope only)."""

@mcp.tool()
async def delete_collection_tool(collection_id: int, session_id: str = "") -> str:
    """Delete a collection. Projects are not deleted (global scope only)."""
```

---

## Architecture

### Data Flow: Cross-Project Reference Query

```
cross_refs("hook_name")
  ↓
validate_session() → check global or project scope
  ↓
_get_project_dbs(scope) → resolve collection IDs to projects, return allowed DBs
  ↓
asyncio.gather(
    db1.lookup_symbols("hook_name") → [symbol_record_1a, ...],
    db2.lookup_symbols("hook_name") → [symbol_record_2a, ...],
    db3.get_refs(to_symbol_name="hook_name") → [ref_3a, ...],
    ...
)
  ↓
Match: symbol definitions across projects + refs in non-definition projects
  ↓
Format output: {from_project, to_project, references[]}
  ↓
Return JSON
```

### Data Flow: Parallel Query Execution

Example for `search()` tool:

```
search(query="authenticate")
  ↓
_get_project_dbs(scope) → [(1, "auth-lib", db1), (2, "oauth-plugin", db2), ...]
  ↓
tasks = [asyncio.to_thread(db.keyword_search, ...) for each db]
  ↓
results_list = await asyncio.gather(*tasks, return_exceptions=True)
  ↓
Merge results, add project_id/project_name metadata
  ↓
Sort by score, limit, return JSON
```

**Latency expectation:** Sequential 5-project query ~25ms → Parallel ~10-15ms (from research).

### Collection Membership Validation

```
validate_session(...) → ScopeInfo{level="collection", collections=[42]}
  ↓
_get_project_dbs(scope)
  ↓
if scope.level == "collection":
    for cid in scope.collections:
        projects = _global_db.get_collection_projects(cid)
        allowed_ids.update(p["id"] for p in projects)
  ↓
Only those project DBs are accessed
```

---

## Files Modified

| File | Changes | LOC |
|------|---------|-----|
| **src/tessera/db.py** | Add 6 GlobalDB collection CRUD methods | ~150 |
| **src/tessera/server.py** | 1. Parallelize 4 query tools (asyncio.gather) 2. Add cross_refs & collection_map tools 3. Add collection CRUD tools 4. Update _get_project_dbs for collection scope | ~300 |
| **src/tessera/auth.py** | No changes required (ScopeInfo already supports collections) | 0 |
| **tests/test_federation.py** (new) | Test cross_refs, collection_map, scope enforcement | ~250 |

**Total:** ~700 LOC (target <1,500 cumulative with existing code).

---

## Acceptance Criteria

### 1. Collection CRUD
- [ ] `create_collection(name, [project_ids])` creates collection, returns ID
- [ ] `get_collection(id)` retrieves collection with projects list
- [ ] `list_collections()` returns all collections
- [ ] `get_collection_projects(id)` returns member projects
- [ ] `add_project_to_collection(cid, pid)` adds project to collection
- [ ] `delete_collection(id)` deletes collection (projects unaffected)

### 2. Parallel Query Execution
- [ ] `search()` uses asyncio.gather() for parallel queries
- [ ] `symbols()` uses asyncio.gather() for parallel queries
- [ ] `references()` uses asyncio.gather() for parallel queries
- [ ] `impact()` uses asyncio.gather() for parallel queries
- [ ] Error in one project DB doesn't block others (return_exceptions=True)
- [ ] Query latency <100ms p95 at 5 projects (benchmark test)

### 3. `cross_refs` Tool
- [ ] Accepts symbol_name, returns cross-project references
- [ ] Tested on WordPress hooks: plugin-a defines hook, plugin-b calls it
- [ ] Output includes from_project, to_project, file, line, kind
- [ ] Scope gating: collection-scoped agent only sees projects in collection
- [ ] Deduplicates references (same ref doesn't appear twice)

### 4. `collection_map` Tool
- [ ] Accepts collection_id or defaults to all allowed projects
- [ ] Returns adjacency list: {projects, edges}
- [ ] Edge includes: from, to, cross_refs count, symbol names
- [ ] Tested on 3+ WordPress plugins
- [ ] Visualizable (JSON structure supports D3/graph rendering)

### 5. Collection Scope Enforcement
- [ ] `_get_project_dbs()` explicitly resolves collection IDs to projects
- [ ] Project-scoped agent cannot access cross-collection projects
- [ ] Collection-scoped agent accessing undefined collection returns empty
- [ ] Session with `level="collection"` and `collections=[1]` sees only projects in collection 1

### 6. MCP Tools Registration
- [ ] `create_collection_tool` available at global scope
- [ ] `add_to_collection_tool` available at global scope
- [ ] `list_collections_tool` available at global scope
- [ ] `delete_collection_tool` available at global scope
- [ ] Non-admin scopes get "Insufficient scope" error

### 7. Integration
- [ ] All 5 query tools work across multiple projects in a collection
- [ ] Scope inheritance: global agent accesses all projects; collection agent sees only collection; project agent sees single project
- [ ] PHP hook extraction still works (unchanged)

---

## Implementation Details

### GlobalDB Collection CRUD (db.py)

**Schema:** Existing `collections` table (id, name, scope_id, created_at). Use `projects.collection_id` FK to track membership. Remove `collections.projects_json` in migration (denormalized).

```python
# pseudocode for GlobalDB methods

def create_collection(self, name: str, project_ids: List[int] = None) -> int:
    scope_id = str(uuid.uuid4())
    with self.conn:
        cursor = self.conn.execute(
            "INSERT INTO collections (name, scope_id) VALUES (?, ?)",
            (name, scope_id)
        )
        collection_id = cursor.lastrowid

        if project_ids:
            for pid in project_ids:
                self.conn.execute(
                    "UPDATE projects SET collection_id = ? WHERE id = ?",
                    (collection_id, pid)
                )
    return collection_id

def get_collection(self, collection_id: int) -> Optional[Dict[str, Any]]:
    row = self.conn.execute(
        "SELECT id, name, scope_id, created_at FROM collections WHERE id = ?",
        (collection_id,)
    ).fetchone()
    if not row:
        return None
    coll = dict(row)
    coll["projects"] = self.get_collection_projects(collection_id)
    return coll

def get_collection_projects(self, collection_id: int) -> List[Dict[str, Any]]:
    rows = self.conn.execute(
        "SELECT id, name, path, language FROM projects WHERE collection_id = ? ORDER BY name",
        (collection_id,)
    ).fetchall()
    return [dict(r) for r in rows]

def list_collections(self) -> List[Dict[str, Any]]:
    rows = self.conn.execute(
        "SELECT id, name, scope_id, created_at FROM collections ORDER BY name"
    ).fetchall()
    return [dict(r) for r in rows]

def add_project_to_collection(self, collection_id: int, project_id: int) -> None:
    # Validate both exist
    project = self.get_project(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    row = self.conn.execute(
        "SELECT id FROM collections WHERE id = ?",
        (collection_id,)
    ).fetchone()
    if not row:
        raise ValueError(f"Collection {collection_id} not found")

    with self.conn:
        self.conn.execute(
            "UPDATE projects SET collection_id = ? WHERE id = ?",
            (collection_id, project_id)
        )

def delete_collection(self, collection_id: int) -> None:
    with self.conn:
        # Clear collection_id from projects
        self.conn.execute(
            "UPDATE projects SET collection_id = NULL WHERE collection_id = ?",
            (collection_id,)
        )
        # Delete collection
        self.conn.execute(
            "DELETE FROM collections WHERE id = ?",
            (collection_id,)
        )
```

### Parallel Query Refactor (server.py)

**Pattern for all 4 tools (search, symbols, references, impact):**

```python
async def search(query: str, limit: int = 10, session_id: str = "") -> str:
    scope, err = _check_session({"session_id": session_id}, "project")
    if err:
        return err

    dbs = _get_project_dbs(scope)
    if not dbs:
        return "Error: No accessible projects"

    try:
        # Create tasks for parallel execution
        tasks = [
            asyncio.to_thread(db.keyword_search, query, limit)
            for pid, pname, db in dbs
        ]

        # Gather results with error handling
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for (pid, pname, db), result in zip(dbs, results_list):
            if isinstance(result, Exception):
                logger.warning("Search on project %d failed: %s", pid, result)
                continue

            # Add project metadata
            for r in result:
                r["project_id"] = pid
                r["project_name"] = pname
            all_results.extend(result)

        # Merge and sort
        all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        all_results = all_results[:limit]

        _log_audit("search", len(all_results), agent_id=...)
        return json.dumps(all_results, indent=2)
    except Exception as e:
        logger.exception("Search error")
        return f"Error: {str(e)}"
```

### cross_refs Implementation (server.py)

```python
@mcp.tool()
async def cross_refs(symbol_name: str, session_id: str = "") -> str:
    scope, err = _check_session({"session_id": session_id}, "project")
    if err:
        return err

    dbs = _get_project_dbs(scope)
    if not dbs:
        return "Error: No accessible projects"

    try:
        # Step 1: Find which projects define this symbol
        definition_projects = {}  # pid → project_name

        for pid, pname, db in dbs:
            syms = await asyncio.to_thread(db.lookup_symbols, symbol_name)
            if syms:
                definition_projects[pid] = pname

        if not definition_projects:
            # Symbol not found in any project
            return json.dumps({"symbol": symbol_name, "cross_refs": []})

        # Step 2: Find all references to this symbol across projects
        all_refs = []
        for pid, pname, db in dbs:
            refs = await asyncio.to_thread(db.get_refs, symbol_name=symbol_name)
            for ref in refs:
                all_refs.append((pid, pname, ref))

        # Step 3: Filter for cross-project matches
        cross_refs = []
        for from_pid, from_pname, ref in all_refs:
            # Check if this reference is to a symbol defined in a different project
            to_symbol_name = ref.get("to_symbol_name")
            if not to_symbol_name:
                continue

            # Find which project defines to_symbol_name
            to_pid = None
            for def_pid, def_pname in definition_projects.items():
                # Check if def_pid defines this symbol
                syms = await asyncio.to_thread(db.lookup_symbols, to_symbol_name)
                if syms and def_pid != from_pid:
                    to_pid = def_pid
                    to_pname = def_pname
                    break

            if to_pid and to_pid != from_pid:
                cross_refs.append({
                    "from_project_id": from_pid,
                    "from_project_name": from_pname,
                    "to_project_id": to_pid,
                    "to_project_name": to_pname,
                    "file": ref.get("file"),
                    "line": ref.get("line"),
                    "kind": ref.get("kind"),
                })

        result = {
            "symbol": symbol_name,
            "definition_projects": definition_projects,
            "cross_refs": cross_refs
        }

        _log_audit("cross_refs", len(cross_refs), agent_id=...)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception("cross_refs error")
        return f"Error: {str(e)}"
```

### _get_project_dbs Collection Scope Enforcement (server.py)

```python
def _get_project_dbs(scope: Optional[ScopeInfo]) -> list[tuple[int, str, ProjectDB]]:
    """Resolve which ProjectDBs the request can access.

    Handles three scope levels:
    1. 'project': List of specific project IDs
    2. 'collection': Collection IDs → resolved to project IDs via GlobalDB
    3. 'global': All registered projects
    """
    if not _global_db:
        return []

    # Step 1: Determine allowed project IDs
    if _locked_project is not None:
        # Server locked to single project via --project flag
        allowed_ids = [_locked_project]
    elif scope and scope.level == "collection" and scope.collections:
        # Collection scope: resolve collection IDs to project memberships
        all_allowed = set()
        for cid in scope.collections:
            try:
                collection_projects = _global_db.get_collection_projects(int(cid))
                all_allowed.update(p["id"] for p in collection_projects)
            except Exception as e:
                logger.warning("Failed to resolve collection %s: %s", cid, e)
                continue
        allowed_ids = list(all_allowed)
        if not allowed_ids:
            logger.warning("Collection scope resolved to 0 projects")
    elif scope and scope.level == "project" and scope.projects:
        # Project scope: use explicit list
        allowed_ids = [int(p) for p in scope.projects]
    else:
        # Dev mode: all registered projects
        projects = _global_db.list_projects()
        allowed_ids = [p["id"] for p in projects]

    # Step 2: Lazy-load ProjectDBs
    result = []
    for pid in allowed_ids:
        # [existing lazy-load logic: check cache, retrieve metadata, open DB]
    return result
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **False positive symbol matches** (e.g., `get()` matches dozens of unrelated symbols) | Low confidence in cross_refs results | Validate symbol kind (function vs constant) in matching. Document limitation. Phase 4: add namespace scope validation |
| **Query timeout at 10+ projects** (parallel query becomes bottleneck) | Violates latency target | Set per-query timeout (500ms). Skip slow projects with warning. Benchmark at 5, 10, 15 projects. If >100ms p95, fallback to sequential |
| **Collection membership cycles** (A→B→C→A) in dependency map | Confusing visualization | Directed acyclic graph validation in collection_map. Reject cycles at create time (deferred to Phase 4) |
| **Stale collection membership** (project deleted but collection references it) | Orphaned collection entry | Foreign key ON DELETE SET NULL on projects.collection_id. Admin tools validate before query |
| **Race condition: project added to collection during query** | Inconsistent results (project in collection but query started before) | Acceptable: queries are point-in-time snapshots. Document as eventual consistency |

---

## Test Strategy

### Unit Tests (test_federation.py)

1. **Collection CRUD (GlobalDB)**
   - Create collection with/without projects
   - Retrieve collection, list all
   - Add project to collection
   - Delete collection (projects unaffected)
   - Errors: duplicate name, invalid IDs

2. **Parallel Query (asyncio)**
   - search() with 3+ mock DBs
   - symbols() with 3+ mock DBs
   - references() with 3+ mock DBs
   - impact() with 3+ mock DBs
   - One DB raises exception → others continue
   - All results have project_id, project_name metadata

3. **Scope Enforcement**
   - Session with level="collection", collections=[1]
   - _get_project_dbs() resolves to projects in collection 1 only
   - Session with level="project", projects=[1,2]
   - _get_project_dbs() returns projects 1 and 2
   - Project-scoped agent cannot access cross-collection projects

4. **cross_refs Tool**
   - Mock 3 projects: A defines hook, B references hook, C has unrelated code
   - cross_refs("hook") returns only A→B reference
   - Deduplicates: same reference twice → returned once
   - Empty result if symbol not found

5. **collection_map Tool**
   - Mock 3 projects with cross-dependencies
   - collection_map(1) returns projects in collection 1
   - Edges include count of cross-refs
   - Symbol names listed for each edge

### Integration Tests (test_federation.py)

1. **WordPress Plugin Ecosystem (3 plugins)**
   - Plugin A: defines hook "myapp_filter"
   - Plugin B: calls do_action("myapp_filter")
   - Plugin C: unrelated code
   - cross_refs("myapp_filter") → [(A, B, {...})]
   - collection_map() → edge A→B with count=1

2. **Latency Benchmark**
   - 5 projects, each ~1000 symbols
   - search() query → p50, p95 latency < 100ms
   - Measure sequential vs parallel speedup

3. **Scope Gating**
   - Create 2 collections: {proj1, proj2} and {proj3, proj4}
   - Agent scoped to collection 1: cross_refs visible only within {proj1, proj2}
   - Agent scoped to project 1: cross_refs empty (no cross-project access)

### Manual Validation

1. **Real WordPress plugins** (PM plugin ecosystem)
   - Index 5+ real plugins
   - cross_refs for known inter-plugin dependencies
   - Spot-check 10 hooks across 3+ plugins
   - Verify collection_map shows correct dependency graph

2. **Latency validation**
   - Profile cross_refs at 5, 10 projects
   - Confirm p95 < 100ms

---

## Non-Goals

- **YAML manifest files:** Database registration via MCP tools is sufficient (no external config needed)
- **Cross-language reference resolution:** Deferred to Phase 5+
- **Pre-computed transitive closure graphs:** Query-time resolution is sufficient
- **Redis caching layer:** SQLite + in-process FAISS is embedded-only
- **TS/JS REST endpoint extraction:** Deferred to Phase 4 (Phase 3 focuses on federation infrastructure)
- **Automatic collection creation:** Collections are admin-created; projects assigned explicitly
- **Circular dependency detection:** Acceptable to return cycles; Phase 4 may add validation

---

## Dependencies

- **Python 3.11+** (existing)
- **SQLite with WAL mode** (existing, used for concurrent reads)
- **asyncio** (standard library, already used in server.py)
- **tree-sitter** (existing, for symbol extraction)
- **PHP WordPress hook extraction** (existing in parser.py, unchanged)

**No new external dependencies.**

---

## Deferred to Phase 4

- **TS/JS string literal analysis** — Extract REST endpoint URLs (Express routes, Next.js API routes, NestJS decorators) and event/hook patterns (EventEmitter.on/emit) from JavaScript/TypeScript. Requires tree-sitter string node parsing, escape sequence handling, and framework-specific pattern matching. Estimated 3-4 days, +3-5% cross-project coverage. PHP WordPress hooks already handled in Phase 1.
- Cross-language reference resolution (e.g., Python imports JS modules)
- Circular dependency detection
- Dynamic dependency discovery (runtime analysis)

---

## Verification Checklist

Before marking Phase 3 complete:

- [ ] All unit tests pass (CRUD, parallel execution, scope enforcement, cross_refs, collection_map)
- [ ] Integration tests pass (WordPress plugins, latency benchmark, scope gating)
- [ ] Manual validation: Real plugin ecosystem, 10 hooks spot-checked
- [ ] Latency: p95 < 100ms at 5 projects, p95 < 150ms at 10 projects
- [ ] Code review: db.py, server.py, test_federation.py
- [ ] Documentation: inline comments on collection resolution logic, asyncio patterns
- [ ] No scope bypass: Collection-scoped agent verified to be unable to access other collections

---

## Appendix: Example Workflows

### Creating a Collection and Querying It

```bash
# 1. Register 3 WordPress plugins
register_project("/path/to/plugin-a", "plugin-a")
register_project("/path/to/plugin-b", "plugin-b")
register_project("/path/to/plugin-c", "plugin-c")

# 2. Create collection
create_collection_tool("wordpress-ecosystem", [1, 2, 3])
# → Returns: {id: 42, name: "wordpress-ecosystem", projects: [...]}

# 3. Create agent session scoped to collection
create_scope_tool(
    agent_id="wp-analyst",
    scope_level="collection",
    collection_ids=[42],
    ttl_minutes=60
)
# → Returns: {session_id: "uuid", ...}

# 4. Query across collection with agent's session
search(query="hook", session_id="uuid")
# → Results from all 3 plugins

cross_refs("my_filter_hook", session_id="uuid")
# → Cross-project references

collection_map(collection_id=42, session_id="uuid")
# → Dependency graph
```

### Collection Scope Enforcement

```bash
# Agent with collection scope
session1 = create_scope_tool(
    agent_id="restricted",
    scope_level="collection",
    collection_ids=[42]
)

# Agent can only see projects in collection 42
search(query="hook", session_id=session1)
# → Only results from plugin-a, plugin-b, plugin-c

# Project outside collection 42 is registered
register_project("/path/to/other-plugin", "other-plugin")
# → Becomes project 4, not in any collection

# Agent cannot see project 4
cross_refs("symbol", session_id=session1)
# → Only cross-refs within collection 42
```

---

**Spec Version:** 1.0
**Ready for implementation:** Yes
**Target completion:** 3–4 weeks
**Owner:** Implementation team
