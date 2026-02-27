# Plan: Phase 3 — Collection Federation + Cross-Project References

**Date:** 2026-02-26
**Tier:** Quick
**Status:** Approved

## Executive Summary

Phase 3 enables Tessera to query across multiple related projects as a single collection. Agents managing plugin ecosystems (WordPress, Node.js monorepos) can trace cross-project dependencies, find where symbols are defined vs. referenced across projects, and visualize inter-project dependency graphs.

Key design decision: **query-time resolution** of cross-project references (no pre-computed edge table). Parallel query execution via `asyncio.gather()` targets <100ms p95 at 5-10 projects. ~700 LOC across 3 files.

## Specification

Full spec: [spec-v1.md](spec-v1.md)

### Changes

#### Change 1: Collection CRUD (GlobalDB)
**File:** `src/tessera/db.py` (~150 LOC)

Add 6 methods to GlobalDB for collection management:
- `create_collection(name, project_ids)` → collection_id
- `get_collection(collection_id)` → dict with projects list
- `list_collections()` → all collections
- `get_collection_projects(collection_id)` → member project dicts
- `add_project_to_collection(collection_id, project_id)`
- `delete_collection(collection_id)` — clears FK, deletes row

Uses existing `projects.collection_id` foreign key. Remove denormalized `projects_json` from collections table if present.

**Acceptance:** Unit tests for all 6 methods including error cases (duplicate name, invalid IDs).

#### Change 2: Parallel Query Execution
**File:** `src/tessera/server.py` (~80 LOC refactor)

Convert sequential loops in 4 query tools to `asyncio.gather()`:
- `search()`, `symbols()`, `references()`, `impact()`
- `file_context()` stays sequential (early-exit logic)

```python
tasks = [asyncio.to_thread(db.method, args) for pid, pname, db in dbs]
results_list = await asyncio.gather(*tasks, return_exceptions=True)
```

Error handling: failed DBs logged at WARNING, skipped. Others continue.

**Acceptance:** 4 tools parallelized. One DB exception doesn't block others. Latency <100ms p95 at 5 projects.

#### Change 3: `cross_refs` MCP Tool
**File:** `src/tessera/server.py` (~80 LOC)

New tool finding cross-project references:
```python
async def cross_refs(symbol_name: str, session_id: str = "") -> str:
```

Algorithm:
1. Find which projects **define** the symbol
2. Find which projects **reference** the symbol
3. Filter for cross-project matches (def in project A, ref in project B)

Output: JSON with `{symbol, definition_projects, cross_refs: [{from_project, to_project, file, line, kind}]}`

**Acceptance:** WordPress hook defined in plugin A, called in plugin B → cross_refs returns the match.

#### Change 4: `collection_map` MCP Tool
**File:** `src/tessera/server.py` (~70 LOC)

New tool for inter-project dependency visualization:
```python
async def collection_map(collection_id: int = 0, session_id: str = "") -> str:
```

Output: JSON adjacency list with `{projects: {name: {id, symbol_count}}, edges: [{from, to, cross_refs, symbols}]}`

**Acceptance:** 3+ WordPress plugins → shows which plugins depend on which, with edge counts.

#### Change 5: Collection Scope Enforcement
**File:** `src/tessera/server.py` (~20 LOC)

Update `_get_project_dbs()` to explicitly resolve collection IDs → project IDs:
```python
if scope.level == "collection" and scope.collections:
    for cid in scope.collections:
        collection_projects = _global_db.get_collection_projects(int(cid))
        all_allowed.update(p["id"] for p in collection_projects)
```

**Acceptance:** Collection-scoped agent sees only projects in its collection. Project-scoped agent cannot access cross-collection data.

#### Change 6: Collection Management MCP Tools
**File:** `src/tessera/server.py` (~80 LOC)

4 admin-level tools (global scope only):
- `create_collection_tool(name, project_ids)`
- `add_to_collection_tool(collection_id, project_id)`
- `list_collections_tool()`
- `delete_collection_tool(collection_id)`

**Acceptance:** All 4 tools registered, global scope required, non-admin gets "Insufficient scope".

### New Test File
**File:** `tests/test_federation.py` (~250 LOC)

- Collection CRUD unit tests
- Parallel query tests (3+ mock DBs, error handling)
- Scope enforcement tests
- cross_refs tool tests (3 projects, cross-project hook matching)
- collection_map tool tests (adjacency structure validation)
- Integration: WordPress plugin ecosystem (3 plugins, 10 hooks spot-checked)
- Latency benchmark: <100ms p95 at 5 projects

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Query-time resolution (no edge table) | Always fresh, no write amplification, fits per-project DB autonomy. Sourcegraph + OpenGrok precedent. |
| asyncio.gather parallelism | SQLite WAL supports concurrent readers. Expected 25ms → 10-15ms at 5 projects. |
| Defer TS/JS string analysis to Phase 4 | PHP hooks work. TS/JS adds 3-4 days for +3-5% coverage. Better bundled with document indexing. |
| FK-based collection membership | Uses existing `projects.collection_id`. Simpler than denormalized JSON. |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| False positive symbol matches (`get()`) | Low confidence in cross_refs | Validate symbol kind in matching. Document limitation. |
| Query timeout at 10+ projects | Latency target violation | Per-query timeout (500ms). Skip slow projects. Benchmark. |
| Stale collection membership | Orphaned entries | FK ON DELETE SET NULL. Admin tools validate. |
| Race condition during query | Inconsistent results | Acceptable: point-in-time snapshots. Document. |

## Files Modified

| File | Changes | LOC |
|------|---------|-----|
| `src/tessera/db.py` | 6 collection CRUD methods | ~150 |
| `src/tessera/server.py` | Parallel queries, cross_refs, collection_map, collection admin tools, scope enforcement | ~330 |
| `tests/test_federation.py` (new) | Unit + integration tests | ~250 |
| **Total** | | **~730** |

## Deferred to Phase 4

- **TS/JS string literal analysis** — REST endpoint URLs (Express, Next.js, NestJS) and event patterns (EventEmitter). Requires tree-sitter string parsing, escape handling, framework-specific patterns. Est. 3-4 days, +3-5% coverage.
- Cross-language reference resolution
- Circular dependency detection
- Dynamic dependency discovery

## Verification

```bash
uv run pytest tests/test_federation.py -v    # New federation tests
uv run pytest tests/ -q                      # Full suite regression

# Manual: index 5+ WordPress plugins, run cross_refs on known hooks,
# verify collection_map shows correct dependency graph
```

## Follow-up

- Execute with `/execute` when ready
- Branch: `phase3-federation`
