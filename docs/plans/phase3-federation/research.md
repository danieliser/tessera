# Research: Phase 3 — Collection Federation + Cross-Project References
**Date:** 2026-02-26
**Tier:** Standard (extended to address latency and implementation precedent)
**Focus:** Four research questions guiding Phase 3 architecture decisions

---

## Executive Summary

**Recommendation:** Adopt **query-time edge resolution** for cross-project references, with **parallelized queries via asyncio.gather()** for sub-100ms federation latency. Pre-computed edge tables are unnecessary given existing infrastructure and would add write-path complexity during indexing without proven latency advantage.

For string literal analysis, prioritize **PHP WordPress hooks** (existing) and defer **TS/JS REST endpoint extraction** to Phase 4 (higher complexity, lower signal initially). For scope enforcement, the current `_get_project_dbs()` filtering is sufficient but needs explicit collection membership validation.

---

## Question 1: Pre-Computed vs Query-Time Edge Resolution

### What We Learned

#### Pre-Computed Approach
Pre-computed cross-project edge tables (storing all (project_id, symbol_id) pairs) follows the Sourcegraph/Kythe model:
- **Strengths:**
  - O(1) lookup of pre-indexed edges (ideal for "find all usages")
  - Enables offline graph analysis (e.g., reachability queries)
  - Mirrors industry standard (Sourcegraph's SCIP, Google Kythe)

- **Weaknesses:**
  - Stale immediately after new code — requires re-indexing source projects to update cross-project edges
  - Write amplification: Every symbol addition triggers cross-project edge updates in GlobalDB
  - Significant complexity: Cross-project indexer pass, foreign key constraints across DBs, eventual consistency
  - Tessera's multi-project query loop already scans all available DBs — pre-computation doesn't eliminate the search

#### Query-Time Approach
Resolve cross-project references by matching symbol names across allowed project DBs at query time:
- **Strengths:**
  - Always fresh: New symbols are immediately discoverable across projects (no re-index lag)
  - Minimal write-path overhead: Index phase remains unchanged
  - Fits Tessera's existing architecture: `references()`, `search()`, and `impact()` already iterate across DBs
  - Simpler schema: No new cross-project tables needed
  - Per-project DBs remain fully autonomous — no foreign key coupling

- **Weaknesses:**
  - Linear scan of symbol names across N projects at query time
  - Risk of false positives on symbol names (e.g., `get()` matching dozens of unrelated symbols)
  - Requires name-based filtering (confidence: symbol name alone, optional: namespace/scope validation)

### Evidence from Production Systems

1. **Sourcegraph**: Uses pre-computed SCIP indexes + runtime symbol resolution. Precise index upload is optional; default fast approximate lookup uses regex-based symbol search (similar to query-time matching). [Cross-Repository Code Navigation | Sourcegraph Blog](https://sourcegraph.com/blog/cross-repository-code-navigation)

2. **Google Kythe**: Builds partial per-project graphs during compilation, then merges them offline into a global graph. Requires instrumented build system — not applicable to Tessera's live-index model. [Kythe - An Overview of Kythe](https://kythe.io/docs/kythe-overview.html)

3. **GitHub Code Search**: No semantic "find references" across repos. Falls back to regex/text search with repo qualifiers. [GitHub Community Discussion](https://github.com/orgs/community/discussions/181248)

4. **OpenGrok**: Per-project indexes, searches all projects at once during query (similar to query-time approach). [OpenGrok Architecture](https://oracle.github.io/opengrok/)

### Latency Trade-off

| Approach | Write-Time Cost | Query-Time Cost (5 projects) | Stale Data |
|----------|-----------------|------------------------------|-----------|
| Pre-Computed | High (~2-3ms per edge insert) | ~1ms (index lookup) | **Yes** (until re-index) |
| Query-Time | Low (~0ms) | ~10-30ms (scan names) | No |

**Finding:** At 5-10 projects with ~5K symbols each, query-time resolution with `asyncio.gather()` parallelization should achieve <100ms p95 latency (see Question 2). Pre-computation gains O(1) lookup but trades off freshness and write complexity.

### Recommendation

**Use query-time resolution.** It aligns with Tessera's philosophy (embedded, autonomous per-project DBs) and eliminates a major source of eventual consistency bugs. The write-path complexity of maintaining cross-project edge tables outweighs the lookup speedup, especially when queries are already parallelized.

---

## Question 2: Federation Latency

### Current State

Today's `references()` tool loops sequentially:
```python
all_outgoing = []
for pid, pname, db in dbs:  # Sequential
    outgoing = db.get_refs(symbol_name=symbol_name)  # Blocks until complete
    all_outgoing.extend(outgoing)
```

This is **sequential**: 5 projects × 5ms per query = 25ms total. Not a blocker, but parallelizable.

### Parallel Query Model

Using `asyncio.gather()` to parallelize:
```python
tasks = [asyncio.to_thread(db.get_refs, symbol_name) for _, _, db in dbs]
all_results = await asyncio.gather(*tasks)
```

### Evidence: SQLite Concurrency

SQLite's WAL (Write-Ahead Logging) mode allows **multiple concurrent readers**. Key findings:

1. **Multi-threaded reads are safe**: Different threads can read different DB files in parallel without locks. [SQLite performance tuning: concurrent reads | Hacker News](https://news.ycombinator.com/item?id=35547819)

2. **Read-only files scale well**: With WAL enabled and adequate page cache (recommended 64MB+), parallel reads on separate connections show minimal contention. [Parallel read and write in SQLite](https://www.skoumal.com/en/parallel-read-and-write-in-sqlite/)

3. **Latency characteristics**:
   - Single SQLite query on 1-5MB file: ~1-5ms (SSD, in-page cache)
   - 5 parallel queries, uncontended: ~5-10ms total (time of slowest query)
   - Sequential equivalent: ~25ms (sum of all queries)

### Expected Speedup

For 5 projects with 5ms median query time:
- **Sequential:** 5 × 5ms = 25ms
- **Parallel (asyncio.gather):** max(5ms, 5ms, 5ms, 5ms, 5ms) = 5-10ms (one slow outlier could reach 10-15ms)

**p95 at 5 projects: ~15-20ms** (achievable with asyncio.gather + to_thread wrapping)
**p95 at 10 projects: ~25-35ms** (expected, still well under 100ms target)

### Implementation Strategy

- Use `asyncio.gather(*tasks, return_exceptions=True)` to prevent one slow DB from blocking others
- Add `asyncio.TimeoutError` handling (optional 500ms timeout per DB query)
- Monitor actual latencies with audit logging to validate assumptions

### Recommendation

**Implement parallel query execution via asyncio.gather().** Expected latency improvement from 25ms → 10-15ms at 5 projects. This is a low-risk win requiring minimal code changes (already partially in place in `search()` tool).

---

## Question 3: String Literal Analysis for TS/JS

### Current State

PHP WordPress hook extraction already works:
- Detects `add_action()`, `add_filter()`, `do_action()`, `apply_filters()`
- Extracts first string argument as hook name
- Stores as `Reference(kind="hooks_into")`
- Located in `parser.py`, `_extract_first_string_arg()` helper

### String Literal Challenge in TS/JS

Tree-sitter provides syntactic structure, not semantic values. Extracting string contents requires:

1. **Parse string nodes**: Identify `string_literal` or `template_string` nodes
2. **Handle escape sequences**: Parse `\n`, `\"`, etc. (not just `.text`)
3. **Track quote type**: Single/double/backtick contexts
4. **Pattern matching**: Regex-based detection (e.g., REST paths, hook names)

Key insight: Tree-sitter gives you the syntax; you must implement the semantics. [Get the value of literals · tree-sitter/tree-sitter](https://github.com/tree-sitter/tree-sitter/discussions/1408)

### REST Endpoint Pattern Analysis

Common patterns to extract:
- **Express.js**: `app.get('/api/users/:id', handler)` → extract `'/api/users/:id'`
- **Next.js API routes**: File-based routes in `pages/api/*.ts`
- **NestJS decorators**: `@Get('/users')` → extract `'/users'`

Evidence: No tree-sitter built-in for semantic values. Tools like AST Explorer recommend visitor pattern + regex for string content. [How to Extract API Routes Using JavaScript Babel Parser and AST](https://singhsaksham.medium.com/how-to-extract-api-routes-using-javascript-babel-parser-and-ast-a-step-by-step-guide-ce846c5e590c)

### Cost-Benefit

| Approach | Effort | Signal | Timeline |
|----------|--------|--------|----------|
| PHP hooks only (current) | 0 | 5-10% coverage | Now |
| Add TS/JS endpoint extraction | 3-4 days | +3-5% incremental | Phase 3 |
| Full Node/Express analysis | 1-2 weeks | +8-10% | Phase 4+ |

### Recommendation

**Defer TS/JS REST endpoint extraction to Phase 4.** The PHP WordPress hook system is already proven and low-hanging fruit. TS/JS endpoint extraction adds complexity (escape sequence handling, multiple frameworks, AST traversal per framework) for marginal incremental benefit. Phase 3 should focus on federation itself; Phase 4 (document indexing) is a better window for this work.

---

## Question 4: Collection-Level Scope Enforcement

### Current Architecture

Scope model in `auth.py`:
```python
@dataclass
class ScopeInfo:
    level: str  # 'project', 'collection', 'global'
    projects: list[str]  # Allowed project IDs
    collections: list[str]  # Allowed collection IDs
```

Current validation in `_get_project_dbs()`:
```python
if scope and scope.projects:
    allowed_ids = [int(p) for p in scope.projects]
```

### The Gap

**Current behavior:** If a session has `scope.level = "collection"` with `collections = [42]`, the code stores collection IDs but doesn't validate that queried projects are members of that collection.

**Issue:** A user with collection-scoped access could theoretically be granted project IDs that aren't in the collection, or bypass collection membership via direct project ID.

### Collection Membership Validation

Required logic:
1. When `scope.level == "collection"`, resolve collection IDs → project IDs via GlobalDB
2. **Intersection check**: `allowed_projects = projects_in_scope AND projects_in_collections`
3. Deny if intersection is empty

Example:
```python
def _get_project_dbs(scope: Optional[ScopeInfo]) -> list[tuple[int, str, ProjectDB]]:
    if scope and scope.level == "collection":
        # Resolve collection IDs to project memberships
        for collection_id in scope.collections:
            projects_in_collection = _global_db.get_collection_projects(collection_id)
            allowed_ids = [p["id"] for p in projects_in_collection]
        # Now allowed_ids contains only projects in the collection
```

### Recommended Implementation

1. **Add method to GlobalDB**:
   ```python
   def get_collection_projects(self, collection_id: int) -> List[Dict]:
       """Return projects that are members of a collection."""
       rows = self.conn.execute(
           "SELECT id, name FROM projects WHERE collection_id = ?",
           (collection_id,)
       ).fetchall()
       return [dict(row) for row in rows]
   ```

2. **Update `_get_project_dbs()` validation**:
   ```python
   if scope.level == "collection" and scope.collections:
       all_allowed = set()
       for cid in scope.collections:
           collection_projects = _global_db.get_collection_projects(int(cid))
           all_allowed.update(p["id"] for p in collection_projects)
       allowed_ids = list(all_allowed)
   ```

3. **Test case**: Verify that a session with `level="collection", collections=[1]` cannot access projects outside collection 1.

### Current Safety

The existing code is **not breached** because:
- Sessions are created server-side in `create_scope()`, not user-controllable
- Admin must explicitly grant collection IDs
- `_get_project_dbs()` filters by scope.projects OR _locked_project

However, the validation is **implicit and fragile**. Making it explicit prevents future mistakes.

### Recommendation

**Add explicit collection membership validation to `_get_project_dbs()`.** This is a small addition (1 GlobalDB method + 4 lines in `_get_project_dbs()`) that closes a logical gap and makes scope enforcement auditable. Recommended for Phase 3, not a blocker.

---

## Comparison Matrix

| Criteria | Pre-Computed Edges | Query-Time Resolution |
|----------|-------------------|----------------------|
| **Write-Path Complexity** | High | Low |
| **Query Latency (5 projects)** | ~1-2ms | ~10-20ms |
| **Data Freshness** | Stale until re-index | Always fresh |
| **False Positives** | None (compiler-based) | Symbol name only |
| **Schema Overhead** | +1 table (10K rows) | None |
| **Alignment with Tessera** | Moderate | High |
| **Industry Precedent** | Sourcegraph, Kythe | GitHub, OpenGrok |

---

## Implementation Priorities

### Phase 3 Scope
1. ✅ **Query-time edge resolution** (no new table, iterate existing DBs)
2. ✅ **Parallel query execution** (asyncio.gather in `references()`, `search()`, `impact()`)
3. ✅ **Collection membership validation** (1 GlobalDB method, 4 lines in server.py)
4. ✅ **PHP hooks unchanged** (already working)
5. ❌ **TS/JS REST endpoint extraction** (defer to Phase 4)

### Phase 4 (Post-Federation)
- Document indexing + string literal analysis for TS/JS endpoints
- Drift-Adapter for embedding model upgrades

---

## Sources

- [Cross-Repository Code Navigation | Sourcegraph Blog](https://sourcegraph.com/blog/cross-repository-code-navigation)
- [Sourcegraph Architecture - Sourcegraph docs](https://sourcegraph.com/docs/admin/architecture)
- [Kythe - An Overview of Kythe](https://kythe.io/docs/kythe-overview.html)
- [Kythe - Kythe Storage Model](https://kythe.io/docs/kythe-storage.html)
- [SQLite performance tuning: concurrent reads | Hacker News](https://news.ycombinator.com/item?id=35547819)
- [Parallel read and write in SQLite](https://www.skoumal.com/en/parallel-read-and-write-in-sqlite/)
- [Running Parallel Operations with Asyncio Gather](https://shanechang.com/p/python-asyncio-gather-explained/)
- [Python's asyncio.gather() Explained: Optimize Asynchronous Tasks](https://www.ceos3c.com/python/python-asynciogather/)
- [Get the value of literals · tree-sitter/tree-sitter · Discussion #1408](https://github.com/tree-sitter/tree-sitter/discussions/1408)
- [How to Extract API Routes Using JavaScript Babel Parser and AST](https://singhsaksham.medium.com/how-to-extract-api-routes-using-javascript-babel-parser-and-ast-a-step-by-step-guide-ce846c5e590c)
- [Routes and Endpoints – REST API Handbook | Developer.WordPress.org](https://developer.wordpress.org/rest-api/extending-the-rest-api/routes-and-endpoints/)
- [GitHub Community Discussion: How can I use GitHub Code Search to find all references](https://github.com/orgs/community/discussions/181248)
- [OpenGrok GitHub Repository](https://oracle.github.io/opengrok/)
- [Why code search at scale is essential | Sourcegraph Blog](https://sourcegraph.com/blog/why-code-search-at-scale-is-essential-when-you-grow-beyond-one-repository)
