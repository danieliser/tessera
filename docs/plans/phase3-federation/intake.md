# Intake: Phase 3 — Collection Federation + Cross-Project References

**Date**: 2026-02-26
**Tier**: Quick
**Project Type**: technical (infrastructure)

## Idea

Implement Tessera Phase 3: Collection Federation + Cross-Project References. This enables agents to query across multiple related projects (e.g., a WordPress plugin ecosystem) as a single collection, with cross-project reference tracing and dependency mapping.

## Question Framing

**Right question**: What's the minimum implementation to enable cross-project queries that agents actually use, given that multi-project query infrastructure already exists?

**Adjacent questions**:
- Do we need a pre-computed `cross_project_edges` table, or can cross-project references be resolved at query time by matching symbols across DBs? (User chose: **research first**)
- Is string literal analysis (WordPress hooks) orthogonal to federation? (User chose: **keep bundled**)

**Hidden assumptions**:
- The spec assumes cross-project edges need pre-computation. But existing `references()` already searches symbol names across all allowed DBs.
- The `<100ms at 5 projects` latency target assumes sequential queries are too slow. Need benchmarking.

## What Already Exists

### Infrastructure (from Phase 1-2)
- **Collections table** in GlobalDB schema (id, name, projects_json, scope_id, created_at)
- **Collection-level scope** in auth.py (ScopeInfo, create_scope with collections param)
- **Multi-project query loop** via `_get_project_dbs()` — all 5 tools iterate across DBs
- **PHP WordPress hook extraction** — add_action, add_filter, do_action, apply_filters (parser.py)
- **RRF merging** for multi-source search results

### Missing
- Collection CRUD methods in GlobalDB (create/get/list/add_project/get_projects/delete)
- `cross_refs` MCP tool
- `collection_map` MCP tool
- Cross-project edge resolution (pre-computed table OR query-time — to be researched)
- Parallel query execution (asyncio.gather for federation latency)
- TS/JS string literal analysis for REST endpoints
- Scope filtering validation (collection membership check)

## Spec Reference

Primary spec: `docs/plans/architecture/spec-v2.md`, lines 564-587.

## Constraints
- Python 3.11+, tree-sitter, SQLite-only (no external servers)
- Must maintain backward compatibility — existing single-project mode unaffected
- Line budget: <1,500 new lines (spec target)
- Latency target: <100ms p95 federated query at 5-10 projects

## Success Criteria
- Collection CRUD works: create, list, add projects, query
- Federated search returns results from all collection projects with project metadata
- `cross_refs` returns cross-project references for a given symbol
- `collection_map` returns inter-project dependency graph
- String literal analysis recovers +5-10% hook coverage
- Scope gating: project-scoped agents cannot access cross-collection data
- Latency validated at 5+ projects

## Non-Goals
- YAML manifest files (database registration is sufficient)
- Cross-language reference resolution (deferred to Phase 5+)
- Pre-computed transitive closure graphs
- Redis caching layer
