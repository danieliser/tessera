# Plan: CodeMem — Hierarchical Codebase Memory for AI Agents

**Date:** 2026-02-22
**Tier:** Full (with --red-team)
**Status:** Conditional Go

---

## Executive Summary

CodeMem is a persistent, scope-gated codebase indexing and memory system for always-on AI agents, exposed via MCP. It combines symbol-level code intelligence (tree-sitter), vector semantic search (LanceDB + local embeddings), and graph-based relationship traversal (in-memory PPR) — all without external servers or daemons.

**Why build it:** No existing tool covers >40% of the five critical capabilities needed: symbol-level code intelligence, vector semantic search, graph relationships, multi-repo hierarchy with scope gating, and non-code document indexing. The gap between "tree-sitter symbol lookup" and "persistent hierarchical codebase memory" remains wide as of Feb 2026.

**How it works:** Three-tier hierarchy (project → collection → global) with capability-based scope tokens. Federated search-time merging (data stays at project level, never copies up). Server-side session tokens for access control. ~10.5K lines of medium-weight middleware orchestrating proven libraries.

**Key risk:** This is foundational infrastructure — wrong architecture cascades everywhere. The 5-person red-team panel stress-tested 7 specific risks across 2 rounds. Five were resolved architecturally, two were accepted with mitigations. CTO exec review: Conditional Go with 5 implementation conditions.

---

## Specification

See: `spec-v2.md` (final version, incorporating all panel feedback)

### Key Design Decisions

| Decision | Choice | Rationale | Panel Validation |
|----------|--------|-----------|-----------------|
| Parsing | Tree-sitter universal | 0.902 coverage vs 0.641 LLM-extracted (Jan 2026). Deterministic, fast, language-agnostic. | Unanimous |
| Structural index | SQLite (adjacency tables) | No external server. Proven for code graphs (Aider, RepoGraph). | Architect: sound |
| Vector search | LanceDB (embedded) | Zero-copy versioned, built-in BM25 FTS, no server. | Architect: sound |
| Graph retrieval | In-memory PPR (scipy CSR) | HippoRAG 2 algorithm. Loaded from SQLite at server start. +7% associative reasoning. | Red-team: resolved after R1 |
| Access control | Server-side session tokens | Opaque UUID4, scope in SQLite, validated per-call. Replaced JWT after red-team attack. | Security: approved |
| Hierarchy | Federated search-time merging | Glean-inspired. Data stays at project level. No duplication. | Ops: approved |
| Embeddings | Local OpenAI-compatible endpoint | Zero cost, privacy, swappable. | Unanimous |
| Line budget | ~10.5K total (6 phases) | Revised from 2-3K after red-team detailed count. Medium-weight middleware, not thin glue. | Red-team: partially resolved |

### Architecture Overview

```
MCP Server (stdio + SSE)
├── Scope Validator (server-side session lookup, deny-by-default)
├── Query Executor (routes by scope tier)
│   ├── Search (BM25 + semantic + PPR via RRF)
│   ├── Symbols / References / Impact
│   └── Admin (global scope only)
├── Per-Project Indexes
│   ├── SQLite (symbols, references, edges, files, chunk_meta)
│   └── LanceDB (code_chunks, doc_chunks)
├── Global SQLite (~/.codemem/global.db)
│   ├── projects, collections, sessions
│   ├── indexing_jobs, cross_project_edges
│   └── audit_log
└── In-Memory PPR Graph (CSR sparse matrix, rebuilt after re-index)
```

### Implementation Phases

| Phase | What | Line Budget | Cumulative | Key Gate |
|-------|------|------------|------------|----------|
| 1 | Single-project indexer + scoped MCP | <2,500 | <2,500 | Query latency <100ms p95 |
| 2 | Persistence agent + incremental indexing | <2,000 | <4,500 | Incremental re-index <5s |
| 3 | Collection federation + cross-project refs | <1,500 | <6,000 | Federation latency <100ms (5 projects) |
| 4 | Document indexing + Drift-Adapter | <1,500 | <7,500 | PDF extraction <30s |
| 5 | PPR graph intelligence + impact analysis | <1,500 | <9,000 | PPR <100ms at 50K edges |
| 6 | Always-on persistence + file watcher | <1,500 | <10,500 | File watcher reliable at 20+ dirs |

### Coverage Expectations (Honest)

| Code Pattern | Coverage | Why |
|-------------|----------|-----|
| Procedural PHP / TS / Python (direct calls, imports) | 85-92% | Tree-sitter resolves static references |
| WordPress OOP PHP (`$this->method()`) | 50-60% | Dynamic dispatch unresolvable without type inference |
| WordPress hooks (add_action/apply_filters) | 70-80% | String literal analysis recovers hook names |
| Cross-language REST API (PHP ↔ TypeScript) | 10-20% | Template literals + env vars defeat string matching |

---

## Key Decisions Made During Review

### Architectural Changes from Panel Review

1. **JWT → Server-side session tokens** (Round 1 → v2)
   - Red-team demonstrated HMAC symmetric key is vulnerable on single-machine systems
   - Server-side opaque sessions eliminate the shared-secret problem entirely
   - Simpler, more secure, O(1) revocation

2. **PPR: persistent SQLite → in-memory at server start** (Round 1 → v2)
   - Red-team showed Aider uses in-memory NetworkX, not persistent SQLite queries
   - CSR sparse matrix loaded from SQLite at startup, rebuilt after re-index
   - Eliminates per-query disk I/O for graph traversal

3. **Coverage claims recalibrated** (Round 1 → v2)
   - Original claim "85-92% static coverage" was wrong for WordPress OOP PHP
   - Red-team estimated 50-60% for `$this->method()` dominant patterns
   - Cross-language coverage reduced from "50-70%" to "10-20%"
   - Phase 1 accuracy gates now have per-tier targets

4. **Line budgets revised** (Round 1 → v2)
   - Original "2-3K lines of glue" was 3-5x underestimate
   - Red-team detailed count: Phase 1 alone needs 1,600-2,200 lines
   - Revised to 10.5K total across 6 phases
   - "Thin glue layer" language dropped; now "medium-weight middleware"

5. **Operational runbook added** (Round 1 → v2)
   - DA and Ops flagged zero operational story
   - Added: crash recovery (job queue), embedding fallback (BM25-only), backup/restore, health monitoring, `reset-project` command

---

## Conditions (CTO Exec Review)

Before Phase 1 coding starts:

1. **LanceDB async wrapper**: Add `asyncio.to_thread()` specification for all LanceDB queries to avoid blocking the event loop. Test with 100+ concurrent requests.

2. **Session token distribution protocol**: Document where/how persistence agent transmits session IDs to task agents. Security properties: "Tokens generated in-process, passed via MCP initialize message over stdio. Not transmitted beyond MCP client/server. Assumes trusted environment."

3. **Re-index crash recovery (Phase 1)**: Document limitation: "Re-indexing is NOT idempotent in Phase 1. Crash mid-index requires manual `codemem reindex --force <project>`. Phase 2 adds transaction-based recovery."

4. **LanceDB↔SQLite coherence prototype**: 2-3 hour spike before coding. Prototype the join pattern (SQLite edges → LanceDB chunks → merged results). If latency/consistency issues surface, gate review before Phase 2.

5. **Phase 1 latency gate criteria**: Keyword search <20ms, semantic search <30ms, RRF merge <10ms, total <100ms p95.

---

## Risk Register

| # | Risk | Severity | Status | Mitigation |
|---|------|----------|--------|------------|
| 1 | SQLite adjacency graph at 100K+ edges | Medium | Phase 1 gate | Benchmark at 5-10K edges; contingency: DuckDB |
| 2 | Federation latency at 10+ projects | Low | Validated by Glean | Async parallel queries; target <100ms |
| 3 | Tree-sitter dynamic dispatch (PHP OOP) | Low | Accepted | 50-60% coverage documented; hooks recoverable |
| 4 | Scope token security | Low | Resolved | Server-side sessions; no shared secret |
| 5 | Line budget discipline | Medium | Revised | 10.5K total; per-phase gates; non-goals shield |
| 6 | Embedding model drift | Low | Deferred to Phase 4 | Drift-Adapter (same-dimension only); cross-dimension = full re-index |
| 7 | Cross-language references | Medium | Accepted | 10-20% automated; annotation tool Phase 5+ |
| 8 | LanceDB maturity (<3yr old) | Medium | New in v2 | Pin version; abstraction layer; test upgrades in staging |

---

## Follow-up Items

### Deferred from panel (document but don't implement in Phase 1)
- RS256 asymmetric keys (upgrade path if multi-machine deployment needed)
- Audit log tamper protection (separate append-only store, Phase 3+)
- Incremental indexing injection (prompt injection via indexed content — documented as known risk)
- Schema migration system (numbered .sql files at startup)
- .env file exclusion from document indexing
- PPR cross-project graph at collection scope (Phase 5 design decision)
- Disk footprint estimates (Phase 1 measurement)

### Panel suggestions worth considering
- `codemem doctor` command — runtime self-check (validates SQLite integrity, LanceDB schema, embedding endpoint)
- `--dry-run` flag for reindex command
- `freshness_score` in search results when index is stale
- `source_hash` in doc_chunks for sensitive file identification
- Rate-limiting on failed session lookups

---

<details>
<summary>Panel Scorecard</summary>

### Round 1 Scores (BLOCKED — 3 dimensions below 3.0)

| Dimension | DA | Arch | Ops | Sec | RT | Avg |
|-----------|-----|------|-----|-----|-----|------|
| Problem-Solution Fit | 4 | 4 | 4 | 4 | 4 | 4.0 |
| Feasibility | 2 | 3 | 3 | 3 | 2 | 2.6 |
| Completeness | 3 | 3 | 2 | 2 | 3 | 2.6 |
| Risk Awareness | 2 | 4 | 3 | 3 | 2 | 2.8 |
| Clarity | 4 | 4 | 4 | 4 | 4 | 4.0 |
| Elegance | 3 | 3 | 4 | 3 | 3 | 3.2 |

### Spec Revised (v1 → v2): 6 critical + 9 significant changes

### Round 2 Scores (PASSES — all dimensions ≥ 3.5, converged)

| Dimension | DA | Arch | Ops | Sec | RT | Avg | Δ from R1 |
|-----------|-----|------|-----|-----|-----|------|-----------|
| Problem-Solution Fit | 4 | 4 | 4 | 4 | 4 | 4.0 | — |
| Feasibility | 3 | 4 | 4 | 4 | 3 | 3.6 | +1.0 |
| Completeness | 4 | 3 | 4 | 3 | 4 | 3.6 | +1.0 |
| Risk Awareness | 3 | 4 | 4 | 4 | 4 | 3.8 | +1.0 |
| Clarity | 4 | 4 | 4 | 4 | 4 | 4.0 | — |
| Elegance | 3 | 4 | 4 | 4 | 4 | 3.8 | +0.6 |

Weighted (infrastructure: Feasibility 1.5x, Risk Awareness 1.5x): All pass.

### Red Team Attack Resolution

| Attack | R1 Severity | R2 Status |
|--------|-------------|-----------|
| SQLite graph cliff | Critical | RESOLVED |
| HMAC key compromise | Critical | RESOLVED |
| "2-3K lines" math | High | PARTIALLY RESOLVED |
| LanceDB async blocking | High | UNRESOLVED (one-sentence fix) |
| WordPress PHP coverage | High | RESOLVED |
| Drift-Adapter dimensions | Medium | PARTIALLY RESOLVED |
| Cross-language refs | Medium | RESOLVED |

</details>

<details>
<summary>Executive Review</summary>

### CTO Lens — Conditional Go

**Architecture evaluation**: Technically sound. Tree-sitter, SQLite + LanceDB, in-memory PPR, server-side sessions, federated search — all validated by research and industry precedent. No architectural blockers.

**Risk assessment**: High-confidence mitigations for SQLite scale, scope security, line budget, federation latency. Medium-confidence for LanceDB async API, re-index idempotency, session distribution. Low-risk: embedding drift, hook coverage, disk footprint.

**Decision**: Conditional Go with 5 conditions (all addressable in 48 hours). Proceed to Phase 1 after conditions met.

**Key recommendations**: Start with latency spike prototype. Lock down session model immediately. Treat 10.5K as hard ceiling. Document non-goals obsessively. Defer Drift-Adapter to Phase 4.

</details>
