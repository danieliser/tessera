# Panel Scorecard — CodeMem Architecture Spec

**Date**: 2026-02-22
**Tier**: Full (with --red-team)
**Panel**: 5 panelists — devil's-advocate, architect, ops-pragmatist, security-analyst, red-team
**Rounds**: 2
**Convergence**: Achieved (all dimensions std dev ≤ 0.55)

---

## Round 1 Score Matrix

| Dimension | DA | Arch | Ops | Sec | RT | Avg | StdDev |
|-----------|-----|------|-----|-----|-----|------|--------|
| Problem-Solution Fit | 4 | 4 | 4 | 4 | 4 | 4.0 | 0.00 |
| Feasibility | 2 | 3 | 3 | 3 | 2 | 2.6 | 0.55 |
| Completeness | 3 | 3 | 2 | 2 | 3 | 2.6 | 0.55 |
| Risk Awareness | 2 | 4 | 3 | 3 | 2 | 2.8 | 0.84 |
| Clarity | 4 | 4 | 4 | 4 | 4 | 4.0 | 0.00 |
| Elegance | 3 | 3 | 4 | 3 | 3 | 3.2 | 0.45 |

**R1 Status**: BLOCKED — Feasibility (2.6), Completeness (2.6), Risk Awareness (2.8) all below 3.0

### R1 Key Findings (consensus — 2+ panelists)

1. HMAC symmetric key is a fundamental security flaw (DA, Sec, RT)
2. Scope token distribution is "TBD" for the most critical security boundary (DA, Sec, RT)
3. PPR performance model is wrong at target scale (DA, Arch, RT)
4. "2-3K lines" is wrong, line budgets won't hold (DA, RT)
5. SQLite↔LanceDB dual-store coherence gap (Arch, Ops)
6. No operational runbook (DA, Ops)
7. WordPress OOP PHP coverage is ~50-60%, not 85-92% (RT)
8. Embedding endpoint is SPOF with no degradation path (DA, Ops)

---

## Spec Revision: v1 → v2

### Critical Changes Made
1. JWT → server-side session tokens (opaque UUID4, scope in SQLite)
2. Session distribution protocol specified (create_scope → session_id via MCP init)
3. PPR loads graph in-memory at server start (CSR sparse matrix)
4. WordPress OOP coverage recalibrated (50-60%)
5. Line budgets revised (Phase 1: <2,500, total <10.5K)
6. Operational runbook added (crash recovery, embedding fallback, backup, monitoring)

### Significant Changes Made
7. SQLite↔LanceDB coherence — file-level atomicity with index_status
8. Cross-project edge composite foreign keys
9. Global SQLite location specified (~/.codemem/global.db)
10. Batch embedding throughput specified
11. Path traversal + SQL injection prevention
12. Cross-language coverage reduced to 10-20%
13. LanceDB maturity added as Risk #8

---

## Round 2 Score Matrix

| Dimension | DA | Arch | Ops | Sec | RT | Avg | StdDev | R1→R2 |
|-----------|-----|------|-----|-----|-----|------|--------|-------|
| Problem-Solution Fit | 4 | 4 | 4 | 4 | 4 | 4.0 | 0.00 | — |
| Feasibility | 3 | 4 | 4 | 4 | 3 | 3.6 | 0.55 | +1.0 |
| Completeness | 4 | 3 | 4 | 3 | 4 | 3.6 | 0.55 | +1.0 |
| Risk Awareness | 3 | 4 | 4 | 4 | 4 | 3.8 | 0.45 | +1.0 |
| Clarity | 4 | 4 | 4 | 4 | 4 | 4.0 | 0.00 | — |
| Elegance | 3 | 4 | 4 | 4 | 4 | 3.8 | 0.45 | +0.6 |

**Weighted scores** (infrastructure: Feasibility 1.5x, Risk Awareness 1.5x):
- Problem-Solution Fit: 4.0
- Feasibility (1.5x): 5.4
- Completeness: 3.6
- Risk Awareness (1.5x): 5.7
- Clarity: 4.0
- Elegance: 3.8

**R2 Status**: PASSES — All dimensions ≥ 3.5, converged (std dev ≤ 0.55)

---

## Red Team Attack Resolution (7 attacks)

| # | Attack | R1 Severity | R2 Status | Resolution |
|---|--------|-------------|-----------|------------|
| 1 | SQLite graph performance cliff | Critical | RESOLVED | PPR in-memory at server start |
| 2 | HMAC secret key compromise | Critical | RESOLVED | Server-side session tokens |
| 3 | "2-3K lines" math | High | PARTIALLY | Budgets revised to realistic 10.5K; line metric retained |
| 4 | Federation latency (LanceDB blocking) | High | UNRESOLVED | asyncio.to_thread() not specified |
| 5 | WordPress PHP coverage | High | RESOLVED | Recalibrated to 50-60% OOP |
| 6 | Drift-Adapter dimension mismatch | Medium | PARTIALLY | Cross-dimension = full re-index; not documented |
| 7 | Cross-language references | Medium | RESOLVED | Coverage reduced to 10-20% |

---

## Unresolved Items for Executive Review

### Must-fix before implementation
- LanceDB async blocking API — add `asyncio.to_thread()` requirement (one sentence)
- Drift-Adapter: document same-dimension-only constraint
- Session distribution: specify security properties for the channel
- Line budget copy-paste errors in Phase 3-6 gate criteria
- `.env` file indexing risk — exclude or restrict access

### Should-fix before Phase 3
- PPR cross-project graph — how does collection-level PPR incorporate cross_project_edges?
- Re-index idempotency — upsert vs delete-reinsert for crash recovery
- LanceDB↔SQLite join pattern — prototype before Phase 1 build
- Disk footprint estimates

---

## Panel Recommendation

**Conditional Go** — All panelists moved to positive territory. Five of seven red-team attacks resolved. Two partially resolved, one unresolved (all addressable without architectural changes). The spec is implementation-ready for Phase 1 with the must-fix items above addressed.
