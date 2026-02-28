# Plan: Phase 5 — PPR Graph Intelligence

**Date:** 2026-02-28
**Tier:** Standard
**Status:** Conditional Go

## Executive Summary

Phase 5 adds Personalized PageRank (PPR) as a third ranking signal to Tessera's search, alongside BM25 keyword and FAISS semantic. PPR uses the existing tree-sitter-extracted call graph (edges table) to rank symbols by structural importance — surfacing architecturally critical functions that text similarity alone misses. The implementation uses scipy CSR sparse matrices with hand-coded power iteration (~50 lines), adding zero new dependencies. A blocking spike test (Phase 5a) validates ≥2% precision lift on real projects before committing to full implementation.

## Specification

See `spec-v2.md` in this directory for the full specification (v2.1, revised after Round 2 panel review).

### Key Deliverables
- New `src/tessera/graph.py` module (~300 LOC): `ProjectGraph` class with PPR computation
- Modified `src/tessera/search.py` (~80 LOC): Three-way RRF integration
- Modified `src/tessera/server.py` (~250 LOC): Graph lifecycle, monitoring, PPR-enhanced impact
- Modified `src/tessera/db.py` (~60 LOC): Graph query methods
- Graceful degradation when graph sparse (edge_count < symbol_count)
- Simple LRU cap (MAX_CACHED_GRAPHS=20) with eviction logging
- Benchmark suite with CI performance gates (<100ms PPR, <50ms search overhead)

### Implementation Roadmap

| Task | Description | LOC | Effort | Dependencies |
|------|-------------|-----|--------|--------------|
| 0 (Phase 5a) | Spike test — validate ≥2% precision lift | — | 1 day | None (BLOCKING GATE) |
| 1 | `graph.py` + performance benchmarks | 350 | 1.5 days | Spike passes |
| 2 | `db.py` graph query methods | 60 | 0.5 day | Spike passes (parallel with T1) |
| 3 | `search.py` three-way RRF | 80 | 1 day | T1, T2 |
| 4 | `server.py` lifecycle + monitoring | 250 | 1.5 days | T1, T3 |
| 5 | Memory monitoring & ops docs | 100 | 0.5 day | T1-T4 |
| 6 | Documentation & cleanup | 100 | 0.5 day | All |
| **Total** | | **~840** | **~6.5 days** | |

### Spike Test Protocol (Phase 5a)

- Index 3 real projects: PHP (~1K symbols), Python (~5K symbols), JavaScript (~3K symbols)
- 10+ queries per project using taxonomy: 30% function-name, 30% domain-concept, 30% code-pattern, 10% known-negatives
- Single developer annotates top-10 results per query as relevant/not-relevant
- Compute nDCG@5 for 2-way vs 3-way RRF
- **Gate:** ≥2 of 3 projects show >2% nDCG lift → proceed. Otherwise defer to Phase 6.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PPR library | scipy CSR + hand-coded power iteration | Zero new deps (scipy already used for Phase 4 Drift-Adapter); full control over convergence |
| Spike test gate | Blocking before full implementation | All 3 panelists + CTO agreed: validate graph density before committing |
| Startup loading | Synchronous | MCP servers start once and run; blocking ~200ms/project is acceptable |
| Graph swap safety | Explicit `threading.Lock` | Don't rely on GIL assumption; proactive safety |
| Memory management | Simple LRU cap (20 graphs) in Phase 5 | Prevents unbounded growth; full configurable LRU in Phase 6 |
| Symbol-to-chunk mapping | `max()` aggregation | Prevents dilution from low-scoring symbols in mixed chunks |
| Graceful degradation threshold | edge_count < symbol_count | Spike test will provide empirical data to refine |

## Conditions (CTO — all resolved in spec v2.1)

1. ~~Spike test protocol~~ — Query taxonomy and annotation method specified in spec
2. ~~Startup blocking~~ — Committed to synchronous; per-project timing + 5s total warning
3. ~~LRU scope~~ — MAX_CACHED_GRAPHS=20 in Phase 5; full LRU in Phase 6
4. ~~AC#4 latency metric~~ — P95 delta from baseline <100ms

## Risk Register

| Risk | Severity | Mitigation | Owner |
|------|----------|------------|-------|
| PPR adds <2% lift (sparse graphs) | High | Phase 5a spike test as blocking gate | Implementer |
| Startup latency regression | Medium | Per-project timing, WARN >500ms, total >5s | Implementer |
| Memory growth (many projects) | Medium | LRU cap at 20, monitoring + logging | Implementer |
| Convergence failure (>50 iterations) | Low | Log warning, return partial results | Implementer |
| Ranking inversion (PPR hurts results) | Medium | Graceful degradation, benchmark suite | Implementer |
| Concurrent reindex race condition | Medium | Explicit threading.Lock, AC#4 test | Implementer |

## Follow-up Items

- **Phase 6: Weighted RRF tuning** — A/B test optimal weights for keyword/semantic/PPR signals
- **Phase 6: Configurable LRU** — Memory-budget-based eviction, disk serialization (.npz)
- **Phase 6: Lazy graph loading** — Load on first search instead of startup (if startup >5s becomes an issue)
- **Post-Phase: Graph backend comparison** — Benchmark NetworkX, scikit-network, fast-pagerank as drop-in replacements (per user request)
- **Graceful degradation threshold refinement** — Use spike test empirical data to adjust from 1.0 avg degree

<details>
<summary>Panel Scorecard</summary>

### Round 1 Scores

| Dimension | DA | Arch | Ops | Avg | StdDev |
|---|---|---|---|---|---|
| Problem-Sol Fit | 3 | 3 | 4 | 3.33 | 0.47 |
| Feasibility | 4 | 4 | 4 | 4.00 | 0.00 |
| Completeness | 2 | 4 | 3 | 3.00 | 0.82 |
| Risk Awareness | 3 | 3 | 3 | 3.00 | 0.00 |
| Clarity | 3 | 4 | 4 | 3.67 | 0.47 |
| Elegance | 3 | 4 | 4 | 3.67 | 0.47 |

### Round 2 Scores (final)

| Dimension | DA | Arch | Ops | Avg | StdDev |
|---|---|---|---|---|---|
| Problem-Sol Fit | 4 | 4 | 4 | 4.00 | 0.00 |
| Feasibility | 4 | 4 | 4 | 4.00 | 0.00 |
| Completeness | 3 | 4 | 3 | 3.33 | 0.47 |
| Risk Awareness | 4 | 5 | 4 | 4.33 | 0.47 |
| Clarity | 4 | 4 | 4 | 4.00 | 0.00 |
| Elegance | 3 | 4 | 4 | 3.67 | 0.47 |

**Convergence:** All StdDev ≤ 0.75 — converged after Round 2.

**Advancement:** Passed with flagged concern (Completeness 3.33 — resolved in spec v2.1).

### Key Panel Contributions
- **Devil's Advocate:** Forced spike test as blocking gate; identified power iteration normalization ambiguity; flagged symbol-to-chunk max() as semantically loose
- **Architect:** Validated architecture fit; pushed benchmarks to Task 1; identified symbol-to-chunk mapping concern; confirmed graph module isolation is clean
- **Ops Pragmatist:** Drove memory monitoring into Phase 5; defined concurrent reindex test criteria; pushed for startup profiling and graph freshness tracking

</details>

<details>
<summary>Executive Review</summary>

### CTO Decision: Conditional Go

**Rationale:** PPR via scipy CSR is proven (HippoRAG 2, Aider). Clean integration with existing RRF. Spike test gates implementation on empirical evidence. ~1 week effort within budget.

**Conditions:** All four resolved in spec v2.1 (spike protocol, startup blocking, LRU scope, AC#4 metric).

**Risks Accepted:** Spike may show <2% lift (defer gracefully); synchronous startup blocking (acceptable for MCP); GIL reliance mitigated by threading.Lock.

**Risks Rejected:** No new dependencies; no >1,000 LOC; no PPR >100ms; no startup >2s regression.

See `exec-review-cto.md` for full review.

</details>
