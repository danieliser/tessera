# Remediation Plan — Phase 5: PPR Graph Intelligence

**Source:** `docs/plans/phase5-ppr-graph/VALIDATION.md`
**Date:** 2026-02-28
**Gaps:** 14 missing, 8 partial, 2 deviated
**Remediation tasks:** 10

---

## Batch 1: Critical Fixes (No Dependencies)

### Task 1: Fix doc_search_tool argument order bug

**File:** `src/tessera/server.py`
**Line:** ~410
**Type:** Bug fix
**Severity:** HIGH

Fix positional argument mismatch in `doc_search_tool` call. Currently `limit` maps to `graph` and `effective_formats` maps to `limit` due to positional ordering.

**Change:**
```python
# FROM (positional — wrong order):
asyncio.to_thread(doc_search, query, query_embedding, db, limit, effective_formats)

# TO (keyword — correct):
asyncio.to_thread(doc_search, query, query_embedding, db, graph=None, limit=limit, formats=effective_formats)
```

**Acceptance:** doc_search_tool calls use keyword arguments matching function signature.

---

### Task 2: Add DEBUG log for graph version in search tool

**File:** `src/tessera/server.py`
**Assertion:** 4.2.3
**Type:** Implement per spec
**Severity:** LOW

Add `logger.debug()` call in `search()` tool to log which graph version is being used for each project query.

**Acceptance:** When search executes with graph, DEBUG log emits `graph.loaded_at` value.

---

### Task 3: Add yellow performance threshold (80ms)

**File:** `tests/test_performance_ppr.py`
**Assertions:** 6.1.3, 8.5.3
**Type:** Implement per spec
**Severity:** LOW

Add 20K-edge benchmark test with yellow threshold at 80ms, complementing existing red (100ms) and green (50ms) gates.

**Acceptance:** `test_20k_symbols_under_80ms` exists and asserts <80ms.

---

## Batch 2: Concurrency Tests (Depends on Batch 1 Task 1)

### Task 4: Implement concurrent reindex + search test

**File:** `tests/test_search_with_ppr.py` (or new `tests/test_concurrent_ppr.py`)
**Assertions:** 6.3.1, 6.3.2, 6.3.3, 6.3.4, 8.3.2 (partial)
**Type:** Implement per spec (AC#4)
**Severity:** HIGH

Create test with:
- 1K+ symbol test project
- 10 concurrent search threads, each running 5 queries (50 total)
- 3 reindex cycles overlapping randomly
- Measure P95 search latency during reindex vs baseline (no reindex)
- Assert: no crashes, P95 delta <100ms, graph version updates atomically
- Verify threading.Lock is exercised (not just defined)

**Acceptance:** Test passes on CI. P95 latency delta <100ms. No exceptions in any thread.

---

### Task 5: Add search latency overhead benchmark

**File:** `tests/test_performance_ppr.py`
**Assertions:** 6.2.1, 8.5.4
**Type:** Implement per spec (AC#5)
**Severity:** MEDIUM

Benchmark end-to-end `hybrid_search()` latency with and without graph parameter. Assert overhead <50ms.

**Acceptance:** `test_search_latency_overhead_under_50ms` exists and passes.

---

## Batch 3: Spike Test Completion (Independent of Batches 1-2)

### Task 6: Expand spike test to 3 projects

**File:** `tests/test_spike_ppr.py`
**Assertions:** 8.4.2
**Type:** Complete partial implementation
**Severity:** HIGH

Index 2 additional real codebases beyond Tessera:
- PHP project (~1K symbols) or TypeScript project (~3K symbols)
- Measure graph density (edge_count / symbol_count) on each

**Acceptance:** Spike test indexes ≥2 projects total, reports density metrics for each.

---

### Task 7: Implement nDCG@5 metric computation

**File:** `tests/test_spike_ppr.py`
**Assertions:** 8.4.3, 8.4.4
**Type:** Implement per spec (Phase 5a gate)
**Severity:** HIGH

For each project:
1. Create 10+ representative queries (30% function-name, 30% domain-concept, 30% code-pattern, 10% known-negative)
2. Compute recall@5 and nDCG@5 for 2-way RRF baseline vs 3-way RRF with PPR
3. Report lift: `(3-way nDCG - 2-way nDCG) / 2-way nDCG`
4. Gate: ≥2 of 3 projects show >2% lift

**Acceptance:** nDCG@5 computed and reported in spike-results.md. Gate pass/fail documented.

---

### Task 8: Rename spike test file

**File:** `tests/test_spike_ppr.py` → `tests/test_spike_ppr_validation.py`
**Assertion:** 8.4.1
**Type:** Align with spec naming
**Severity:** LOW

Rename file to match spec convention. Update any imports or references.

**Acceptance:** File exists at `tests/test_spike_ppr_validation.py`.

---

## Batch 4: Polish (Depends on Batches 1-3)

### Task 9: Add NetworkX reference validation test

**File:** `tests/test_graph.py`
**Assertion:** 8.1.5
**Type:** Implement per spec (AC#7)
**Severity:** LOW

Add test that computes PPR on a star graph using both Tessera's implementation and NetworkX `pagerank_scipy()`. Assert scores match within 1e-5 tolerance.

**Acceptance:** Test imports networkx, computes reference PPR, asserts equivalence.

---

### Task 10: Enhance audit log for PPR tracking

**File:** `src/tessera/server.py`
**Assertion:** 4.5.5
**Type:** Complete partial implementation
**Severity:** LOW

Update `_log_audit()` calls in search tool to include whether PPR was used (boolean or rank_sources list).

**Acceptance:** Audit log entries distinguish PPR-enhanced vs keyword-only searches.

---

## Dependency Graph

```
Batch 1 (no deps):  Task 1, Task 2, Task 3
                        |
Batch 2 (after T1): Task 4, Task 5

Batch 3 (parallel): Task 6 → Task 7, Task 8

Batch 4 (after all): Task 9, Task 10
```

## Estimated Effort

| Batch | Tasks | Effort |
|-------|-------|--------|
| 1 | Fix bug, add log, add threshold | ~1 hour |
| 2 | Concurrent test, latency benchmark | ~3-4 hours |
| 3 | Spike expansion, nDCG metrics, rename | ~4-6 hours |
| 4 | NetworkX test, audit enhancement | ~1-2 hours |
| **Total** | **10 tasks** | **~9-13 hours** |

---

*Generated by /validate on 2026-02-28. Execute with `/execute docs/plans/phase5-ppr-graph/REMEDIATION-PLAN.md`*
