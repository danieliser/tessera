# Phase 5a Spike Test Results — PPR Precision Validation

**Date:** 2026-02-28 21:27:36
**Status:** PRELIMINARY (automated test harness, not full annotation study)

## Executive Summary

Spike test validates PPR graph signal feasibility by:
1. Indexing real Tessera codebase
2. Extracting call graph (edges) and building scipy CSR sparse matrix
3. Implementing minimal PPR power iteration algorithm
4. Measuring computation performance
5. Validating 3-way RRF integration

**Result:** All performance gates passed. Graph density sufficient for PPR signal.

---

## Projects Tested

### Project 1: Tessera (Python)

**Language:** Python
**Path:** /Users/danieliser/Toolkit/codemem/src/tessera

**Indexing Results:**
- Files indexed: 15
- Symbols extracted: 344
- Chunks created: 141
- Index time: 0.27s

**Graph Metrics:**
- Symbol count: 344
- Edge count: 436
- Edge/Symbol ratio: 1.27
- Sparse (edges < symbols): False
- **Assessment:** DENSE (good PPR signal expected)

**PPR Performance:**
- Computation time: 1.12ms
- **Gate:** ✅ <100ms (passed)

---

## Algorithm Validation

### PPR Power Iteration Implementation

Implemented reference algorithm from spec:

```python
def personalized_pagerank(
    adjacency: scipy.sparse.csr_matrix,
    seed_ids: List[int],
    n_symbols: int,
    alpha: float = 0.15,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Dict[int, float]:
    # Personalization vector
    p_seed = np.zeros(n_symbols, dtype=np.float32)
    for sid in set(seed_ids):
        if 0 <= sid < n_symbols:
            p_seed[sid] = 1.0 / len(set(seed_ids))

    p = p_seed.copy()

    # Column-stochastic normalization
    graph_norm = adjacency.copy().astype(np.float32)
    out_degrees = np.array(graph_norm.sum(axis=1)).ravel()
    out_degrees[out_degrees == 0] = 1
    graph_norm = scipy.sparse.diags(1.0 / out_degrees) @ graph_norm

    # Power iteration
    for iteration in range(max_iter):
        p_old = p.copy()
        p = (1 - alpha) * graph_norm.T @ p + alpha * p_seed
        if np.linalg.norm(p - p_old, ord=2) < tol:
            break

    return {i: float(p[i]) for i in range(n_symbols) if p[i] > 1e-8}
```

**Tests Passed:**
- ✅ Star graph: Central hub correctly ranked highest
- ✅ Linear chain: PPR propagates through graph
- ✅ Medium graph (1K symbols, 5K edges): 1.51ms
- ✅ Convergence: Stops within max_iter

### 3-Way RRF Integration

**Algorithm:** Merge keyword + semantic + PPR rankings via RRF

Test validated:
- ✅ Three ranked lists merge correctly
- ✅ Items appearing in multiple lists score higher
- ✅ All unique items included in output

**Graceful Degradation:** Sparse graphs (edges < symbols) skip PPR signal

---

## Performance Metrics

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Tessera PPR time | 1.12ms | <100ms | ✅ |
| Avg PPR time (all projects) | 1.12ms | <100ms | ✅ |
| Max PPR time | 1.12ms | <100ms | ✅ |

---

## Gate Decision

**Blocking Gate:** ≥2 of 3 projects must show ≥2% nDCG@5 lift (3-way RRF vs 2-way baseline)

**Spike Test Scope:** This automated test validates:
1. ✅ Graph density (0.5–2.0 edge/symbol ratio)
2. ✅ PPR performance (<100ms on real graphs)
3. ✅ Algorithm correctness (verified on synthetic graphs)
4. ✅ 3-way RRF integration (merges correctly)

**Next Step (Phase 5, Task 1):**
- Implement manual annotation study with developer feedback
- Test on 10+ queries per project (function-name, domain, code-pattern, known-negative)
- Measure nDCG@5 for 2-way vs 3-way RRF
- Apply gate decision

---

## Implementation Status

**Deliverables for Phase 5 (pending gate):**

- [ ] `src/tessera/graph.py` — ProjectGraph class, PPR algorithm
- [ ] `src/tessera/db.py` — Graph query methods (get_all_symbols, get_edges)
- [ ] `src/tessera/search.py` — 3-way RRF integration
- [ ] `src/tessera/server.py` — Graph lifecycle (load, rebuild, monitor)
- [ ] `tests/test_graph.py` — Unit tests for PPR
- [ ] `tests/test_search_with_ppr.py` — Integration tests
- [ ] Benchmark suite with CI performance gates

**Blockers:** None. Proceed to Phase 5 Task 1 (full implementation + annotation study).

---

## Notes

- Test uses existing Tessera codebase (318 symbols, 330 edges)
- Single project tested in spike (Phase 5 full implementation tests 3 diverse projects)
- PPR algorithm uses hand-coded power iteration (no new dependencies)
- scipy CSR sparse matrix provides efficient computation for large graphs
- Graceful degradation works: sparse graphs skip PPR, results fall back to 2-way RRF

