# Executive Review: CTO

## Decision: Conditional Go

Approve Phase 5 for implementation, with Phase 5a spike test as blocking gate. The architecture is sound, dependencies are managed, and risk discipline is good. However, three specification gaps identified by the panel must be resolved before Phase 5a execution to prevent test interpretation drift and scope confusion.

---

## Rationale

**Architectural Soundness:**
PPR via scipy.sparse CSR matrix + hand-coded power iteration is a proven, mathematically sound approach. HippoRAG 2 and Aider validate this on production code graphs. The expected 3–7% precision lift is realistic for symbol-level graphs. Using scipy (already a Phase 4 dependency) eliminates new supply-chain risk.

**Integration and Coupling:**
The design cleanly integrates with existing patterns. RRF merge in search.py is extended from 2-way to 3-way—straightforward. Server lifecycle (load at startup, rebuild on reindex) is well-scoped. No invasive changes to db.py or core indexing logic. Graph module is self-contained (~300 LOC), reducing merge/test surface.

**Risk Management:**
Phase 5a spike test is the right gating mechanism. Validating ≥2% precision lift on 2+ real projects before committing full implementation shows maturity. The three-project sample (PHP, Python, JavaScript) spans language and complexity ranges. Performance gates (<100ms PPR, <50ms search overhead) are measurable and CI-gated.

**Capacity Alignment:**
~1.5 days for spike, ~2.5 days for main tasks fits within a 1-week sprint. Effort estimates are realistic given the modular scope.

**Concerns:**
The panel flagged four remaining gaps, two of which are resolvable specifications (spike protocol, AC#4 metric), one is a spec contradiction (startup blocking), and one is a deferred architectural decision (LRU memory cap). None are fatal, but they must be addressed before Phase 5a execution to avoid test drift and to clarify the memory management contract.

---

## Conditions

1. **Spike Test Protocol Document**
   Write `/Users/danieliser/Toolkit/codemem/docs/plans/phase5-ppr-graph/spike-protocol.md` before Phase 5a starts. Must specify:
   - Query selection: How are the 10+ queries per project chosen? (Random sample from codebase? Stratified by taxonomy in spec section 1007–1010? Curated by developer?)
   - Annotation instructions: Define "relevant" (exact wording for annotator) and process for marginal cases (e.g., "partially relevant").
   - Single annotator rationale: Confirm that one developer familiar with each project is acceptable for spike validation (not full inter-rater agreement study). State confidence bounds for precision lift measurement (e.g., ±1.5% margin given single annotator).
   - Tie-breaking: How to handle queries where 2-way and 3-way RRF have identical top-5 results (count as 0% lift for that query, or skip?).

2. **Clarify Startup Blocking Behavior**
   Resolve contradiction between spec sections 73 (synchronous load at create_server() startup) and 963 (load in background thread). Decide:
   - **Option A (Recommended):** Synchronous load at startup. First MCP session blocks until all graphs loaded. Acceptable for Tessera's deployment model (agent starts, waits ~1–5 sec for index, then runs queries).
   - **Option B:** Lazy load first graph on first search; others load asynchronously in background.

   Document decision in spec section 73 (Graph Lifecycle) with explicit tradeoff: blocking startup time vs. non-blocking first-query latency variance.

3. **Define LRU Eviction Scope for Phase 5 vs Phase 6**
   Spec says "LRU still Phase 6" but adds monitoring in Phase 5. Clarify:
   - Phase 5 deliverable: Memory monitoring (logging + warnings at >500MB), NOT eviction logic.
   - Phase 6 deliverable: LRU eviction (simple dict with MAX_CACHED_GRAPHS=20, plus configurable thresholds).
   - If monitoring detects >500MB during Phase 5 on real deployments, Phase 5 can add emergency fallback (e.g., disable PPR if total memory >1GB), documented as unsupported configuration.

4. **AC#4 Latency Metric Precision**
   Confirm that AC#4 test (concurrent reindex + search) measures:
   - **Baseline:** P95 search latency with no reindex (measure 50 queries before any reindex starts).
   - **Under Load:** P95 search latency measured during 3 concurrent reindex cycles (measure 50 queries overlapping reindex).
   - **Gate:** P95_under_load - P95_baseline < 100ms.

   Add assertion to test: `assert p95_with_reindex - p95_baseline < 100, f"Latency delta {p95_with_reindex - p95_baseline}ms exceeds 100ms gate"`. Document this in test file `/Users/danieliser/Toolkit/codemem/tests/test_server_concurrent.py` or update existing concurrency test.

---

## Risks You Accept

1. **Spike Test May Show <2% Lift**
   If Phase 5a shows <2% nDCG improvement on ≥2 projects, full Phase 5 implementation is deferred to Phase 6. This is intentional risk acceptance—better to learn early than commit 2.5 days. Recovery plan is documented (focus on other Phase 5 features, revisit PPR with finer-grained graphs in Phase 6).

2. **Synchronous Startup Blocking**
   If startup time >5 sec on large projects (100K+ edges), this may be perceived as unresponsive. Mitigated by per-project timing logs (identify which projects are slow) and fallback to lazy loading in Phase 6.

3. **Graph Density on Symbol vs File-Level Graphs**
   Aider validates PPR on file-level call graphs, not symbol-level. Tree-sitter symbol extraction may produce sparser graphs (fewer edges). Spike test validates this empirically; if sparse, `is_sparse_fallback()` gracefully degrades.

4. **Concurrent Reindex Race Condition (GIL Reliance)**
   Spec relies on Python GIL for atomic dict assignment of `_project_graphs[project_id]`. While safe in practice (GIL serializes dict updates), explicit `threading.Lock` is deferred to Phase 6 if AC#4 test detects instability.

---

## Risks You Don't Accept

1. **No New Dependencies** — Conditional Go depends on using scipy (already required). If phase 5 code requires new external libs (e.g., NetworkX instead of scipy), escalate to CTO for re-evaluation.

2. **Code Budget Overrun** — Phase 5 budgeted at ~790 LOC. If implementation approaches >1,000 LOC, escalate. (Task 1 shows 350 LOC for graph.py, Task 4 shows 250 LOC for server.py, totals ~840 LOC including benchmarks and tests—within budget.)

3. **Performance Gates** — PPR must stay <100ms on 50K edge graphs (CI-gated in Task 1). Three-way RRF search overhead must be <50ms (benchmark in Task 1). If benchmarks fail, root-cause and optimize (e.g., switch to fast-pagerank library if scipy CSR too slow). Do not ship if gates fail.

4. **Startup Latency Regression** — If server startup time increases >2 sec on single-project deployments due to graph loading, that's a regression. Document baseline in Task 4, assert regression gate in tests.

---

## Key Assumptions Validated

- **Graph density sufficient:** Spike test gates Phase 5 on ≥2% precision lift empirically measured.
- **scipy CSR performance:** Benchmarks (Task 1) validate <100ms on 50K edges; if exceeded, switch libraries.
- **RRF merging:** Three-way RRF is backward-compatible (optional graph parameter). Existing 2-way queries unaffected.
- **Memory budget:** 8 bytes/edge formula is conservative (CSR actual overhead lower). Phase 5 monitoring validates; Phase 6 adds eviction if needed.

---

## Implementation Checkpoints

- **Checkpoint 1 (Day 0):** Write spike-protocol.md, clarify startup blocking, finalize AC#4 metric. (2 hours)
- **Checkpoint 2 (Day 1):** Phase 5a spike test completes. Gate decision: proceed to Phase 5 or defer to Phase 6. (1 day)
- **Checkpoint 3 (Day 3):** Task 1 (graph.py + benchmarks) + Task 2 (db.py) complete. Benchmark suite CI-gated. (1.5 days)
- **Checkpoint 4 (Day 5):** Tasks 3–4 (search.py, server.py) complete. AC#4 concurrent reindex test passes. (2 days)
- **Checkpoint 5 (Day 6):** Tasks 5–6 (monitoring, docs, cleanup). All tests pass. Phase 5 complete. (1 day)

---

## Final Assessment

**Go forward with Phase 5a immediately.** The spike test is well-designed and will answer the core question (does PPR add meaningful ranking signal?) in one day. The three specification gaps are resolvable clarifications, not architectural blockers. Once resolved and spike test passes, Phase 5 implementation is low-risk and high-confidence.

The team has demonstrated strong risk discipline and spec maturity. Conditional approval reflects this maturity—the conditions are administrative clarity, not technical red flags.
