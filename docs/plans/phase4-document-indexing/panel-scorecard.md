# Panel Scorecard — Phase 4: Document Indexing + Drift-Adapter

**Date:** 2026-02-27
**Tier:** Standard
**Panelists:** Devil's Advocate, Ops Pragmatist, Security Analyst

---

## Score Matrix — Round 1

| Dimension | DA | Ops | SecA | Avg | StdDev | Flag |
|-----------|-----|------|------|------|--------|------|
| Problem-Sol Fit | 4 | 4 | 3 | 3.67 | 0.47 | |
| Feasibility | 2 | 3 | 4 | 3.00 | 0.82 | |
| Completeness | 3 | 2 | 2 | 2.33 | 0.47 | |
| Risk Awareness | 2 | 2 | 2 | 2.00 | 0.00 | Unanimous low |
| Clarity | 3 | 4 | 3 | 3.33 | 0.47 | |
| Elegance | 3 | 4 | 4 | 3.67 | 0.47 | |

**Round 1 Average: 3.00/5**

### Round 1 Consensus Concerns
1. Zero security risks in risk register (all 3 panelists)
2. Drift-Adapter operational workflow undefined (DA, Ops)
3. Incremental re-indexing ignores documents (Ops)
4. LOC budget ambiguity — tests included or not? (DA)

### Round 1 Critical Lone Dissents
- FAISS source_type filtering gap (DA) — architectural
- `.tesseraignore` negation bypasses security defaults (SecA)
- `np.load()` deserialization risk (SecA)

---

## Spec Revision: v1 to v2

All consensus concerns + critical lone dissents addressed:
1. Security risk register added (credential blocklist, prompt injection, safe numpy loading, yaml.safe_load)
2. Two-tier IgnoreFilter (SECURITY_PATTERNS un-negatable)
3. Drift-Adapter operational workflow (auto-load, drift_train() MCP tool, rollback)
4. Incremental re-indexing extended for document extensions
5. FAISS over-fetch + post-filter strategy (3x multiplier)
6. Schema migration via _meta version table
7. Error isolation in _index_document_file
8. LOC budget clarified (1,500 = production only, tests separate)

---

## Score Matrix — Round 2

| Dimension | DA | Ops | SecA | Avg | StdDev | R1 to R2 |
|-----------|-----|------|------|------|--------|-------|
| Problem-Sol Fit | 4 | 4 | 4 | 4.00 | 0.00 | +0.33 |
| Feasibility | 4 | 4 | 4 | 4.00 | 0.00 | +1.00 |
| Completeness | 4 | 4 | 3 | 3.67 | 0.47 | +1.34 |
| Risk Awareness | 3 | 4 | 4 | 3.67 | 0.47 | +1.67 |
| Clarity | 4 | 4 | 4 | 4.00 | 0.00 | +0.67 |
| Elegance | 3 | 4 | 4 | 3.67 | 0.47 | +0.00 |

**Round 2 Average: 3.83/5** (up from 3.00)

### Weighted Scores (Infrastructure: 1.5x on Risk Awareness + Feasibility)

| Dimension | Weight | Weighted Avg |
|-----------|--------|-------------|
| Problem-Sol Fit | 1.0x | 4.00 |
| Feasibility | 1.5x | 6.00 |
| Completeness | 1.0x | 3.67 |
| Risk Awareness | 1.5x | 5.50 |
| Clarity | 1.0x | 4.00 |
| Elegance | 1.0x | 3.67 |

### Convergence
- All StdDev <= 0.47
- All averages >= 3.5
- No individual scores <= 2

---

## Advancement: PASS

All dimensions >= 3.5 average. Auto-advance to exec review.

---

## Residual Items for Implementation

1. Over-fetch 3x insufficient for skewed corpora (doc/code ratio <1%) — use adaptive multiplier or document limitation
2. Security pattern negation check uses string equality, not glob matching — fix to use pathspec
3. `asyncio.run()` in async context will deadlock — use `asyncio.to_thread()` wrapping sync call
4. Drift-Adapter validation set too small at default sample_size=50 — raise default to 200+ or document
5. `drift_train()` needs global scope gating — one-line addition

---

## Panel Recommendation

**Conditional Go** — proceed to exec review. Five implementation-level fixes noted above.
