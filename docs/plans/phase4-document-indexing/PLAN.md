# Plan: Phase 4 — Document Indexing + Drift-Adapter + Per-Project Ignore Config

**Date:** 2026-02-27
**Tier:** Standard
**Status:** Conditional Go

## Executive Summary

Phase 4 extends Tessera's indexing pipeline to handle non-code documents (PDF, Markdown, YAML, JSON) alongside existing code intelligence. Three orthogonal modules are added:

1. **Document Indexing** — PDF extraction (PyMuPDF4LLM), Markdown splitting (header-based with hierarchy metadata), YAML/JSON parsing (structural key-path chunking). Documents flow through the same embedding and search pipeline as code, enabling unified `search()` across code + docs with RRF ranking.

2. **Drift-Adapter** — Orthogonal Procrustes transformation (scipy) for embedding model migration without re-indexing. Train on 1-5% corpus sample in <2 minutes; recover 95%+ retrieval recall. Infrastructure-grade tooling with manual operator workflow.

3. **Per-Project Ignore Config** — `.tesseraignore` file using `.gitignore` syntax (pathspec library). Two-tier system: security-critical patterns (`.env*`, `*.pem`, `*credentials*`, etc.) are un-negatable; standard defaults are user-overridable.

**Architecture**: Extend existing `chunk_meta` table with 5 nullable columns. Single FAISS index. No separate doc tables. Search tools: `search()` gets `source_type` filter; new `doc_search()` for document-only queries.

**Budget**: ~1,350 LOC production (within 1,500 cap). Tests: ~300 LOC separately.

---

## Specification

See: `spec-v2.md` (final version, incorporating panel feedback from 2 rounds)

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Schema | Extend chunk_meta (5 nullable columns) | Single table, single FAISS index, RRF works unchanged. No separate doc table. |
| PDF extraction | PyMuPDF4LLM v0.2.0+ | 0.12s/doc, markdown output with headers preserved. No OCR. |
| Markdown chunking | Header-based splitting | Preserves document structure. 1024-char chunks, 128-char overlap. |
| YAML/JSON | Structural key-path chunking | Parse to dict, extract sections. Better retrieval than monolithic. |
| Drift-Adapter | Orthogonal Procrustes (scipy) | Same-dimension rotation. <100ms training, <1us query overhead. |
| Ignore patterns | .tesseraignore + pathspec | .gitignore syntax. Two-tier: security (un-negatable) + user defaults. |
| FAISS filtering | Over-fetch 3x + post-filter | Avoids separate indices. Source_type filter applied after retrieval. |
| Search UX | Unified + dedicated | search() with source_type filter + doc_search() convenience tool |

### Module Structure

| Module | LOC | Purpose |
|--------|-----|---------|
| `document.py` | ~400 | PDF/Markdown/YAML/JSON extraction + chunking |
| `drift_adapter.py` | ~200 | Procrustes training + query transformation |
| `ignore.py` | ~120 | Two-tier .tesseraignore parsing |
| `db.py` updates | ~100 | Schema migration + nullable columns |
| `indexer.py` updates | ~300 | Document discovery + incremental re-index |
| `search.py` updates | ~150 | Over-fetch + post-filter + doc_search |
| `server.py` updates | ~80 | MCP tool registration (doc_search, drift_train) |
| **Total** | **~1,350** | Within 1,500 LOC cap |

### New Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| pymupdf4llm | >=0.2.0 | PDF text extraction to markdown |
| pathspec | >=0.12.0 | gitignore-style pattern matching |
| scipy | >=1.10.0 | Orthogonal Procrustes (scipy.linalg) |

---

## Conditions (Mandatory Before Merge)

### C1: asyncio.to_thread() for Blocking I/O (CRITICAL)

Use `asyncio.to_thread()` to wrap blocking PyMuPDF calls, not `asyncio.run()` which deadlocks in async contexts. Verify with unit test that calls `_index_document_file()` from an async context.

### C2: Glob-Aware Security Pattern Negation (IMPORTANT)

Security pattern negation check must use `pathspec.PathSpec.match_file()`, not string equality. Prevents bypass of `.env*` pattern via exact-match variants like `!.env.local`. Unit test required.

### C3: PDF Extraction <30s Gate Test (MASTER PLAN)

Integration test with real 50-page PDF verifying extraction + chunking + embedding in <30 seconds. This is the Phase 4 gate from the master architecture plan.

### C4: Over-Fetch Multiplier Validation (ARCHITECTURAL)

Empirically validate 3x over-fetch multiplier on skewed corpora (<<1% documents). If >5% of queries return insufficient results after filtering, increase to 5x. Document results in search.py comment.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| PDF quality varies by type | Medium | High | Fallback to base PyMuPDF. Test on representative PDFs. |
| Credential file indexing | Medium | Critical | Two-tier IgnoreFilter. SECURITY_PATTERNS un-negatable. |
| Prompt injection via docs | Medium | High | `trusted` field on search results. Agents treat docs as untrusted. |
| Procrustes insufficient for large drift | Low | High | Monitor recall. Escalate to Low-Rank Affine (Phase 5). |
| YAML/JSON edge cases | Medium | Medium | yaml.safe_load + json.loads. Error isolation. |
| Over-fetch 3x insufficient | Medium | Medium | Tunable constant. Validate empirically (C4). |
| asyncio deadlock | High (if uncaught) | Critical | Use asyncio.to_thread() (C1). |
| numpy load safety | Low | High | allow_pickle=False + shape/dtype validation. |

---

## Follow-up Items

- Phase 5: Graph edges from documents to code symbols
- Phase 5: Low-Rank Affine Drift-Adapter variant for large model drift
- Phase 5+: Automated drift-adapter triggering (currently manual operator workflow)
- Phase 6: Real-time document watching + incremental re-indexing
- Phase 6+: DOCX, Confluence, Slack format support
- Future: `--dry-run` flag for .tesseraignore validation
- Future: Disk footprint projections and eviction policy

---

<details>
<summary>Panel Scorecard</summary>

### Round 1

| Dimension | DA | Ops | SecA | Avg |
|-----------|-----|------|------|------|
| Problem-Sol Fit | 4 | 4 | 3 | 3.67 |
| Feasibility | 2 | 3 | 4 | 3.00 |
| Completeness | 3 | 2 | 2 | 2.33 |
| Risk Awareness | 2 | 2 | 2 | 2.00 |
| Clarity | 3 | 4 | 3 | 3.33 |
| Elegance | 3 | 4 | 4 | 3.67 |

**Key Issues Found:**
- Zero security risks in risk register (all 3)
- Drift-Adapter workflow undefined (DA, Ops)
- Incremental re-index ignores documents (Ops)
- FAISS source_type filtering gap (DA)
- Credential file blocklist missing (SecA)
- .tesseraignore negation bypass (SecA)

### Spec Revision v1 to v2

All issues addressed: security risk register, two-tier ignore, drift workflow, incremental doc re-index, FAISS over-fetch strategy, schema migration, error isolation, LOC budget clarification.

### Round 2

| Dimension | DA | Ops | SecA | Avg | Delta |
|-----------|-----|------|------|------|-------|
| Problem-Sol Fit | 4 | 4 | 4 | 4.00 | +0.33 |
| Feasibility | 4 | 4 | 4 | 4.00 | +1.00 |
| Completeness | 4 | 4 | 3 | 3.67 | +1.34 |
| Risk Awareness | 3 | 4 | 4 | 3.67 | +1.67 |
| Clarity | 4 | 4 | 4 | 4.00 | +0.67 |
| Elegance | 3 | 4 | 4 | 3.67 | +0.00 |

**Round 2 Average: 3.83/5** (up from 3.00). All dimensions >= 3.5. Converged (StdDev <= 0.47).

**Panel Recommendation:** Conditional Go

### Residual Items (Implementation-Level)

1. Over-fetch 3x may be insufficient for skewed corpora — validate empirically
2. Security pattern negation uses string equality — fix to glob matching
3. asyncio.run() deadlock in async context — use asyncio.to_thread()
4. Drift validation set too small at sample_size=50 — recommend default 200
5. drift_train() needs global scope gating — one-line addition

</details>

<details>
<summary>Executive Review</summary>

### CTO Review — Conditional Go

**Architecture Assessment:** Excellent fit with Phase 1-3. Unified chunk_meta, FAISS over-fetch, two-tier IgnoreFilter are all sound. Dependencies are low-risk. No architectural concerns.

**Technical Risk:** High-confidence mitigations for all major risks. All risks have rollback paths (fallback PyMuPDF, adjust multiplier, delete drift matrix). No silent failures expected.

**Four Mandatory Conditions:**
1. asyncio.to_thread() for blocking I/O (prevents deadlock)
2. Glob-aware security pattern negation (prevents bypass)
3. PDF extraction <30s acceptance test (master plan gate)
4. Over-fetch multiplier empirical validation (architectural risk)

**Implementation Guidance:**
- Mark 3x multiplier as tunable constant (`SEMANTIC_SEARCH_OVER_FETCH_MULTIPLIER`)
- Scope Drift-Adapter hard — minimal, no auto-triggering
- Test incrementally (unit tests per module before integration)
- Monitor thread pool behavior under concurrent document indexing

</details>
