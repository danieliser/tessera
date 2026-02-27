# CTO Executive Review — Phase 4: Document Indexing + Drift-Adapter

**Date:** 2026-02-27
**Reviewer:** CTO
**Recommendation:** Conditional Go

---

## Decision: Conditional Go

Phase 4 is architecturally sound and ready for implementation with **four mandatory conditions** that address latent technical risks. The spec v2 incorporates panel feedback well. Proceed after conditions are documented and gated.

---

## Architecture Assessment

### Design Fit

Phase 4 cleanly extends the existing unified search pipeline. Key architectural decisions are all sound:

1. **Unified chunk_meta table** — Adding `source_type`, `section_heading`, `key_path`, `page_number`, `parent_section` as nullable columns is the right call. Avoids separate indices/schemas and keeps RRF merge simple. The alternative (separate `doc_chunks` table) was correctly deferred to Phase 5+.

2. **FAISS over-fetch + post-filter strategy** — The panel's concern about this was valid. Fetching 3x limit and discarding non-matching source_type is pragmatic but creates an implicit assumption: that document and code chunks are roughly uniformly distributed in the FAISS index. This holds for most projects but could fail catastrophically if documents are <<1% of the corpus (fetch 30 items, post-filter to 0 documents). I'm conditioning on empirical validation.

3. **Unified search with source_type filter** — Consistent with Phase 3's RRF pattern. The `doc_search()` convenience tool is appropriate for agents that want document-only results without constructing source_type lists.

4. **Schema migration strategy** — Using `_meta` table with schema_version + conditional ALTER TABLE wrapped in try/except is correct. The spec handles the "already exists" case explicitly, which is good defensive coding.

### Dependencies

Three new libraries, all low-risk:

- **pymupdf4llm** (0.2.0+) — Actively maintained, narrow scope (PDF text extraction). The research validated 0.12s/doc performance. No native code dependencies beyond PyMuPDF itself.
- **pathspec** (0.12.0+) — Stable gitignore-matching library. Used by pre-commit, pip, and others. Industry standard.
- **scipy** (1.10.0+) — Orthogonal Procrustes via scipy.linalg is a single, well-tested function. No risk of scope creep.

No heavy frameworks (pytorch, tensorflow, ML systems). **Dependency risk: LOW**.

---

## Technical Risk Assessment

### High-Confidence Mitigations

1. **PDF extraction latency** — The 22s target for a 50-page PDF (6s PyMuPDF4LLM + 1s markdown chunking + 15s embedding) is achievable and validated by research. Acceptance test AC-D1 is non-negotiable.

2. **Security: credential file blocking** — Two-tier IgnoreFilter (SECURITY_PATTERNS un-negatable + user defaults) is the right pattern. The spec explicitly lists patterns (`.env*`, `*.pem`, `*.key`, `*credentials*`, `id_rsa`, `*.token`, `service-account.json`, etc.). The implementation correctly prevents user override of these.

3. **Drift-Adapter validation** — Using `np.load(..., allow_pickle=False)` + shape/dtype checks on load is correct defensive programming (explicitly rejects pickled objects to prevent code execution). The EMNLP 2025 paper (peer-reviewed) validates 95-99% recall recovery. No hidden risks here.

4. **Schema backward compatibility** — Migration handles "column already exists" gracefully. Phase 3 databases will migrate cleanly on open.

### Medium-Confidence Risks (Conditional)

1. **Over-fetch multiplier for skewed corpora** — The 3x strategy assumes reasonable distribution. For projects where documents are <1-2% of corpus, a 50-item fetch with post-filtering could return 0-1 results frequently. The panel flagged this; I'm conditioning on empirical validation against representative corpora before release.

2. **asyncio.run() inside async context** — The panel noted that `_index_document_file()` calling `asyncio.run(extract_pdf(...))` will deadlock if called from an async context. The spec shows `asyncio.run()` in pseudocode; the implementation **must** use `asyncio.to_thread()` instead to wrap blocking I/O. This is critical.

3. **Security pattern negation check** — The spec shows string equality: `if negated_pattern == sp for sp in SECURITY_PATTERNS`. This is fragile. `.env*` won't match `.env_staging` if the user tries `!.env_staging`. Using `pathspec` for glob matching here is more robust.

### Low-Confidence Risks

1. **Drift-Adapter infrastructure shipping before proven at scale** — The spec explicitly marks this as "infrastructure-grade tooling" with manual ops workflows. It's not automated, not user-facing. If it doesn't work in the field, ops can delete the drift_matrix.npy file and fall back to raw embeddings. Acceptable.

2. **YAML/JSON edge cases** — Using `yaml.safe_load()` and `json.loads()` is correct. Edge cases (nested arrays, circular references) are unlikely to break indexing because the spec wraps extraction in try/except with error isolation.

---

## Panel Review Assessment

### What the Panel Got Right

1. **Security risk register (now v2)** — The panel correctly identified zero security risks in v1. V2 adds explicit sections on credential blocking, prompt injection (`trusted` field), array validation with `allow_pickle=False`, and yaml.safe_load. This is comprehensive.

2. **Drift-Adapter operational workflow** — V1 left workflow undefined. V2 now specifies startup auto-loading, `drift_train()` MCP tool, validation loop, and rollback. Panel's push was justified.

3. **Incremental re-indexing for documents** — V1 ignored document extensions in `_get_changed_files()`. V2 now handles `.pdf`, `.md`, `.yaml`, `.yml`, `.json`. Correct catch.

4. **Over-fetch + post-filter strategy** — The panel flagged the FAISS gap; v2 now specifies fetch 3x limit + post-filter explicitly in search.py updates section. Honest about the tradeoff.

5. **LOC budget clarification** — V1 was ambiguous. V2 clearly separates production (1,350 LOC) from tests (300 LOC). Good.

### What Needs Implementation Attention

The panel listed five residual items. I'm filtering on architectural vs. implementation significance:

1. **Over-fetch 3x insufficient for skewed corpora** — *Architectural risk.* Conditioning on this (see below).

2. **Security pattern negation uses string equality, not glob matching** — *Implementation detail, but subtle.* Conditioning on this (see below).

3. **asyncio.run() in async context will deadlock** — *Critical bug, not caught.* Conditioning on this (see below).

4. **Drift-Adapter validation set too small (default sample_size=50)** — *Implementation tuning.* The spec sets `sample_size: int = 50` with a note "optional." I'll recommend 200+ in conditions but don't hard-gate on this (ops can tune).

5. **drift_train() needs global scope gating** — *Security detail.* One-line addition to server.py. Low risk to overlook, but should be checked in code review.

**Panel did good work.** The residual items are implementation-level catches, not architectural showstoppers.

---

## Conditions

**All conditions are mandatory gates. Implementation team must verify before merge to main.**

### 1. asyncio.run() → asyncio.to_thread() Wrapping (CRITICAL)

**Spec section:** `_index_document_file()` in indexer.py updates.

**Current risk:** If the indexer is called from an async context (which it will be during server operation), calling `asyncio.run(extract_pdf(...))` will raise `RuntimeError: asyncio.run() cannot be called from a running event loop`.

**Requirement:**
```python
# WRONG (in spec pseudocode)
markdown_text = asyncio.run(extract_pdf(file_path))

# CORRECT (implementation)
markdown_text = await asyncio.to_thread(extract_pdf, file_path)
```

Document the pattern in all async orchestration calls. Verify with unit tests that block synchronous I/O operations.

**Verification:** Run unit test that calls `_index_document_file()` from an async context (e.g., within a pytest-asyncio test). Failure = deadlock. Success = extraction completes without RuntimeError.

**Gate:** Code review + test pass before merge.

---

### 2. Security Pattern Negation Using Glob Matching (IMPORTANT)

**Spec section:** IgnoreFilter.load() in ignore.py.

**Current risk:** String equality check `if negated_pattern == sp for sp in SECURITY_PATTERNS` doesn't account for glob patterns. A user attempting `!.env_*` in `.tesseraignore` won't match the security pattern `.env*` (different syntax). This allows accidental bypass.

**Requirement:**

```python
# Current (fragile)
if any(negated_pattern == sp for sp in self.SECURITY_PATTERNS):
    logger.warning(...)
    continue

# Correct (glob-aware)
import pathspec

security_spec = pathspec.PathSpec.from_lines('gitwildmatch', self.SECURITY_PATTERNS)
if security_spec.match_file(negated_pattern):
    logger.warning(f"Attempted to negate security pattern matching '{negated_pattern}'")
    continue
```

This uses pathspec consistently with the rest of the ignore system and prevents glob-pattern bypass.

**Verification:** Write unit test that attempts to negate `.env_staging`, `*.pem`, etc. with glob variants. Verify warning is logged and negation is ignored.

**Gate:** Code review + test pass before merge.

---

### 3. PDF Extraction <30s Acceptance Test (MASTER PLAN GATE)

**Spec section:** AC-D1 in acceptance criteria.

**Current risk:** None. The research validated 22s on a 50-page PDF. But this is the Phase 4 gate condition from the master architecture plan (`docs/plans/architecture/PLAN.md`), so it **must** be tested before merge.

**Requirement:**

Create integration test `tests/test_phase4_performance.py`:
```python
async def test_pdf_extraction_50page_under_30s():
    """AC-D1: Extract + chunk + embed 50-page PDF in <30s."""
    # Use a real 50-page PDF from tests/fixtures/
    # Measure total time: extract + chunking + embedding
    # Assert: elapsed < 30.0 seconds
```

Run this test on CI before merge. Document the PDF source and platform (CPU model, RAM) where test was validated.

**Verification:** CI pass on Phase 4 branch.

**Gate:** Test passes on main before Phase 4 release.

---

### 4. Over-Fetch Multiplier Validation (SKEWED CORPUS TEST)

**Spec section:** semantic_search() in search.py updates, FAISS over-fetch + post-filter strategy.

**Current risk:** The 3x fetch multiplier works for typical projects but could fail if documents are <<1% of corpus. Need empirical validation that 3x is sufficient for diverse corpora.

**Requirement:**

Before final merge, run over-fetch validation test on at least two representative corpora:
1. **Well-balanced corpus** — Code-heavy project with ~10% documents (e.g., WordPress plugin with README, config YAML)
2. **Skewed corpus** — Code-heavy with <<1% documents (e.g., pure Python package with only setup.md)

For each:
- Index the corpus
- Run `search(query, source_type=['markdown'], limit=10)` on 20 representative queries
- Measure: how many times did post-filtering return <10 results after 3x fetch?
- If >5% of queries fail (return <10 doc results when docs exist), recommend increasing multiplier to 5x

**Verification:** Document results in a comment in search.py. Acceptable outcomes:
- "Validated on WordPress plugin (12% docs) and Python package (0.8% docs). 3x multiplier sufficient for both."
- "Validated on WordPress plugin. Recommend increasing to 5x for <1% document projects." (then adjust code)

**Gate:** Test results documented before merge.

---

### 5. Drift-Adapter Default Sample Size Recommendation (OPTIONAL IMPLEMENTATION NOTE)

**Spec section:** `drift_train(sample_size: int = 50)` in server.py.

**Note:** The panel flagged that 50 samples might be too small for validation set (10% holdout = 5 items). While this is tunable at runtime, the **default should be higher for ops confidence**.

**Recommendation (not a gate):**
- Default `sample_size: int = 200` instead of 50 (provides 20-item validation set)
- Document that operators can override lower for quick testing, but production training should use 200+

This is a tunability note, not a blocker. Code review should verify.

---

## Implementation Guidance

### CTO-Level Direction

1. **Test incrementally:** Phase 4 adds complexity (new modules, schema migration, document I/O). Write unit tests for each module independently before integration. Don't defer testing to the end.

2. **Monitor thread pool behavior:** Using `asyncio.to_thread()` for PDF extraction means blocking I/O happens on a thread pool. Verify this doesn't exhaust thread pool under concurrent indexing (e.g., 10+ concurrent documents). Monitor with `concurrent.futures.ThreadPoolExecutor` metrics.

3. **Scope the Drift-Adapter hard:** The spec says "infrastructure-grade tooling" — that's accurate. Keep the `drift_train()` implementation minimal (sample, train, save, report). Don't add auto-triggering, cron scheduling, or state machines. That's Phase 5+. If ops ask for more, that's a separate feature request.

4. **Document the ignore system clearly:** The two-tier pattern (security un-negatable + user-overridable) is powerful but confusing. Add a `.tesseraignore` example file to the repo showing how it works.

5. **Mark the 3x multiplier as tunable:** Add a configuration constant at the top of search.py:
   ```python
   SEMANTIC_SEARCH_OVER_FETCH_MULTIPLIER = 3  # Tunable per Phase 5 requirements
   ```
   Then use it in `semantic_search()`. This signals that it's a known tradeoff, not set-and-forget.

### Risk Mitigation Strategy

- **If PDF extraction degrades in the field:** Fallback plan is documented (base PyMuPDF instead of PyMuPDF4LLM). No architecture change needed.
- **If over-fetch proves insufficient:** Adjust multiplier to 4x or 5x. No architecture change.
- **If Drift-Adapter fails in ops:** Delete `.tessera/data/{slug}/drift_matrix.npy` and restart. Falls back to raw embeddings. No architecture change.

All risks have rollback paths. **No silent failures expected.**

---

## Concerns for the Panel

The panel's Round 2 work was solid. No concerns to escalate back.

---

## Kill Justification

Not applicable. Phase 4 is a **Go** with conditions.

---

## Summary

**Phase 4 is technically sound.** The spec v2 incorporates panel feedback. The architecture is clean (unified chunk_meta table, FAISS over-fetch, two-tier IgnoreFilter). Dependencies are low-risk. Security is well-considered.

**Four mandatory conditions** protect against latent risks that are subtle enough to slip through code review:
1. asyncio deadlock from asyncio.run() in async context
2. Glob-aware security pattern negation (prevent bypass of security patterns)
3. Acceptance test for <30s PDF extraction (master plan gate)
4. Empirical validation of 3x over-fetch multiplier

**Proceed to implementation** after conditions are gated and documented.

---

**CTO Lens: Technical Leadership Assessment**

- **Architectural fit:** Excellent. Extends Phase 1-3 patterns cleanly.
- **Risk acceptance:** High-confidence. All major risks have mitigations and rollback paths.
- **Code quality expectations:** Medium-to-high bar. New modules (document.py, drift_adapter.py, ignore.py) should be well-tested given their criticality to indexing pipeline.
- **Operational load:** Low-to-medium. Drift-Adapter is opt-in ops tooling. Document chunking and ignore filtering are transparent to users.

**Recommendation: Conditional Go.** Implement with the four conditions above. Ship to main on condition verification.
