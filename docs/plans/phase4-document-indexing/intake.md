# Phase 4: Document Indexing + Drift-Adapter — Intake

**Date:** 2026-02-26
**Tier:** Standard
**Project Type:** Infrastructure
**Slug:** phase4-document-indexing

---

## Idea

Extend Tessera's indexing pipeline to support non-code documents (PDF, Markdown, YAML, JSON) alongside existing code intelligence. Add Drift-Adapter for embedding model migration. Add per-project ignore config (`.tesseraignore`).

This is Phase 4 of the 6-phase master architecture plan (`docs/plans/architecture/PLAN.md`).

---

## Question Framing

### The Right Question
"What's the minimum viable document indexing that integrates cleanly with Tessera's existing search/federation pipeline, and how do we handle per-project file exclusion at the same time?"

### Adjacent Questions
- Should document chunks participate in the graph (edges from README → module it describes)?
- The original spec assumed LanceDB — we use FAISS+SQLite. Document chunk schema needs redesign for the actual stack.
- Should document search federate across collections the same way code search does?

### Wrong Questions to Avoid
- "Which PDF library?" — pymupdf is the clear winner. Don't bikeshed.
- "Should we support DOCX/Confluence?" — Explicitly out of scope per spec. Extensible later.

### Hidden Assumptions (Addressed)
- LanceDB schema is stale → will redesign for FAISS+SQLite
- Embeddings are optional → structural-only (FTS) document search should work without embedding endpoint
- Drift-Adapter assumes heavy embedding usage → shipping anyway per user decision, as infrastructure for future phases

---

## Premise Validation

### PyMuPDF Performance
- Benchmarked at ~0.1s per document, 7,031 pages in test suite
- pymupdf4llm variant optimized for markdown output
- No OCR needed — spec explicitly excludes scan-to-PDF
- **Premise: VALID** — 50-page PDF in <30s is trivially achievable

### Drift-Adapter Research
- Published at EMNLP 2025 (ACL Anthology)
- Three parameterizations: Orthogonal Procrustes, Low-Rank Affine, Residual MLP
- Recovers 95-99% retrieval recall on MTEB corpora
- <10μs query latency overhead, >100× cheaper than full re-index
- Training on 5% sample in <2 minutes
- **Premise: VALID** — peer-reviewed, implementation sketch in spec

### Competitive Landscape
- MCP Document Indexer (yairwein): PDF/Word/MD with Ollama — no code intelligence, no scope gating
- MCP-Markdown-RAG: Milvus-backed markdown search — no multi-format, no federation
- Code-Index-MCP: Code indexing only, no documents
- **Gap: VALID** — no existing MCP server combines code intelligence + document search + scope gating + federation

---

## Scope Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Drift-Adapter inclusion | Yes, ship with Phase 4 | User decision. Infrastructure for future embedding model changes. |
| Per-project ignore config | Yes, `.tesseraignore` | Natural fit — documents need exclusion patterns too. Addresses `bin/stubs` issue from Phase 3 validation. |
| Search UX | Both unified + dedicated | `search()` returns code+docs with `source_type` filter. New `doc_search()` for document-only queries. |
| Document formats | PDF, Markdown, YAML, JSON | Per original spec. Extensible later. Not DOCX/Confluence. |

---

## Constraints

- **Line budget:** <1,500 LOC (cumulative for Phase 4)
- **No external servers:** Must use embedded libraries only (pymupdf, numpy for Procrustes)
- **Schema compatibility:** Must work with existing FAISS+SQLite stack, not LanceDB
- **Federation:** Document search must federate across collections like code search
- **Embeddings optional:** FTS-only document search must work without embedding endpoint

## Success Criteria

1. Extract + index 50-page PDF in <30 seconds
2. `search()` returns mixed code + document results ranked via RRF
3. `doc_search()` returns document-only results
4. Per-project `.tesseraignore` excludes specified paths/patterns from indexing
5. Drift-Adapter trains on 5% corpus sample in <2 minutes, recovers 95%+ retrieval performance
6. Document search federates across collections

## Non-Goals

- OCR / scan-to-PDF support
- DOCX, Confluence, Slack, or other proprietary formats
- Document version control / history tracking
- Graph edges from documents to code symbols (defer to Phase 5)
- Real-time document watching (defer to Phase 6)

---

## Sources

- [Drift-Adapter (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.805/)
- [PyMuPDF benchmarks](https://pymupdf.readthedocs.io/en/latest/app4.html)
- [Best Python PDF to Text Parser Libraries (2026)](https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/)
- [Microsoft markitdown](https://github.com/microsoft/markitdown)
