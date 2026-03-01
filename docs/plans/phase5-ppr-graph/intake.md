# Phase 5 — PPR Graph Intelligence: Intake

**Date:** 2026-02-28
**Tier:** Standard
**Project Type:** Technical

## Idea

Implement Personalized PageRank (PPR) for Tessera's code graph to add graph-aware ranking to search and impact analysis. Currently, Tessera's search uses two-way RRF (BM25 keyword + FAISS semantic). Impact analysis uses BFS traversal over the edges table. PPR adds structural importance ranking — surfacing the most relevant symbols based on graph topology, not just text/vector similarity.

## Question Framing

**The right question:** Does Tessera's actual graph data (tree-sitter-extracted edges) have sufficient density for PPR to produce meaningfully better rankings than BFS/flat lists? The algorithm is straightforward; the value depends on graph quality.

**Adjacent questions:**
- Graph density: How many edges does a typical indexed project produce? If sparse, PPR degenerates to BFS.
- Three-way RRF: Adding PPR as a third signal changes result ranking for every query. Graceful degradation when graph is empty or sparse?
- Server lifecycle: In-memory graph loading at startup increases boot time. Rebuild cost after reindex?
- Dependency choice: scipy (spec assumption), numpy-only, fast-pagerank, or NetworkX — which performs best empirically?

**Hidden assumptions:**
- Tree-sitter edges are dense enough for meaningful PPR differentiation (unvalidated on real Tessera data)
- PPR can be added as a third RRF signal without degrading quality when graph is sparse
- <100ms at 50K edges (from HippoRAG benchmarks on knowledge graphs, not code graphs)

## Premise Validation

**Validated.** HippoRAG/HippoRAG 2 (NeurIPS'24, ICML'25) demonstrate 7-20% improvement in associative retrieval using PPR over knowledge graphs. Aider uses tree-sitter + NetworkX PageRank in production. fast-pagerank library shows scipy CSR PPR outperforms NetworkX's implementation. The approach is proven; the question is whether Tessera's graph topology supports it.

## Constraints

- **Dependencies:** No upfront constraints. Research phase evaluates scipy, numpy-only, fast-pagerank, and NetworkX empirically. Spec recommends the winner.
- **Budget:** ~1,500 LOC (from architecture spec)
- **Performance gate:** PPR <100ms for graphs up to 50K edges
- **Existing architecture:** Must integrate with existing RRF merge in search.py, existing edges table in db.py

## Success Criteria

1. Three-way hybrid search: BM25 + semantic + PPR merged via RRF
2. PPR-enhanced impact tool with graph-aware ranking
3. In-memory graph loaded at server start, rebuilt after reindex
4. Published benchmark results comparing search/impact with and without PPR on real projects
5. Graceful fallback when graph is sparse or empty
6. All existing tests still pass

## Non-Goals (Phase 6+)

- WebP dimension extraction
- EXIF/IPTC metadata extraction
- Graph edges from documents to code symbols
- Low-Rank Affine drift adapter variant
- Cross-language annotation tool
- Document version history
- Advanced semantic chunking
- OCR from images
- Archive content extraction
- DOCX/Word support
- Automated drift-adapter triggering
- Audit logging
- File watcher implementation

## Follow-up (Post-Phase 5)

- **Graph backend comparison:** After PPR pipeline works, benchmark NetworkX, scipy, fast-pagerank, and numpy-only as drop-in replacement modules. Document tradeoffs for future optimization.

## Prior Art in This Codebase

- Architecture spec: `docs/plans/architecture/spec-v2.md` lines 331-373 (PPR design), 611-628 (Phase 5 scope)
- Architecture research: `docs/plans/architecture/research.md` lines 125-167 (HippoRAG PPR validation)
- Existing edges table: `src/tessera/db.py` (edges, caller_refs tables)
- Existing RRF merge: `src/tessera/search.py` (hybrid_search, rrf_merge)
- Existing impact tool: `src/tessera/server.py` (impact tool handler)
