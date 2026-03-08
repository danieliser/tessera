# Changelog

All notable changes to Tessera are documented in this file.

## [0.10.1] — 2026-03-08

### Added
- **`events()` MCP tool** — query event/hook registrations and emissions across indexed projects
  - Filter by event name (supports `%` wildcard), direction (`registers_on`, `fires`, or both)
  - Mismatch detection: `ORPHANED_LISTENER` (registered but never fired), `UNHEARD_HOOK` (fired but no listeners)
  - Multi-project parallel queries with scope-gated access
- **`ProjectDB.get_events()` method** — SQL query layer for directional event edges

## [0.10.0] — 2026-03-08

### Added
- **Per-language parser plugin architecture** — auto-discovery of language modules, one file per language
  - `LanguageExtractor` abstract base class with declarative event pattern registry
  - Drop-in language support: create a module in `parser/languages/`, it auto-registers
  - Shared AST helpers extracted to `_helpers.py` and `_base.py`
- **Directional event/hook edges** — replaces undirected `hooks_into` with `registers_on` and `fires`
  - PHP: `add_action`/`add_filter` → `registers_on`, `do_action`/`apply_filters` → `fires`
  - JS/TS: `on`/`once`/`addEventListener` → `registers_on`, `emit`/`dispatchEvent`/`trigger` → `fires`
  - Python: `signal.connect` → `registers_on`, `signal.send`/`send_robust` → `fires`
- **Expanded default ignore patterns** — `.pytest_cache/`, `.ruff_cache/`, `.webpack-cache/`, `.nx/`, `.cursor/`, `.husky/`, `.serena/`, `.phpstan-cache/`

### Changed
- Parser dispatch uses plugin registry with fallback to legacy functions
- `build_edges()` allows `registers_on` and `fires` edges for unresolved external symbols
- All existing `hooks_into` references become `registers_on` or `fires` on re-index

### Removed
- `hooks_into` edge kind — replaced by directional `registers_on`/`fires`

## [0.9.0] — 2026-03-07

### Added
- **MDX file indexing** — `.mdx` files indexed as markdown documents
- **Filename-aware RRF boosting** — post-merge score boost when query tokens match filename
- **File-level dedup** — keeps best chunk per file, frees result slots for diverse files
- **Search quality validation** — Next.js v16.1.6 benchmark (3,677 chunks): Doc 100%, Cross 100%, Code 70%, Blend MRR 0.748
- **Popup Maker benchmark** — 20-query PHP benchmark suite with VEC MRR 0.542
- Query expansion, depth penalty, symbol boost levers (tested, some disabled)
- Top-5 and Top-10 accuracy metrics in benchmark output

### Changed
- PPR graph boost disabled in search (neutral-to-harmful in benchmarks) — graphs still load for impact/references
- Default search stack: BGE-base (768d) with filename-aware RRF, no reranker

## [0.8.0] — 2026-03-05

### Added
- **Model profiles** — per-model embedding configuration (dimensions, chunk size, query prefix)
- **Reranker support** — configurable reranker endpoint with pool size sweep
- **CSN benchmark** — CodeSearchNet evaluation, #2 on CoIR leaderboard at NDCG@10 91.4
- Flexible reranker endpoints in model profiles
- 4-lever benchmark sweep infrastructure (CSV export, HyDE, RRF weights, chunk budget)

## [0.7.0] — 2026-03-03

### Added
- **Break-point Markdown Chunker** — ported from QMD (tobi/qmd) TypeScript implementation
  - Distance-decay scored break points: headers (H1-H6), code fences, blank lines, list items, horizontal rules
  - Squared distance decay so headings far back still beat low-quality breaks near the boundary
  - Code fence protection — never selects break points inside ``` regions
  - 15% overlap between adjacent chunks for context continuity
  - Heading hierarchy tracking (`section_heading`, `parent_section`) on every chunk
  - Defaults: 3600 chars max (~900 tokens), 800 char search window (~200 tokens)
- New module: `src/tessera/markdown_chunker.py` — standalone, no external dependencies

### Changed
- Indexer pipeline now uses `chunk_markdown_breakpoint()` for `.md` and `.pdf` files (was `chunk_markdown()`)
- Old `chunk_markdown()` in `document.py` retained for backward compatibility

## [0.6.0] — 2026-03-02

### Added
- **Collapsed Ancestry Snippets** — search results now show structural nesting context (class → method → match) with collapsed regions and line numbers on every visible line
- `expand_context` param: `"lines"` (default) for collapsed skeleton, `"full"` for complete symbol expansion
- `max_depth` param to limit ancestor nesting levels shown
- `end_line` tracking on all symbols (schema v3 migration) via tree-sitter `node.end_point`
- `get_ancestor_symbols()` range query on ProjectDB
- Signature reconstitution for display — `def`/`class` keywords prepended when missing from stored signatures

### Fixed
- Off-by-one in snippet line numbers (chunk start_line is 0-based, display is 1-based)
- Multiline signatures truncated to first line for clean rendering

## [0.5.0] — 2026-03-02

### Added
- **Structured Query Types** — `lex:`, `vec:`, `hyde:` prefix syntax and `search_mode` param for explicit search list routing
- **FTS5 Advanced Mode** — `advanced_fts` param enables phrase queries, negation (NOT), prefix (*), and proximity (NEAR) operators
- **Configurable RRF Weights** — `weights` param on search tool allows per-query tuning of keyword/semantic/graph fusion balance
- **Cross-file Edge Resolution v2** — file-proximity tiebreaking for ambiguous symbol names, suffix matching for namespaced symbols (`helper` → `App\Utils\helper`), resolution stats dict for diagnostics
- **Rich MCP Tool Descriptions** — all 18 tool docstrings expanded with strategy guides, worked examples, and parameter explanations
- **Empirical Search Quality Benchmarks** — 25-test benchmark suite covering BM25 normalization, RRF fusion, structured queries, FTS5 operators, latency
- Content-addressable document IDs (`docid`) for stable result references
- Snippet extraction with keyword highlighting
- `embed_query` with retrieval prefix for search-optimized embeddings
- Multi-format output (json, markdown, csv, files) via `output_format` param
- BM25 short-circuit for high-confidence keyword matches (skips expensive semantic/PPR)
- Weighted RRF fusion with per-list weights (keyword=1.5x, semantic=1.0x, graph=0.8x)

### Fixed
- FTS5 sort order now returns results by BM25 rank (was arbitrary)
- BM25 score normalization to 0-1 range for consistent cross-query comparison
- Snippet tokenization aligned with FTS5 word boundaries
- `file_path` enrichment in hybrid search results
- MockDB signatures updated for `advanced_fts` param across test suite

## [0.4.0] — 2026-03-01

### Added
- **PPR Graph Intelligence** — Personalized PageRank as third ranking signal in hybrid search (three-way RRF)
- `graph.py` — PPR via scipy CSR sparse matrices with LRU-cached per-query results
- Adaptive sparse threshold based on percolation theory (density + largest connected component)
- Cross-file edge resolution pass connects unresolved references across files (LCC 1.4% → 67.7%)
- Memory monitoring with LRU eviction for graph cache
- Extended JS/TS symbol extraction: object property functions, assignment expressions, function expressions
- Multi-project spike tests with nDCG@5 metrics
- Search latency benchmarks and 80ms threshold guard
- Configurable embedding endpoint for hybrid search and reindex

### Fixed
- `doc_search` argument order bug
- `variable_declarator` now handles `function_expression` (previously only `arrow_function`)

## [0.3.0] — 2026-02-28

### Added
- **Media & Binary File Metadata Catalog** — index images, PDFs, and binary files with extracted metadata
- Document text extraction via PyMuPDF4LLM
- Per-project `.tesseraignore` with two-tier security model
- Drift-Adapter for embedding model migration without re-indexing (Orthogonal Procrustes)
- JSDoc/PHPDoc type reference extraction from doc comments
- PHP type reference extraction and `implements` support

## [0.2.0] — 2026-02-27

### Added
- **Document Indexing & Text Format Support** — markdown, plaintext, and structured document indexing
- Hybrid search with keyword + semantic RRF merge
- Incremental re-indexing with file hash comparison
- MCP server with search, symbols, references, impact, and file_context tools
- Scope-gated sessions with project/collection/global tiers
- Tree-sitter parsing for PHP, TypeScript, Python, Swift
