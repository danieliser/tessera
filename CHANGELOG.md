# Changelog

All notable changes to Tessera are documented in this file.

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
