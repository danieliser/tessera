# Tessera — TODO

## Shipped

### ~~Embedding-based snippet scoring~~ — v0.6.0

Semantic sliding-window scoring in `_find_best_match_line` / `_semantic_best_line`. Falls back to keyword overlap when embeddings are unavailable.

### ~~Symbol-aware snippet expansion (`expand_context`)~~ — v0.6.0

Collapsed ancestry snippets with `expand_context` ("lines" / "full") and `max_depth` params.

### ~~TESSERA_SESSION_ID env var~~ — v0.6.0

Implicit session scoping via environment variable. Precedence: explicit tool param > env var > dev mode.

---

## Up Next

### Markdown-aware chunking (break-point algorithm)

**Priority: HIGH** — Current `chunk_markdown()` in `document.py` splits on headers only. The QMD branch (`feature/qmd-feature-adoption`, Phase 2) has a break-point algorithm (`markdown_chunker.py`, 228 lines) that scores split points by structural boundaries: headers, code fences, blank lines, list items, thematic breaks. Never splits inside fenced code blocks.

**Why it matters:** Blog posts, support tickets, documentation — anything written in markdown gets chunked poorly today. Header-only splitting produces chunks that are either too large (long sections) or too granular (many sub-headers). The break-point algorithm produces semantically coherent chunks regardless of header structure.

**Approach:** Port `markdown_chunker.py` from the QMD branch to main. It's a clean standalone module with no dependencies on other QMD changes. Wire it into the indexer pipeline as a replacement for the current `chunk_markdown()`.

---

## Backlog

### PyPI packaging

`pyproject.toml` already has the `[project.scripts]` entry point. Needs `uv build && uv publish` to enable `pip install tessera` / `uvx tessera serve`. Lowest effort, highest accessibility win.

### Embedding model validation

Testing plan exists for Jina Code Embeddings 1.5B with Matryoshka truncation (1536 to 384d). Target: >95% quality retention, <100ms P95 latency, NDCG >75% with RRF fusion. 4-7 day validation timeline scoped but not started. Nomic 1.5 as fallback for on-device.

### Cross-encoder reranking (from QMD)

Post-RRF reranking via cross-encoder model. QMD branch has `reranking.py` (172 lines) with a reranking client. Would improve precision on ambiguous queries. Depends on local model infrastructure.

### Content-addressable cache (from QMD)

QMD branch has `cache.py` (134 lines) — unified cache layer for embeddings and search results. Would reduce redundant embedding calls on repeated/similar queries.

### YAML config system (from QMD)

QMD branch has `config.py` (246 lines) — structured configuration via YAML files instead of CLI flags. Useful as the number of config options grows (embedding endpoint, model, chunk sizes, reranking, etc.).

### Local model scaffolding (from QMD)

QMD branch has `local_models.py` (339 lines) — scaffolding for running local embedding/reranking models via llama-cpp-python. No actual model loading yet, just the interface.

### Collection membership validation

`_get_project_dbs()` filters by `scope.projects` but doesn't validate membership in `scope.collections`. Safe today (server-side scope creation) but should add explicit validation: resolve `collection_id` to projects, intersect with scope. Needs `GlobalDB.get_collection_projects(cid)` update.

### TS/JS REST endpoint extraction

Deferred from Phase 3. Extract cross-project references from string literals in TypeScript/JavaScript (REST endpoints, dynamic imports). PHP WordPress hooks already work.

### WebP dimension extraction

Deferred from Phase 4.5. Complex VP8/VP8L bitstream parsing for image dimensions. Low priority — PNG, JPEG, GIF, BMP already supported.

### Phase 6: Always-on file watcher

Auto-reindex on file changes instead of manual `reindex` calls. The next major planned phase per the roadmap.
