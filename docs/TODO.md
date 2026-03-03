# Tessera — TODO

## Shipped

### ~~Embedding-based snippet scoring~~ — v0.6.0

Semantic sliding-window scoring in `_find_best_match_line` / `_semantic_best_line`. Falls back to keyword overlap when embeddings are unavailable.

### ~~Symbol-aware snippet expansion (`expand_context`)~~ — v0.6.0

Collapsed ancestry snippets with `expand_context` ("lines" / "full") and `max_depth` params.

### ~~TESSERA_SESSION_ID env var~~ — v0.6.0

Implicit session scoping via environment variable. Precedence: explicit tool param > env var > dev mode.

### ~~Markdown break-point chunker~~ — v0.7.0

Ported QMD's (tobi/qmd) break-point algorithm to Python. Distance-decay scored split points: headers, code fences, blank lines, list items, horizontal rules. 15% overlap between chunks. Never selects break points inside fenced code blocks. Replaces header-only `chunk_markdown()` in the indexer pipeline.

---

## Backlog

### ~~PyPI packaging~~ — v0.7.1

Published as `tessera-idx` on PyPI via GitHub Actions trusted publisher. `pip install tessera-idx` / `uvx tessera-idx serve`.

### Scoped file read/write with locking

Add `read_file` and `write_file` MCP tools that enforce scope tokens — agents can only access files within their authorized projects. Advisory file locking to prevent concurrent writes. Scope tokens gain read/write permission bits. Turns Tessera into a full file-system ACL layer for multi-agent workflows.

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
