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

### Scoped file access enforcement

Tessera exposes file *discovery* (search, symbols, file_context) but doesn't control what agents do with those paths afterward. An agent can find a file through scoped search, then read/write it via its runtime's native file tools — bypassing the scope gate entirely.

**Core design question:** Tessera defines the *policy* (scope tokens → permitted paths/permissions), but enforcement varies by runtime. Needs a pluggable adapter model:

- **Tessera-native tools** — `read_file`, `write_file`, `edit_file` MCP tools that enforce scope. Works when agents use Tessera exclusively for file I/O.
- **Claude Code hooks** — PreToolUse hooks that intercept Read/Write/Edit/Bash and check against Tessera scope. Claude Code-specific.
- **Codex sandbox rules** — Generate sandbox configs from scope tokens. Codex-specific.
- **Container/mount isolation** — Docker volume mounts or filesystem sandboxing. Runtime-agnostic but heavy.

**Open questions:** File-level vs directory-level vs pattern-based permissions. Read/write/execute granularity. Advisory locking for concurrent writes. How scope delegation interacts with permission bits. Whether Tessera should *enforce* or just *advise*. Needs a proper design session (/strategize).

### ~~Embedding model validation~~ — v0.7.x

Comprehensive benchmark completed. 12 local fastembed models, 4 rerankers, 4 OpenAI cloud models, 2 gateway models — all cross-tested on 20 ground-truth queries against Popup Maker (611 files, 2574 chunks). Results published in `docs/benchmarks.md`. Default stack: BGE-small (67MB) + Jina-tiny reranker (130MB) = 0.739 MRR at ~200MB. See benchmark scripts in `scripts/`.

### ~~Cross-encoder reranking~~ — v0.7.x

Implemented via fastembed's `TextCrossEncoder`. Post-RRF reranking is the single biggest quality lever (+0.13-0.16 MRR). Default: `jinaai/jina-reranker-v1-tiny-en` (130MB). Configurable via `--reranking-model`. Factory: `create_reranker()` in `embeddings.py`.

### ~~PPR symbol-name gate~~ — v0.7.x

PPR graph ranking now gated on symbol-name matching. Query tokens checked against `symbols` table — PPR only fires when query targets actual symbol names, preventing noise on conceptual queries. Improved VEC+PPR from 0.507 to 0.647 MRR.

---

## Search Quality Improvements

Prioritized by expected MRR impact. Based on per-query failure analysis of the PM benchmark (see `scripts/benchmark_pm.py --provider fastembed --all`).

### ~~Chunk metadata enrichment~~ — REJECTED

Tested prepending `// File: Popups.php, Class: PUM_Popups, Package: popup-maker` to chunk text before embedding. **Result: net negative.** VEC-only MRR dropped from 0.609 to 0.536 (-12%), VEC+rerank dropped from 0.739 to 0.735 (-0.5%). The comment-style prefix consumes tokens in BGE-small's 512-token window without creating useful semantic associations. The embedding model treats the structured prefix as noise.

**Alternative approaches not yet tested:** natural language summary prefix, separate metadata embedding field, or late-interaction models that handle metadata differently.

### Hybrid mode RRF weight retuning (MEDIUM — estimated +0.03-0.05 MRR)

Current hybrid mode (keyword + semantic) underperforms VEC-only (0.535 vs 0.609 MRR) because FTS5 tokenization doesn't align with natural language queries against code. Options:
- Lower keyword weight in hybrid mode (currently 1.0)
- Use FTS5 only as a fallback when VEC returns weak results
- Implement query-adaptive mode selection (detect keyword-style vs NL-style queries)

**Why:** Q5 (Exit intent trigger) passes in HYBRID mode (rank 3) but MISS in VEC-only. Some queries benefit from keyword matching. The scoring blend needs work.

### Query expansion / synonym injection (MEDIUM — estimated +0.02-0.05 MRR)

"Exit intent mouse detection trigger" has zero semantic overlap with actual code. A lightweight query expansion step (e.g., mapping "exit intent" → "mouseout", "mouseleave", "ExitIntent") would bridge the vocabulary gap.

**Why:** Q5 is a total MISS in VEC-only — the embedding can't connect natural language concepts to PHP function/class names.

**Approach:** Static synonym map for common code concepts, or LLM-generated query expansion at search time.

### Oversized chunk splitting (LOW-MEDIUM — improves indexing quality)

69 of 2300 chunks exceed 8K chars (up to 60K). These are chunker failures — files that didn't split properly. All embedding models truncate to their context window (512 tokens for BGE, 8K for OpenAI), so most of the content is invisible. Fix the AST chunker to split large functions/classes.

**Why:** Affects both local and cloud model accuracy. OpenAI benchmark may underperform partly due to this.

**Affects:** `src/tessera/chunker.py` — max chunk size enforcement.

### ~~Persistent benchmark storage~~ — v0.7.x

`benchmark_pm.py` now stores indexes at `~/.tessera/benchmarks/{model-key}/{core,pro}/`. Skips re-indexing when git HEAD unchanged. `--reindex` flag forces rebuild.

### Default model upgrade path (LOW — user-facing)

Document and wire upgrade tiers as CLI options:
- `tessera index /path` — BGE-small + Jina-tiny (197MB, 0.739 MRR)
- `tessera index /path --quality` — BGE-base + Jina-tiny (340MB, 0.766 MRR)
- `tessera index /path --max-quality` — GTE-base + Jina-turbo (590MB, 0.825 MRR)

**Affects:** `src/tessera/__main__.py` — add `--quality` / `--max-quality` preset flags.

---

## Other Backlog

### Content-addressable cache (from QMD)

QMD branch has `cache.py` (134 lines) — unified cache layer for embeddings and search results. Would reduce redundant embedding calls on repeated/similar queries.

### YAML config system (from QMD)

QMD branch has `config.py` (246 lines) — structured configuration via YAML files instead of CLI flags. Useful as the number of config options grows (embedding endpoint, model, chunk sizes, reranking, etc.).

### Local model scaffolding (from QMD)

QMD branch has `local_models.py` (339 lines) — scaffolding for running local embedding/reranking models via llama-cpp-python. No actual model loading yet, just the interface. Mostly superseded by fastembed integration.

### Collection membership validation

`_get_project_dbs()` filters by `scope.projects` but doesn't validate membership in `scope.collections`. Safe today (server-side scope creation) but should add explicit validation: resolve `collection_id` to projects, intersect with scope. Needs `GlobalDB.get_collection_projects(cid)` update.

### TS/JS REST endpoint extraction

Deferred from Phase 3. Extract cross-project references from string literals in TypeScript/JavaScript (REST endpoints, dynamic imports). PHP WordPress hooks already work.

### WebP dimension extraction

Deferred from Phase 4.5. Complex VP8/VP8L bitstream parsing for image dimensions. Low priority — PNG, JPEG, GIF, BMP already supported.

### CoIR/CodeSearchNet standard benchmark evaluation

Evaluate Tessera against the CoIR (Code Information Retrieval) benchmark suite or CodeSearchNet to get industry-standard scores comparable to published model leaderboards. Would give us NDCG@10 numbers that mean something to researchers and potential adopters.

### Evaluate microsoft/markitdown for document conversion

https://github.com/microsoft/markitdown

Microsoft's library converts PDF, DOCX, PPTX, XLSX, images, audio, and more to Markdown. Could replace or supplement our current doc parsing pipeline as a pre-processor before indexing — potentially handles all non-code file types in one dependency.

**Compare against:** current chunker.py markdown/JSON/YAML handling, existing doc indexing pipeline
**When:** after benchmarking sprint is complete. Do not prioritize over current work.

### Phase 6: Always-on file watcher

Auto-reindex on file changes instead of manual `reindex` calls. The next major planned phase per the roadmap.
