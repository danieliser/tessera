# Tessera Search Guide

Tessera's `search` tool combines three ranking signals — keyword matching (FTS5), semantic similarity (FAISS vectors), and graph-based importance (Personalized PageRank) — into a single ranked list using weighted Reciprocal Rank Fusion (RRF). This guide shows you how to search effectively.

## Default Hybrid Search

When you call `search("your query")`, Tessera automatically:

1. **Extracts the best-matching line** of code or document from each result
2. **Runs keyword search** (FTS5 BM25) against the entire index
3. **Embeds your query** and runs vector similarity search (if embeddings available)
4. **Runs Personalized PageRank** from keywords/semantics seed symbols (if graph available)
5. **Merges results** using weighted RRF: keyword 1.5x, semantic 1.0x, graph 0.8x
6. **Returns the top results** with context-aware snippets showing ancestors and nesting

This is the right choice for most queries — it balances precision (keyword signal) with recall (semantic signal) and structural importance (graph signal).

**Example:**

```
search("hybrid_search")
```

Returns:

```json
[
  {
    "file_path": "src/tessera/search.py",
    "start_line": 622,
    "end_line": 878,
    "score": 0.92,
    "snippet": "...",
    "rank_sources": ["keyword", "semantic", "graph"],
    "rrf_score": 0.0456
  },
  {
    "file_path": "tests/test_search.py",
    "start_line": 15,
    "end_line": 45,
    "score": 0.87,
    "snippet": "...",
    "rank_sources": ["keyword"],
    "rrf_score": 0.0401
  }
]
```

Each result includes a `score` (individual signal score) and `rrf_score` (fused rank), plus `rank_sources` showing which signals contributed.

---

## Query Mode Prefixes

You can control which search modes fire by prefixing your query with a mode selector. This is useful when you know exactly what you're looking for or want to skip expensive computations.

### `lex:` — Exact Identifier Lookup

Use this when you know the exact name of a function, class, variable, or method.

**What it does:** Keyword-only search via FTS5. Fast, precise, no embeddings needed.

**When to use:**
- Finding where a function is defined: `search("lex:normalize_bm25_score")`
- Locating a class: `search("lex:ProjectDB")`
- Finding exact variable names: `search("lex:_embedding_client")`

**Why:** FTS5 is optimized for exact identifier matching. No semantic drift, no false positives from vector similarity.

**Example:**

```
search("lex:hybrid_search")
```

Returns only chunks containing the exact term "hybrid_search" (or stemmed variants), ranked by keyword relevance. Typically returns the definition and all call sites.

**Latency:** ~1-2ms (FTS5 direct lookup)

### `vec:` — Semantic/Conceptual Search

Use this for "how does X work" questions or when you're exploring patterns.

**What it does:**
1. Embeds your query with a retrieval prefix (`"Represent this search query for retrieval: ..."`), designed for code retrieval
2. Runs FAISS cosine similarity against all chunk embeddings
3. Returns semantically related code even without keyword matches

**When to use:**
- Understanding error handling: `search("vec:error handling strategy")`
- Finding retry logic: `search("vec:exponential backoff")`
- Exploring async patterns: `search("vec:concurrent task execution")`

**Why:** Semantic search finds concepts that don't match keywords. A function named `backoff_delay` will match `"retry strategy"` via vectors, but not via FTS5.

**Example:**

```
search("vec:error handling")
```

Returns chunks related to error handling concepts — try/catch blocks, exception handling, assertion patterns — ranked by conceptual similarity, even if they don't contain the exact phrase "error handling".

**Latency:** ~20-30ms (embedding + vector search)

**Note on `embed_query`:** The retrieval prefix distinguishes between query embeddings (what you search for) and document embeddings. This improves ranking when using retrieval-tuned models like Nomic Embed Text or similar.

### `hyde:` — Hypothetical Document Embedding

Use this for abstract patterns or when describing what code should *do* rather than what it's called.

**What it does:**
1. Embeds your query as if it were a hypothetical code document (`embed_single`, no retrieval prefix)
2. Runs FAISS cosine similarity
3. Finds code that matches the conceptual description

**When to use:**
- Looking for patterns by description: `search("hyde:a function that retries an operation with exponential backoff")`
- Finding implementations of algorithms: `search("hyde:breadth-first graph traversal")`
- Conceptual exploration: `search("hyde:cache invalidation on dependency change")`

**Why:** HYDE treats your query as if it were a document, not a search query. It's useful for longer, more descriptive queries that describe what you're looking for.

**Example:**

```
search("hyde:automatically retry failed network requests with exponential delays")
```

Returns code chunks that implement exactly that pattern, ranked by how closely they match the hypothetical implementation you described.

**Latency:** ~20-30ms (embedding + vector search)

**Difference from `vec:`:** `vec:` uses a retrieval prefix; `hyde:` does not. HYDE treats your query as a hypothetical code snippet, while `vec:` treats it as a query about code. In practice, they're similar, but HYDE may work better for longer, more document-like descriptions.

### `lex,vec:` — Combine Modes

You can also run multiple search modes in one call:

```
search("lex,vec:error handling")
```

This runs both keyword search (for exact matches) and semantic search (for concepts), then merges them via weighted RRF. Useful when you want both precision and recall.

**Valid combinations:**
- `lex:query` — keyword only
- `vec:query` — semantic only (embed_query)
- `hyde:query` — semantic only (embed_single)
- `lex,vec:query` — keyword + semantic
- `lex,hyde:query` — keyword + HYDE semantic

---

## Quick Decision Tree

**When to use which mode:**

```
Do you know the exact name?
├─ Yes → lex:name (fast, precise)
└─ No → How should I describe it?
   ├─ Short keyword phrase ("error handling") → Default hybrid or vec:
   ├─ Long conceptual description → hyde:
   └─ Unsure → Default hybrid search
```

**Real-world examples:**

| Task | Query | Mode | Why |
|------|-------|------|-----|
| Find the `hybrid_search` function definition | `search("lex:hybrid_search")` | lex | Exact name known |
| Understand error handling patterns | `search("vec:error handling")` | vec | Exploring concepts |
| Find retry logic by description | `search("hyde:retry with exponential backoff")` | hyde | Describing behavior |
| Search for "authentication" broadly | `search("authentication")` | default | Mixed relevance expected |

---

## Output Formats

By default, `search()` returns JSON with full metadata. You can request different formats depending on your task.

### `output_format="json"` (default)

Full structured results. Best for programmatic processing, analysis, or when you need rich metadata.

```python
search("hybrid_search", output_format="json")
```

Returns:

```json
[
  {
    "id": 12345678,
    "file_path": "src/tessera/search.py",
    "file_id": 1,
    "start_line": 622,
    "end_line": 878,
    "content": "def hybrid_search(...)",
    "score": 0.92,
    "rrf_score": 0.0456,
    "rank_sources": ["keyword", "semantic"],
    "source_type": "code",
    "trusted": true,
    "section_heading": "",
    "snippet": "622 | def hybrid_search(\n623 | ...",
    "snippet_start_line": 622,
    "snippet_end_line": 630,
    "best_match_line": 624,
    "ancestors": [],
    "docid": "a1b2c3"
  }
]
```

Fields:
- `rrf_score` — final merged score (0-1 range)
- `rank_sources` — which signals ranked this result ("keyword", "semantic", "graph")
- `trusted` — whether this is code (true) or document (false)
- `docid` — stable 6-char content hash for caching/deduplication
- `snippet` — best-matching lines with context and ancestry nesting

### `output_format="markdown"`

Formatted as Markdown with code blocks. Best for displaying to users or pasting into documentation.

```python
search("error handling", output_format="markdown", limit=3)
```

Returns:

```markdown
### 1. `src/tessera/search.py:132-163` (score: 0.87)
```
132 | def _find_best_match_line(
...
```

### 2. `tests/test_search.py:50-65` (score: 0.81)
```
50 | def test_error_handling():
...
```
```

### `output_format="csv"`

Tabular format. Best for analysis, spreadsheet import, or building reports.

```python
search("ProjectDB", output_format="csv")
```

Returns:

```csv
id,file_path,start_line,end_line,source_type,trusted,rrf_score,rank_sources
12345628,src/tessera/db.py,10,150,code,true,0.045,["keyword"]
12345629,tests/test_db.py,5,45,code,true,0.041,["keyword","semantic"]
```

### `output_format="files"`

Deduplicated file paths only. Best for knowing which files to read without full results.

```python
search("authentication", output_format="files")
```

Returns:

```
src/tessera/auth.py
src/tessera/server/tools/_scope.py
tests/test_auth.py
docs/architecture/security.md
```

---

## `doc_search_tool` vs `search`

Tessera indexes everything — code, documentation, config files, media assets. The `search` tool searches everything. For document-specific searches, use `doc_search_tool`.

### `search` — Code + Documents + Assets

Searches across all content types. Use this for broad queries.

```python
search("cache invalidation")  # Searches code, docs, configs, assets
```

Results include:
- Code: function definitions, implementations, comments
- Documents: markdown, PDF, YAML, JSON specs
- Assets: image filenames, media metadata
- All filtered by default

### `doc_search_tool` — Documents Only

Searches only non-code content: markdown, PDF, YAML, JSON, config files, plaintext. Excludes source code and binary assets.

```python
doc_search_tool("authentication flow")  # Markdown specs, architecture docs, config examples
```

**Supported formats:**

`markdown`, `pdf`, `yaml`, `json`, `html`, `xml`, `text`, `txt`, `rst`, `csv`, `tsv`, `log`, `ini`, `cfg`, `toml`, `conf`

**When to use:**

- Finding documentation: `doc_search_tool("API design")`
- Searching config files: `doc_search_tool("database connection", formats="yaml,toml")`
- Finding specifications: `doc_search_tool("authentication", formats="markdown")`
- Searching logs: `doc_search_tool("error timeout", formats="log")`

**Example:**

```python
# Search documentation about caching
doc_search_tool("cache invalidation strategy", formats="markdown")

# Find all YAML config examples
doc_search_tool("Redis", formats="yaml")

# Search error logs for timeouts
doc_search_tool("timeout", formats="log")
```

---

## Advanced: Custom RRF Weights

By default, Tessera weights search signals as: keyword=1.5x, semantic=1.0x, graph=0.8x. You can adjust these based on your use case.

```python
# Boost semantic search (for concept-heavy queries)
search("error handling", weights="keyword=1.0,semantic=2.0,graph=0.8")

# Boost keyword search (for identifier-heavy queries)
search("ProjectDB", weights="keyword=3.0,semantic=0.5,graph=0.8")

# Disable graph ranking (if it's noisy)
search("authentication", weights="keyword=1.5,semantic=1.0,graph=0.0")
```

**Format:** `"key1=weight1,key2=weight2,..."`

**Default weights:**
- `keyword=1.5` — FTS5 BM25 scores (higher for code search precision)
- `semantic=1.0` — FAISS cosine similarity (baseline)
- `graph=0.8` — Personalized PageRank (when graph available)

**Tuning guide:**
- Increase a weight to boost that signal's influence on ranking
- Decrease to dampen it
- Set to 0 to disable entirely

---

## Advanced: Snippet Context and Ancestry

Results include code snippets with ancestor context — showing where in the file hierarchy the match lives.

### `expand_context="lines"` (default)

Shows the best-matching lines plus a collapsed nesting skeleton. Useful for quick context without scrolling.

```
622 | def hybrid_search(query, query_embedding, db, graph=None, ...):
    | ...  (100 lines)
720 |     keyword_results = []
721 |     if run_keyword:
```

The `...  (100 lines)` marker shows collapsed content, preserving the nesting structure while reducing verbosity.

### `expand_context="full"`

Expands the entire containing function or class. Useful when you need the full definition.

```
622 | def hybrid_search(query, query_embedding, db, graph=None, ...):
623 |     """Hybrid search combining keyword and semantic results..."""
...
878 |     return results
```

### `max_depth=N`

Limit ancestor nesting levels to display. Useful for deeply nested code.

```python
# Show only immediate containing function (not outer classes)
search("error handling", expand_context="lines", max_depth=1)

# Show up to 2 levels of nesting
search("error handling", expand_context="lines", max_depth=2)
```

---

## Advanced: FTS5 Operators (Phrase, Negation, Prefix, Proximity)

By default, queries are treated as simple keyword lists. Enable `advanced_fts=True` to use FTS5 operators for precise matching.

### Phrase Matching

```python
# Exact phrase: results must contain these words in order
search('"def hybrid_search"', advanced_fts=True)
```

Returns chunks with the exact phrase `def hybrid_search`, not just chunks containing those words separately.

### Negation

```python
# Exclude results containing "warning"
search("error NOT warning", advanced_fts=True)
```

Returns chunks with "error" but not "warning".

### Prefix Matching

```python
# Match words starting with "embed"
search("embed*", advanced_fts=True)
```

Matches `embed`, `embedding`, `embed_query`, `embedded`, etc.

### Proximity

```python
# Words within 5 tokens of each other
search('NEAR(search query, 5)', advanced_fts=True)
```

Returns chunks where "search" and "query" appear within 5 tokens.

**Default safe mode** (without `advanced_fts=True`):

All query operators are escaped/treated as literals, preventing syntax errors. This is the safe default for user-facing queries.

```python
search('error OR warning')  # Treated as literal phrase, not OR operator
```

**Valid combinations:**

- Phrases + negation: `'"error handling" NOT async'`
- Prefix + phrases: `'hybrid* "search function"'`
- Proximity + negation: `'NEAR(authenticate, scope, 3) NOT global'`

---

## Advanced: Filtering by Language and Source Type

Search can be filtered to specific programming languages or source types.

### Filter by Programming Language

```python
# Search only Python code
search("error handling", filter_language="python")

# Search only TypeScript
search("interface", filter_language="typescript")

# Search only PHP
search("add_action", filter_language="php")
```

**Supported:** `python`, `typescript`, `javascript`, `php`, `swift`

### Filter by Source Type

```python
# Search only code (exclude docs and assets)
search("function", source_type="code")

# Search only markdown documentation
search("authentication", source_type="markdown")

# Search only images and media
search("logo", source_type="asset")
```

**Valid source types:** `code`, `markdown`, `yaml`, `json`, `asset`, and others (see `doc_search_tool` for full list)

---

## Real-World Search Examples

### Find a Function Definition

```python
# If you know the exact name:
search("lex:hybrid_search")

# If unsure of the name:
search("vec:merge multiple ranked search results")
```

Both return the function definition plus all its call sites, ranked by relevance.

### Understand Error Handling Patterns

```python
search("vec:error handling strategy")
```

Returns try/catch blocks, exception handling, assertion patterns — ranked by how closely they match the concept, regardless of exact keywords.

### Find All Files Touching Authentication

```python
search("authentication", output_format="files")
```

Quick overview of which files need review for a security audit.

### Find Specific Phrases in Specs

```python
search('"API design" "error response"', advanced_fts=True, formats="markdown")
```

Returns documentation chunks containing both phrases in proximity, helping you find related spec sections.

### Find Retry Logic by Behavior

```python
search("hyde:automatically retries failed operations with exponential backoff and max attempts")
```

Finds implementations that match the behavior description, even if they use different terminology.

### Search Config Files Only

```python
doc_search_tool("database host", formats="yaml,toml")
```

Returns YAML and TOML config examples without code clutter.

### Find Test Cases for a Function

```python
search("lex:test_hybrid_search")
```

Returns all test functions for `hybrid_search`, useful for understanding expected behavior.

### Analyze Cross-Project Dependencies

```python
# Search in specific project collection
search("ProjectDB")  # Returns all references across all collected projects

# Get just file paths to know where to start
search("ProjectDB", output_format="files")
```

---

## Performance Notes

Tessera achieves sub-100ms p95 latency on hybrid queries:

| Search Type | Latency | Notes |
|-------------|---------|-------|
| Keyword only (`lex:`) | 1-2ms | FTS5 direct lookup, fastest |
| Semantic only (`vec:` or `hyde:`) | 20-30ms | Embedding + FAISS, no keyword overhead |
| Full hybrid (default) | 50-100ms | All three signals merged; capped by embedding |
| With ancestry nesting | +2-5ms | Extra symbol lookup for context |

**BM25 short-circuit:** When keyword search returns a very confident top match (score >= 0.85 with >= 0.15 gap to #2), Tessera skips expensive semantic and graph searches, returning results in ~1-2ms.

---

## Troubleshooting

**No results returned:**
1. Try the default hybrid search (no prefix)
2. Use a shorter query (fewer words = fewer constraints)
3. Remove `advanced_fts=True` if enabled (syntax errors might be silently failing)
4. Check `output_format="json"` to confirm 0 results vs formatting issue

**Too many results / poor ranking:**
1. Use `lex:` if you know the exact identifier
2. Try `vec:` or `hyde:` for semantic refinement
3. Add language filter: `filter_language="python"`
4. Reduce `limit` parameter and inspect top results

**Slow queries:**
1. Enable keyword short-circuit by making queries more specific
2. Use `lex:` for identifier lookups
3. Reduce `limit` to avoid over-fetching
4. Check if embedding endpoint is slow (profile separately)

**Irrelevant semantic results:**
1. Try `lex:` for exact matching
2. Use `advanced_fts=True` to narrow down with phrases
3. Boost keyword weight: `weights="keyword=2.0,semantic=1.0,graph=0.8"`
4. Try a more specific query description for `hyde:`

---

## Architecture Concepts

**RRF (Reciprocal Rank Fusion):** Combines multiple ranking signals by converting ranks to scores. Each signal contributes: `score(doc) = weight / (k + rank)` where k=60 is the RRF constant. Documents ranked highly by multiple signals bubble to the top.

**BM25:** FTS5's keyword ranking function. Normalizes to [0, 1] range where 1 = perfect match. Higher scores indicate better keyword relevance.

**Embedding models:** Tessera uses OpenAI-compatible endpoints (e.g., LM Studio with Nomic Embed Text). The retrieval prefix in `embed_query` optimizes for code-retrieval scenarios.

**Personalized PageRank (PPR):** Graph-based ranking that spreads importance from matched symbols to their dependent symbols. Useful for finding related code even without keyword/semantic matches.

---

**Questions?** See the [Tessera home page](./index.md) or the [Advanced Search](./search-advanced.md) guide.
