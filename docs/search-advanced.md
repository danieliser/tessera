# Tessera Advanced Search Guide

This guide covers Tessera's power-user search features: FTS5 operators, custom RRF weights, search mode overrides, and document-specific filtering. If you haven't read the basic search guide, start there. This document assumes you're comfortable with the search tool and want more control over how results are ranked and filtered.

## Quick Reference

| Feature | Syntax | Example |
|---------|--------|---------|
| Phrase matching | `"exact phrase"` + `advanced_fts=True` | `"def hybrid_search"` |
| Negation | `term NOT excluded` + `advanced_fts=True` | `error NOT warning` |
| Prefix matching | `prefix*` + `advanced_fts=True` | `hybrid*` |
| Proximity search | `NEAR(term1 term2, distance)` + `advanced_fts=True` | `NEAR(search query, 5)` |
| Keyword-only mode | `lex:query` | `lex:ProjectDB` |
| Semantic-only mode | `vec:query` or `hyde:query` | `vec:error handling` |
| Custom weights | `weights="key=val,key=val"` | `weights="keyword=3.0,semantic=1.0"` |
| Language filter | `filter_language="lang"` | `filter_language="python"` |
| Document format filter | `formats="fmt1,fmt2"` | `formats="markdown,yaml"` |

---

## 1. FTS5 Advanced Operators

Tessera uses SQLite FTS5 for keyword search. By default, operators like `"..."`, `NOT`, `*`, and `NEAR(...)` are escaped and treated as literal strings. Enable `advanced_fts=True` to activate these operators.

### Phrase Matching

**Operator:** `"exact phrase"`

Phrase matching finds exact word sequences in order. Useful for finding function definitions or specific patterns.

```
search('"def hybrid_search"', advanced_fts=True)
```

Expected: Returns chunks containing the exact phrase `def hybrid_search` in sequence. Without quotes, the search returns any chunk with both `def` and `hybrid_search` anywhere in the content (not necessarily adjacent).

**Real benchmark data:**

| Query | Phrase Results | Unquoted Results | Difference |
|-------|----------------|-----------------|-----------|
| `"def hybrid_search"` | 9 | 20 | Phrase is more precise |

**When to use:** Finding specific function signatures, exact error messages, or code patterns that must appear together.

### Negation

**Operator:** `term NOT excluded`

Exclude results containing a specific term.

```
search("error NOT warning", advanced_fts=True)
```

Expected: Returns chunks with "error" but not "warning".

**When to use:** Narrowing broad searches. For example, `"authentication NOT test"` excludes test files while keeping authentication logic.

### Prefix Matching

**Operator:** `prefix*`

Match any term starting with a prefix. Useful for function name families or similar variables.

```
search("hybrid*", advanced_fts=True)
```

Expected: Returns chunks with words starting with `hybrid` (e.g., `hybrid_search`, `hybrid_rrf`, `hybrid_merge`).

**Benchmark latency:**

| Query | Latency (ms) |
|-------|------------|
| `hybrid*` | 0.33 |
| `hybrid_search` (exact) | 0.70 |

Prefix matching is fast because FTS5 indexes pre-fix patterns.

**When to use:** Exploring function families (e.g., `search_*` for all search-related functions) or variable naming patterns.

### Proximity Search

**Operator:** `NEAR(term1 term2, distance)`

Find terms that appear within N tokens of each other. Tokens are individual words or punctuation.

```
search("NEAR(search query, 5)", advanced_fts=True)
```

Expected: Returns chunks where "search" and "query" appear within 5 tokens of each other.

**Real benchmark data:**

| Query | Results with NEAR | Results without |
|-------|-------------------|-----------------|
| `NEAR(search query, 5)` | 3 | 10+ |

**When to use:** Finding concepts that are discussed together. For example, `NEAR(error handling, 3)` finds code discussing error handling in close proximity, which is often more relevant than scattered mentions.

### Mixing FTS5 Operators

Combine operators for powerful queries:

```
search('"error handling" NOT test', advanced_fts=True)
search('NEAR(authentication scope, 5) NOT mock', advanced_fts=True)
```

**Caveat:** FTS5 operator syntax is strict. If a query fails, fall back to `advanced_fts=False` (the default, which treats operators as literal strings) or simplify the query.

### Safe Mode (Default)

When `advanced_fts=False` (the default), all special characters are escaped. This prevents syntax errors but treats operators as literal strings:

```
search("error NOT warning", advanced_fts=False)  # Searches for the literal string "error NOT warning"
```

**Benchmark:** Safe mode is always faster because queries can't fail — no syntax parsing overhead.

---

## 2. Custom RRF Weights

Tessera merges three ranking signals via Weighted Reciprocal Rank Fusion:

1. **keyword** (FTS5 BM25 score) — Exact term matching, precise for code
2. **semantic** (FAISS vector similarity) — Conceptual matching, good for design docs
3. **graph** (PageRank on symbol call graph) — Structural importance, good for understanding dependencies

Default weights: `keyword=1.5, semantic=1.0, graph=0.8`

Keyword is weighted highest because FTS5 precision is higher for code search. You can override these weights to adjust what Tessera prioritizes.

### Boost Keyword (Identifier-Heavy Searches)

```
search("ProjectDB hybrid_search", weights="keyword=3.0,semantic=1.0,graph=0.8")
```

Use when you're looking for specific functions, classes, or variables by name. Keyword matching is already precise; boosting it suppresses noisier semantic results.

**Real benchmark impact:**

| Weights | Result Change | Notes |
|---------|---------------|-------|
| Equal (1.0, 1.0, 1.0) | Baseline | Semantic noise bleeds through |
| Keyword boost (2.0, 1.0) | 2/3 results rank differently | Top-1 + Top-3 swap |

### Boost Semantic (Conceptual Searches)

```
search("how do I implement retry logic", weights="keyword=1.0,semantic=2.0,graph=0.8")
```

Use when you're thinking in concepts, not exact names. A question like "how does authentication work" will find relevant design docs and patterns even if it doesn't contain the word "authentication".

**When to use:**
- Searching for design patterns ("factory pattern", "dependency injection")
- Exploring how a system works ("how does caching work")
- Looking for examples in documentation

### Boost Graph (Dependency-Aware Searches)

```
search("how does ProjectDB impact search", weights="keyword=1.0,semantic=1.0,graph=2.0")
```

Use when you want to understand what code is structurally important. PageRank prioritizes symbols that are called frequently and have many incoming edges (central to the system).

**When to use:**
- Understanding critical paths in the codebase
- Finding bottleneck functions that many other functions depend on
- Mapping influence (what breaks if I change this function?)

### Defaults: When NOT to Override

Default weights are tuned for general-purpose search and work well for most queries. Don't override unless you have a specific reason:

- **Identifier lookups** (finding a function by name): Don't bother tuning — exact keyword match dominates anyway
- **Broad exploratory searches**: Defaults blend all signals well
- **Single-project searches**: Defaults assume balanced index (semantic and graph signals are reliable)

---

## 3. Search Mode Override

By default, Tessera runs both keyword (LEX) and semantic (VEC) search and merges results. You can override this behavior to run only one search type, or force a specific semantic mode.

### Keyword-Only (LEX)

**Inline syntax:** `lex:query`

```
search("lex:hybrid_search")  # Force keyword-only
search("lex:ProjectDB")      # Look for exact function/class name
```

Equivalent to explicit mode:

```
search("ProjectDB", search_mode="lex")
```

**Speed:** Keyword-only search is fast (sub-millisecond for most queries). No embedding cost, no semantic ranking overhead.

**Real benchmark:** Keyword-only vs full hybrid for same query:

| Query | Keyword-only (ms) | Full Hybrid (ms) | Saved |
|-------|-------------------|-----------------|-------|
| `ProjectDB` | 0.81 | 14.3 | 13.5 ms |

**When to use:**
- You know the exact name of what you're searching for
- Latency is critical (e.g., real-time code navigation)
- Semantic results are too noisy (rare, but happens with short queries)

### Semantic-Only (VEC)

**Inline syntax:** `vec:query`

```
search("vec:error handling strategy")  # Conceptual search only
search("vec:how do I validate input")  # Question-like query
```

Runs embedding and FAISS vector search only. Skips keyword matching.

**When to use:**
- Your query is a question or description, not an exact name
- You want pure conceptual matching (useful for design docs, architecture comments)
- Keyword results are too strict (e.g., searching for "retry" misses "exponential backoff")

### Hypothetical Document Embedding (HYDE)

**Inline syntax:** `hyde:query`

```
search("hyde:how to set up authentication")
search("hyde:deploy to production safely")
```

HYDE (Hypothetical Document Embeddings) uses a different embedding strategy: instead of adding a retrieval prefix (which expects you to write in the style of a search query), it embeds your query "as-is" (as if you're asking a hypothetical document).

**When VEC vs HYDE matters:**

VEC uses `embed_query()` which adds a retrieval prefix optimized for short search queries.
HYDE uses `embed_single()` without the prefix, treating your query as natural language.

Real benchmark shows both work well for most queries:

| Query | VEC Top-1 | HYDE Top-1 | Same? |
|-------|----------|-----------|-------|
| `normalize_bm25_score` | search.py | test_search_ben | Different |
| `error handling` | test_server.py | test_server.py | Same |
| `graph traversal` | research.md | spec-v1.md | Different |

Try HYDE when VEC results aren't relevant. The difference is subtle but can matter for natural-language questions.

**When to use:**
- Asking full-sentence questions ("how do I handle errors safely")
- Searching for design patterns described in docs
- Pure exploration (let the embedding decide relevance)

---

## 4. Language Filtering

Restrict search to specific programming languages.

```
search("hybrid_search", filter_language="python")
search("error NOT warning", filter_language="typescript")
```

**Supported languages:** PHP, TypeScript, JavaScript, Python, Swift

This filter works at the database level — non-matching languages are excluded before ranking, saving compute on irrelevant results.

**When to use:**
- Large multi-language codebases where you want results from one language only
- Avoiding false matches from comments (e.g., `error` appears in many language docs)

---

## 5. Source Type Filtering

Restrict search to code only, or documents only.

```
search("authentication", source_type="code")      # Only source files
search("authentication", source_type="markdown")  # Only documentation
```

Valid source types: `code`, `markdown`, `yaml`, `json`, `html`, `xml`, `text`, `txt`, `rst`, `csv`, `tsv`, `log`, `ini`, `cfg`, `toml`, `conf`, `pdf`, `asset`

### Document Search (Convenience Wrapper)

For document-only searches, use `doc_search_tool` instead of `search`:

```
doc_search_tool("authentication flow")  # Searches only docs, auto-excludes code
```

With format filtering:

```
doc_search_tool("database config", formats="yaml,toml")  # Only YAML and TOML files
```

---

## 6. Output Format Selection

Control how Tessera returns results.

### JSON (Default)

```
search("hybrid_search", output_format="json")
```

Returns full metadata: file path, line numbers, content, scores, rank sources. Best for programmatic processing.

### Markdown

```
search("hybrid_search", output_format="markdown")
```

Formats results as markdown with code blocks. Best for reading in a text editor or passing to an LLM.

Sample output:
```markdown
### 1. `search.py:100-110` (score: 0.89)
\`\`\`
def hybrid_search(query, query_embedding, db, limit=10):
    """Hybrid search combining keyword and semantic results."""
    results = []
    ...
\`\`\`
```

### CSV

```
search("hybrid_search", output_format="csv")
```

Tabular format with file path, line numbers, score, snippet. Best for import into spreadsheets or analysis tools.

### Files Only

```
search("hybrid_search", output_format="files")
```

Returns only file paths, one per line. Deduplicated, no snippets. Best when you just want to know which files to read.

---

## 7. Snippet Context Modes

Control how much surrounding code context Tessera shows in results.

### Lines Mode (Default)

```
search("hybrid_search", expand_context="lines")
```

Shows the best-matching lines surrounded by a **collapsed nesting skeleton**. This reveals the function/class hierarchy without showing all intermediate code.

Example output for a query inside a nested function:
```
32 | class MyClass:
        ...  (42 lines)
74 | def method():
        ...  (8 lines)
82 | key = value  <-- match is here
```

Each "..." shows how many lines are hidden. This lets you see structure without code bloat.

**max_depth parameter:** Limit how many nesting levels to show.

```
search("value", expand_context="lines", max_depth=1)
```

This shows only the immediate parent (e.g., the function containing the match), hiding any classes/modules wrapping it.

### Full Mode

```
search("hybrid_search", expand_context="full")
```

Expands all ancestor code without collapsing. Shows the complete function/class from definition to end, even if it's 100+ lines.

Use this when you need the full context to understand what's happening.

---

## 8. BM25 Short-Circuit (Informational)

**Note:** This is automatic and you don't control it. But understanding it explains why some searches are faster than others.

When a keyword search result is unambiguously the best match, Tessera skips expensive semantic and graph searches. Specifically:

- **Trigger:** Top BM25 result score ≥ 0.85 AND gap to second-place ≥ 0.15
- **Savings:** ~30-50ms per query (skips embedding, FAISS search, PageRank computation)

**Real benchmark data:**

| Query | Triggering? | Reason |
|-------|-----------|--------|
| `normalize_bm25_score` (top: 0.876, gap: 0.009) | No | Gap too small |
| `error handling` (top: 0.904, gap: 0.0003) | No | Gap too small |

Most queries don't trigger short-circuit because keyword results are rarely so dominant. When they do (e.g., searching for an exact function name), you get a latency boost automatically.

---

## 9. Combining Advanced Features

Here are realistic query patterns combining multiple advanced features:

### Find exact function implementation, fast

```
search("lex:hybrid_search", filter_language="python", output_format="files")
```

Keyword-only (fast), Python-specific, returns just file paths. Latency: <2ms.

### Search for patterns in docs only

```
doc_search_tool("retry with exponential backoff", formats="markdown")
```

Excludes code files, searches only documentation. Latency: ~10-50ms depending on doc size.

### Boost semantic for architecture questions

```
search(
    "how does the indexing pipeline work",
    weights="keyword=1.0,semantic=2.0,graph=0.8",
    expand_context="full"
)
```

Prioritizes semantic signal (good for conceptual questions), shows full context for understanding the system. Latency: ~20-30ms.

### Find "error" handling but not tests

```
search(
    "error NOT test",
    advanced_fts=True,
    filter_language="typescript"
)
```

Negation to exclude noisy test results, TypeScript-only. Latency: ~1-2ms (FTS5 is very fast).

### Proximity search for related concepts

```
search(
    "NEAR(graph PageRank, 3)",
    advanced_fts=True,
    weights="keyword=2.0"
)
```

Finds "graph" and "PageRank" discussed together, boosts keyword matching. Latency: ~1-2ms.

---

## 10. Troubleshooting Advanced Queries

### Advanced FTS operator syntax errors

**Problem:** Query with operators returns 0 results or an error.

**Fix:** FTS5 operator syntax is strict. If a query fails:
1. Fall back to `advanced_fts=False` (escapes all operators)
2. Simplify the query (e.g., `"phrase"` works, but `"phrase with many words"` might fail)
3. Use simpler operators first (phrases work most reliably; NEAR is pickier)

### Semantic results feel irrelevant (vec mode)

**Problem:** `search("vec:my query")` returns off-topic results.

**Fix:**
1. Try `search("lex:my query")` first to see if keyword results are better
2. Switch to HYDE: `search("hyde:my query")`
3. Boost keyword in hybrid: `weights="keyword=2.0,semantic=1.0"`
4. Provide more context in the query (longer, more descriptive queries embed better)

### Too many results (one language/format has too much content)

**Problem:** 1000+ chunks match, but you only see 10 results.

**Fix:**
1. Use `filter_language` to narrow by language
2. Use `source_type` to narrow by code vs docs
3. Use document `formats` to narrow by file type (e.g., `formats="yaml"` instead of all configs)

### Keyword search is too strict

**Problem:** `search("lex:function_name")` finds nothing, but you know it exists.

**Fix:**
1. Try `search("vec:function_name")` for semantic matching
2. Use prefix: `search("lex:function*", advanced_fts=True)`
3. Provide context: `search("lex:function_name authentication")`

---

## 11. Weight Tuning Guidelines

The default weights `keyword=1.5,semantic=1.0,graph=0.8` are tuned for general-purpose search. Here's how to think about tuning:

| Goal | Suggested Weights | Reasoning |
|------|-------------------|-----------|
| Find by exact name | `keyword=2.0,semantic=0.5` | Minimize noise, exact match only |
| Understand design | `keyword=1.0,semantic=2.0` | Maximize conceptual matching |
| Find critical code | `keyword=1.0,semantic=1.0,graph=2.0` | Structural importance wins |
| Balanced search | `keyword=1.5,semantic=1.0,graph=0.8` | Default (use this most of the time) |

**Don't over-tune:** For most searches, defaults work fine. Reserve custom weights for specific, repeatable queries where you notice results are consistently off.

---

## Summary

Advanced search in Tessera gives you control over:

- **Precision** (FTS5 operators, negation)
- **Speed** (keyword-only mode, short-circuit)
- **Signal balance** (custom weights)
- **Scope** (language, source type, format filters)
- **Presentation** (output format, snippet context)

Start with defaults. When results aren't what you expect, use one advanced feature at a time. The benchmark data in this guide shows real performance numbers — use them to decide if a feature is worth the latency cost for your use case.
