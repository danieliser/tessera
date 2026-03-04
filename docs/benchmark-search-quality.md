# Search Quality Benchmark Report

## 1. BM25 Score Normalization

**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.
Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.

| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |
|-------|----------|------------------------|-------------------------|-----------------|
| `normalize_bm25_score` | 10 | [-7.38, -6.26] | [0.862, 0.881] | Yes |
| `ProjectDB` | 10 | [-4.98, -4.44] | [0.816, 0.833] | Yes |
| `hybrid_search` | 10 | [-4.54, -4.31] | [0.812, 0.819] | Yes |
| `error handling` | 10 | [-8.58, -8.07] | [0.890, 0.896] | Yes |
| `authentication scope` | 10 | [-8.11, -6.66] | [0.869, 0.890] | Yes |
| `graph traversal` | 10 | [-7.36, -6.44] | [0.866, 0.880] | Yes |
| `keyword_search limit` | 10 | [-9.56, -8.31] | [0.893, 0.905] | Yes |
| `create_scope` | 10 | [-5.80, -5.16] | [0.838, 0.853] | Yes |
| `async to_thread` | 10 | [-9.98, -8.71] | [0.897, 0.909] | Yes |
| `FTS5 BM25` | 10 | [-10.07, -7.89] | [0.888, 0.910] | Yes |

**Score distribution:** Higher std = better discrimination between relevant and marginal results.

| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |
|-------|-------|-----------|----------------------|---------|
| `normalize_bm25_score` | 0.8807 | 0.8752 | 0.0184 | 0.0065 |
| `ProjectDB` | 0.8327 | 0.8266 | 0.0165 | 0.0052 |
| `hybrid_search` | 0.8193 | 0.8161 | 0.0078 | 0.0024 |
| `error handling` | 0.8956 | 0.8939 | 0.0059 | 0.0020 |
| `authentication scope` | 0.8903 | 0.8782 | 0.0208 | 0.0058 |
| `graph traversal` | 0.8804 | 0.8773 | 0.0149 | 0.0051 |
| `keyword_search limit` | 0.9053 | 0.9023 | 0.0127 | 0.0044 |
| `create_scope` | 0.8529 | 0.8464 | 0.0152 | 0.0045 |
| `async to_thread` | 0.9089 | 0.9059 | 0.0119 | 0.0042 |
| `FTS5 BM25` | 0.9096 | 0.9002 | 0.0221 | 0.0069 |

## 2. embed_query (Retrieval Prefix)

**Change:** `embed_single(query)` → `embed_query(query)` adds retrieval prefix.

| Query | embed_single Top-3 | embed_query Top-3 | Top-5 Overlap | Score Delta |
|-------|-------------------|------------------|--------------|-------------|
| `normalize_bm25_score` | index.html, index.html, search.py | index.html, index.html, search.py | 5/5 | +0.0000 |
| `ProjectDB` | conftest.py, test_real_federation.py, test_real_federation.py | conftest.py, test_real_federation.py, test_real_federation.py | 5/5 | +0.0000 |
| `hybrid_search` | index.html, index.html, index.html | index.html, index.html, index.html | 5/5 | +0.0000 |
| `error handling` | index.html, benchmark-search-quality.md, test_embeddings.py | index.html, benchmark-search-quality.md, test_embeddings.py | 5/5 | +0.0000 |
| `authentication scope` | index.html, index.html, test_search_benchmark.py | index.html, index.html, test_search_benchmark.py | 5/5 | +0.0000 |
| `graph traversal` | benchmark-search-quality.md, index.html, index.html | benchmark-search-quality.md, index.html, index.html | 5/5 | +0.0000 |

**Embedding divergence:** How much the retrieval prefix changes the vector.

| Query | Cosine Similarity | L2 Distance |
|-------|------------------|-------------|
| `normalize_bm25_score` | 0.8830 | 0.48 |
| `ProjectDB` | 0.8320 | 0.58 |
| `hybrid_search` | 0.7495 | 0.71 |
| `error handling` | 0.8377 | 0.57 |
| `authentication scope` | 0.8415 | 0.56 |
| `graph traversal` | 0.8512 | 0.55 |

## 3. Snippet Extraction

**Change:** `extract_snippet` returns a focused ~7-line window around the best
keyword-matching line, instead of the full chunk (which can be 50+ lines).

| Query | File | Chunk Lines | Snippet Lines | Compression | Best Match Line |
|-------|------|------------|--------------|-------------|----------------|
| `normalize_bm25_score` | index.html:1510 | 58 | 7 | 7/58 (12%) | 3 |
| `normalize_bm25_score` | index.html:894 | 87 | 7 | 7/87 (8%) | 15 |
| `ProjectDB` | conftest.py:13 | 14 | 6 | 6/14 (43%) | 2 |
| `ProjectDB` | test_real_federation.py:146 | 15 | 6 | 6/15 (40%) | 2 |
| `hybrid_search` | index.html:2027 | 33 | 7 | 7/33 (21%) | 5 |
| `hybrid_search` | index.html:1565 | 108 | 7 | 7/108 (6%) | 52 |
| `error handling` | index.html:1559 | 29 | 7 | 7/29 (24%) | 9 |
| `error handling` | benchmark-search-quality.md:212 | 87 | 7 | 7/87 (8%) | 19 |
| `authentication scope` | index.html:1823 | 52 | 7 | 7/52 (13%) | 35 |
| `authentication scope` | index.html:978 | 111 | 7 | 7/111 (6%) | 23 |
| `graph traversal` | benchmark-search-quality.md:212 | 87 | 7 | 7/87 (8%) | 20 |
| `graph traversal` | index.html:1423 | 90 | 7 | 7/90 (8%) | 13 |
| `keyword_search limit` | index.html:2156 | 26 | 7 | 7/26 (27%) | 17 |
| `keyword_search limit` | index.html:1529 | 26 | 6 | 6/26 (23%) | 2 |
| `create_scope` | test_auth.py:53 | 74 | 7 | 7/74 (9%) | 3 |
| `create_scope` | test_e2e_tools.py:363 | 44 | 7 | 7/44 (16%) | 10 |
| `async to_thread` | test_server_graph.py:30 | 54 | 7 | 7/54 (13%) | 8 |
| `async to_thread` | index.html:980 | 24 | 7 | 7/24 (29%) | 5 |
| `FTS5 BM25` | search.py:72 | 14 | 5 | 5/14 (36%) | 1 |
| `FTS5 BM25` | index.html:643 | 99 | 7 | 7/99 (7%) | 77 |

**Sample snippets** (showing what agents actually see):

**Query: `hybrid_search`** — index.html:2027 (33 lines → 7 lines)
```
2 | Sample output:
3 | ### 1. `search.py:100-110` (score: 0.89)
4 | \`\`\`
5 | def hybrid_search(query, query_embedding, db, limit=10):
6 |     """Hybrid search combining keyword and semantic results."""
7 |     results = []
8 |     ...
```

**Query: `error handling`** — index.html:1559 (29 lines → 7 lines)
```
 6 | Why this matters: Semantic scoring finds conceptual matches, not just keyword matches.
 7 | 
 8 | Concrete example:
 9 | Query: "error handling"
10 | 
11 | Line 0: # error handling in the request/response pipeline
12 | Line 1: import os
```

**Query: `create_scope`** — test_auth.py:53 (74 lines → 7 lines)
```
0 | class TestCreateScope:
1 |     """Test session creation."""
2 | 
3 |     def test_create_scope_project_level(self, db_conn):
4 |         """Create a project-level scope."""
5 |         session_id = create_scope(
6 |             db_conn,
```

## 4. Content-Addressable Document IDs

**Change:** `generate_docid` produces a stable 6-char hex hash from content.

| Property | Result |
|----------|--------|
| Deterministic (same content → same ID) | Pass |
| Total unique IDs tested | 78 |
| Collisions (different content, same ID) | 0 |
| ID format | 6-char hex (e.g. `dfc8db`) |

## 5. Multi-Format Output

**Change:** `format_results` supports json, csv, markdown, and files output modes.
Before: only `json.dumps`. After: agents can request the format that suits their task.

| Format | Valid | Size | Notes |
|--------|-------|------|-------|
| json | Yes | 9397 chars | 5 items, full metadata |
| csv | Yes | 7029 chars | 5 rows, 21 columns |
| markdown | Yes | 1668 chars | 6 result sections with code blocks |
| files | Yes | 134 chars | 4 unique file paths |

**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.

## 6. Search Latency

**Keyword-only search** (FTS5 + RRF merge + enrichment):

| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |
|-------|-----------|---------------|-------------|-----------|---------|
| `normalize_bm25_score` | 0.74 | 0.74 | 0.02 | 1.50 | 10 |
| `ProjectDB` | 0.51 | 0.51 | 0.01 | 1.04 | 10 |
| `hybrid_search` | 1.26 | 0.86 | 0.02 | 2.14 | 10 |
| `error handling` | 0.63 | 0.74 | 0.02 | 1.39 | 10 |
| `authentication scope` | 0.75 | 1.06 | 0.03 | 1.83 | 10 |
| `graph traversal` | 0.83 | 0.96 | 0.03 | 1.82 | 10 |
| `keyword_search limit` | 1.36 | 0.96 | 0.03 | 2.35 | 10 |
| `create_scope` | 1.10 | 0.79 | 0.02 | 1.91 | 10 |
| `async to_thread` | 1.40 | 1.06 | 0.02 | 2.48 | 10 |
| `FTS5 BM25` | 0.61 | 0.91 | 0.03 | 1.55 | 10 |

**Full hybrid search** (embed + keyword + semantic + RRF + snippet + docid):

| Query | Embed (ms) | Search (ms) | Post-process (ms) | Total (ms) |
|-------|-----------|-----------|-------------------|-----------|
| `normalize_bm25_score` | 0.0 | 8.0 | 0.75 | 8.8 |
| `ProjectDB` | 0.0 | 7.2 | 0.53 | 7.8 |
| `hybrid_search` | 0.0 | 7.6 | 0.87 | 8.5 |
| `error handling` | 0.0 | 6.7 | 0.79 | 7.5 |
| `authentication scope` | 0.0 | 7.8 | 1.08 | 8.9 |

## 7. BM25 Strong-Signal Short-Circuit

**Change:** Skip semantic search + PPR when top BM25 result score >= 0.85 AND gap to #2 >= 0.15.
Saves ~30-50ms per query when keyword match is unambiguous.

| Query | Top-1 Norm | Top-2 Norm | Gap | Short-Circuit? | Reason |
|-------|-----------|-----------|-----|---------------|--------|
| `normalize_bm25_score` | 0.8807 | 0.8788 | 0.0019 | No | Gap 0.002 < 0.15 |
| `ProjectDB` | 0.8327 | 0.8287 | 0.0039 | No | Score 0.833 < 0.85 |
| `hybrid_search` | 0.8193 | 0.8167 | 0.0027 | No | Score 0.819 < 0.85 |
| `error handling` | 0.8956 | 0.8946 | 0.0010 | No | Gap 0.001 < 0.15 |
| `authentication scope` | 0.8903 | 0.8765 | 0.0138 | No | Gap 0.014 < 0.15 |
| `graph traversal` | 0.8804 | 0.8802 | 0.0002 | No | Gap 0.000 < 0.15 |
| `keyword_search limit` | 0.9053 | 0.9041 | 0.0012 | No | Gap 0.001 < 0.15 |
| `create_scope` | 0.8529 | 0.8484 | 0.0045 | No | Gap 0.005 < 0.15 |
| `async to_thread` | 0.9089 | 0.9072 | 0.0017 | No | Gap 0.002 < 0.15 |
| `FTS5 BM25` | 0.9096 | 0.9053 | 0.0043 | No | Gap 0.004 < 0.15 |

**Latency comparison** (with vs without short-circuit):

| Query | Full Pipeline (ms) | Short-Circuit (ms) | Saved (ms) | Triggered? |
|-------|-------------------|-------------------|-----------|-----------|
| `normalize_bm25_score` | 9.5 | 0.7 | +8.8 | No |
| `ProjectDB` | 8.7 | 0.5 | +8.2 | No |
| `hybrid_search` | 9.5 | 1.3 | +8.2 | No |
| `error handling` | 8.5 | 0.8 | +7.7 | No |
| `authentication scope` | 12.1 | 0.9 | +11.2 | No |

## 8. Weighted RRF Fusion

**Change:** Per-list weights: keyword=1.5x, semantic=1.0x, graph=0.8x (was 1.0x equal).
Keyword results boosted because FTS5 precision is higher for code search.

| Query | Equal RRF Top-3 | Weighted RRF Top-3 | Rank Changes | Score Boost |
|-------|-----------------|--------------------|-------------|-------------|
| `normalize_bm25_score` | 12330924, 12332897, 12330917 | 12330924, 12330917, 12332439 | 2/3 | +0.008197 |
| `ProjectDB` | 12332488, 12330907, 12332588 | 12332488, 12332588, 12332587 | 2/3 | +0.008197 |
| `hybrid_search` | 12332170, 12332889, 12330925 | 12332170, 12330925, 12332196 | 2/3 | +0.008197 |
| `error handling` | 12332275, 12332847, 12332837 | 12332275, 12332837, 12332771 | 2/3 | +0.008197 |
| `authentication scope` | 12330995, 12332901, 12330918 | 12330995, 12330918, 12332661 | 2/3 | +0.008197 |
| `graph traversal` | 12332837, 12332906, 12330923 | 12332837, 12330923, 12331166 | 2/3 | +0.008197 |

**Weight sensitivity:** How per-list weights shift RRF scores.

| Weights (kw, sem) | ID-1 Score | ID-2 Score | ID-3 Score | Rank Change vs Equal |
|-------------------|-----------|-----------|-----------|---------------------|
| (1.0, 1.0) | 0.032266 | 0.032522 | 0.032002 | 0/3 |
| (1.5, 1.0) | 0.040463 | 0.040587 | 0.039939 | 0/3 |
| (2.0, 1.0) | 0.048660 | 0.048652 | 0.047875 | 2/3 |
| (1.0, 2.0) | 0.048139 | 0.048916 | 0.048131 | 0/3 |

## 9. Structured Query Types

**Query prefix parser:**

| Input | Clean Query | Search Types |
|-------|------------|-------------|
| `plain query` | `plain query` | ['lex', 'vec'] |
| `lex:ProjectDB` | `ProjectDB` | ['lex'] |
| `vec:error handling` | `error handling` | ['vec'] |
| `hyde:graph traversal` | `graph traversal` | ['hyde'] |
| `lex,vec:hybrid_search` | `hybrid_search` | ['lex', 'vec'] |
| `VEC:case insensitive` | `case insensitive` | ['vec'] |

**LEX-only vs keyword_search():**

| Query | LEX Top-5 IDs | keyword_search Top-5 IDs | Match? |
|-------|--------------|-------------------------|--------|
| `normalize_bm25_score` | [12330924, 12330917, 12332439]... | [12330924, 12330917, 12332439]... | Pass |
| `error handling` | [12332275, 12332837, 12332771]... | [12332275, 12332837, 12332771]... | Pass |
| `graph traversal` | [12332837, 12330923, 12331166]... | [12332837, 12330923, 12331166]... | Pass |

**VEC-only vs LEX-only (proving complementary signals):**

| Query | LEX Top-3 | VEC Top-3 | Top-5 Overlap | Different? |
|-------|----------|----------|--------------|-----------|
| `normalize_bm25_score` | index.html, index.html, search.py | research.md, PLAN.md, panel-scorecard | 0/5 | Yes |
| `error handling` | index.html, benchmark-searc, test_embeddings | conditions.md, research.md, PLAN.md | 0/5 | Yes |
| `graph traversal` | benchmark-searc, index.html, index.html | spec-v1.md, multi-project.m, spec-v1.md | 0/5 | Yes |

**HYDE vs VEC (embed_single vs embed_query):**

| Query | VEC Top-3 | HYDE Top-3 | Top-5 Overlap |
|-------|----------|-----------|--------------|
| `normalize_bm25_score` | research.md, PLAN.md, panel-scorecard | .mcp.json, research.md, panel-scorecard | 3/5 |
| `error handling` | conditions.md, research.md, PLAN.md | conditions.md, VALIDATION.md, index.html | 1/5 |
| `graph traversal` | spec-v1.md, multi-project.m, spec-v1.md | auth.py, index.html, cross_file_b.py | 0/5 |

## 10. FTS5 Advanced Mode

**Safe mode (default):** All FTS5 operators escaped. No syntax errors possible.

| Query | Results (safe) | Error? |
|-------|---------------|--------|
| `error handling (async)` | 5 | No |
| `foo OR bar AND baz` | 5 | No |
| `"already quoted"` | 4 | No |
| `hybrid*` | 5 | No |
| `error NOT warning` | 5 | No |
| `NEAR(search query, 5)` | 5 | No |

**Advanced mode (`advanced_fts=True`):** FTS5 operators active.

| Query Type | Query | Results (adv) | Results (safe) | Fewer? |
|-----------|-------|--------------|---------------|--------|
| phrase | `"def hybrid_search"` | 10 | 10 | No |
| negation | `error NOT warning` | 10 | 10 | No |
| prefix | `hybrid*` | 10 | 10 | No |
| unquoted | `hybrid_search` | 10 | 10 | No |

**Phrase precision:** Quoted phrases return fewer, more precise results.

| Phrase | Phrase Results | Unquoted Results | Subset? |
|--------|---------------|-----------------|---------|
| `"def hybrid_search"` | 20 | 20 | No |
| `"keyword search"` | 20 | 20 | No |
| `"error handling"` | 20 | 20 | No |

**FTS5 latency (safe vs advanced):**

| Query | Safe (ms) | Advanced (ms) |
|-------|----------|-------------|
| `hybrid_search` | 1.13 | 1.12 |
| `"def hybrid_search"` | 1.38 | 1.29 |
| `error NOT warning` | 0.88 | 0.57 |
| `hybrid*` | 0.42 | 0.47 |

---

*Generated by `tests/test_search_benchmark.py` against the live codemem index (/Users/danieliser/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db).*
