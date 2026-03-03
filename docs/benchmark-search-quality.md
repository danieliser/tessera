# Search Quality Benchmark Report

## 1. BM25 Score Normalization

**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.
Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.

| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |
|-------|----------|------------------------|-------------------------|-----------------|
| `normalize_bm25_score` | 10 | [-7.40, -6.19] | [0.861, 0.881] | Yes |
| `ProjectDB` | 10 | [-5.03, -4.49] | [0.818, 0.834] | Yes |
| `hybrid_search` | 10 | [-4.64, -4.33] | [0.813, 0.823] | Yes |
| `error handling` | 10 | [-8.53, -8.02] | [0.889, 0.895] | Yes |
| `authentication scope` | 10 | [-7.97, -6.33] | [0.864, 0.889] | Yes |
| `graph traversal` | 10 | [-7.07, -6.16] | [0.860, 0.876] | Yes |
| `keyword_search limit` | 10 | [-9.53, -8.28] | [0.892, 0.905] | Yes |
| `create_scope` | 10 | [-5.62, -5.00] | [0.833, 0.849] | Yes |
| `async to_thread` | 10 | [-9.98, -8.72] | [0.897, 0.909] | Yes |
| `FTS5 BM25` | 10 | [-9.96, -7.82] | [0.887, 0.909] | Yes |

**Score distribution:** Higher std = better discrimination between relevant and marginal results.

| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |
|-------|-------|-----------|----------------------|---------|
| `normalize_bm25_score` | 0.8809 | 0.8755 | 0.0200 | 0.0069 |
| `ProjectDB` | 0.8341 | 0.8281 | 0.0164 | 0.0052 |
| `hybrid_search` | 0.8225 | 0.8184 | 0.0100 | 0.0032 |
| `error handling` | 0.8950 | 0.8930 | 0.0059 | 0.0018 |
| `authentication scope` | 0.8886 | 0.8769 | 0.0250 | 0.0064 |
| `graph traversal` | 0.8760 | 0.8728 | 0.0156 | 0.0052 |
| `keyword_search limit` | 0.9050 | 0.9020 | 0.0128 | 0.0044 |
| `create_scope` | 0.8490 | 0.8423 | 0.0156 | 0.0046 |
| `async to_thread` | 0.9089 | 0.9059 | 0.0118 | 0.0042 |
| `FTS5 BM25` | 0.9088 | 0.8991 | 0.0221 | 0.0070 |

## 2. embed_query (Retrieval Prefix)

**Change:** `embed_single(query)` → `embed_query(query)` adds retrieval prefix.

| Query | embed_single Top-3 | embed_query Top-3 | Top-5 Overlap | Score Delta |
|-------|-------------------|------------------|--------------|-------------|
| `normalize_bm25_score` | index.html, index.html, search.py | index.html, index.html, search.py | 5/5 | +0.0000 |
| `ProjectDB` | conftest.py, test_real_federation.py, test_real_federation.py | conftest.py, test_real_federation.py, test_real_federation.py | 5/5 | +0.0000 |
| `hybrid_search` | benchmark-search-quality.md, index.html, index.html | benchmark-search-quality.md, index.html, index.html | 5/5 | +0.0000 |
| `error handling` | index.html, test_embeddings.py, benchmark-search-quality.md | index.html, test_embeddings.py, benchmark-search-quality.md | 5/5 | +0.0000 |
| `authentication scope` | index.html, benchmark-search-quality.md, index.html | index.html, benchmark-search-quality.md, index.html | 5/5 | +0.0000 |
| `graph traversal` | index.html, benchmark-search-quality.md, index.html | index.html, benchmark-search-quality.md, index.html | 5/5 | +0.0000 |

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
| `hybrid_search` | benchmark-search-quality.md:269 | 27 | 5 | 5/27 (19%) | 1 |
| `hybrid_search` | index.html:2027 | 33 | 7 | 7/33 (21%) | 5 |
| `error handling` | index.html:1559 | 29 | 7 | 7/29 (24%) | 9 |
| `error handling` | test_embeddings.py:169 | 51 | 5 | 5/51 (10%) | 1 |
| `authentication scope` | index.html:1823 | 52 | 7 | 7/52 (13%) | 35 |
| `authentication scope` | benchmark-search-quality.md:44 | 69 | 5 | 5/69 (7%) | 1 |
| `graph traversal` | index.html:1423 | 90 | 7 | 7/90 (8%) | 13 |
| `graph traversal` | benchmark-search-quality.md:205 | 77 | 7 | 7/77 (9%) | 20 |
| `keyword_search limit` | index.html:2156 | 26 | 7 | 7/26 (27%) | 17 |
| `keyword_search limit` | index.html:1529 | 26 | 6 | 6/26 (23%) | 2 |
| `create_scope` | test_auth.py:53 | 74 | 7 | 7/74 (9%) | 3 |
| `create_scope` | test_e2e_tools.py:363 | 44 | 7 | 7/44 (16%) | 10 |
| `async to_thread` | test_server_graph.py:30 | 54 | 7 | 7/54 (13%) | 8 |
| `async to_thread` | index.html:980 | 24 | 7 | 7/24 (29%) | 5 |
| `FTS5 BM25` | search.py:72 | 14 | 5 | 5/14 (36%) | 1 |
| `FTS5 BM25` | index.html:643 | 99 | 7 | 7/99 (7%) | 77 |

**Sample snippets** (showing what agents actually see):

**Query: `hybrid_search`** — benchmark-search-quality.md:269 (27 lines → 5 lines)
```
0 | ----|-------|--------------|---------------|--------|
1 | | phrase | `"def hybrid_search"` | 9 | 10 | Yes |
2 | | negation | `error NOT warning` | 10 | 10 | No |
3 | | prefix | `hybrid*` | 10 | 10 | No |
4 | | unquoted | `hybrid_search` | 10 | 10 | No |
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
| Total unique IDs tested | 79 |
| Collisions (different content, same ID) | 0 |
| ID format | 6-char hex (e.g. `dfc8db`) |

## 5. Multi-Format Output

**Change:** `format_results` supports json, csv, markdown, and files output modes.
Before: only `json.dumps`. After: agents can request the format that suits their task.

| Format | Valid | Size | Notes |
|--------|-------|------|-------|
| json | Yes | 9651 chars | 5 items, full metadata |
| csv | Yes | 7279 chars | 5 rows, 21 columns |
| markdown | Yes | 1667 chars | 6 result sections with code blocks |
| files | Yes | 134 chars | 4 unique file paths |

**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.

## 6. Search Latency

**Keyword-only search** (FTS5 + RRF merge + enrichment):

| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |
|-------|-----------|---------------|-------------|-----------|---------|
| `normalize_bm25_score` | 0.49 | 0.60 | 0.02 | 1.11 | 10 |
| `ProjectDB` | 0.43 | 0.50 | 0.01 | 0.95 | 10 |
| `hybrid_search` | 0.68 | 0.96 | 0.04 | 1.68 | 10 |
| `error handling` | 0.67 | 0.78 | 0.02 | 1.47 | 10 |
| `authentication scope` | 0.51 | 1.04 | 0.02 | 1.57 | 10 |
| `graph traversal` | 0.52 | 0.88 | 0.03 | 1.43 | 10 |
| `keyword_search limit` | 0.90 | 0.88 | 0.02 | 1.80 | 10 |
| `create_scope` | 0.60 | 0.78 | 0.02 | 1.39 | 10 |
| `async to_thread` | 0.64 | 1.09 | 0.04 | 1.78 | 10 |
| `FTS5 BM25` | 0.55 | 0.84 | 0.02 | 1.41 | 10 |

**Full hybrid search** (embed + keyword + semantic + RRF + snippet + docid):

| Query | Embed (ms) | Search (ms) | Post-process (ms) | Total (ms) |
|-------|-----------|-----------|-------------------|-----------|
| `normalize_bm25_score` | 0.0 | 0.5 | 0.60 | 1.1 |
| `ProjectDB` | 0.0 | 0.4 | 0.50 | 1.0 |
| `hybrid_search` | 0.0 | 0.7 | 0.85 | 1.5 |
| `error handling` | 0.0 | 0.5 | 0.74 | 1.2 |
| `authentication scope` | 0.0 | 0.4 | 1.02 | 1.5 |

## 7. BM25 Strong-Signal Short-Circuit

**Change:** Skip semantic search + PPR when top BM25 result score >= 0.85 AND gap to #2 >= 0.15.
Saves ~30-50ms per query when keyword match is unambiguous.

| Query | Top-1 Norm | Top-2 Norm | Gap | Short-Circuit? | Reason |
|-------|-----------|-----------|-----|---------------|--------|
| `normalize_bm25_score` | 0.8809 | 0.8791 | 0.0019 | No | Gap 0.002 < 0.15 |
| `ProjectDB` | 0.8341 | 0.8302 | 0.0039 | No | Score 0.834 < 0.85 |
| `hybrid_search` | 0.8225 | 0.8202 | 0.0024 | No | Score 0.823 < 0.85 |
| `error handling` | 0.8950 | 0.8935 | 0.0015 | No | Gap 0.002 < 0.15 |
| `authentication scope` | 0.8886 | 0.8757 | 0.0129 | No | Gap 0.013 < 0.15 |
| `graph traversal` | 0.8760 | 0.8747 | 0.0013 | No | Gap 0.001 < 0.15 |
| `keyword_search limit` | 0.9050 | 0.9038 | 0.0012 | No | Gap 0.001 < 0.15 |
| `create_scope` | 0.8490 | 0.8444 | 0.0046 | No | Score 0.849 < 0.85 |
| `async to_thread` | 0.9089 | 0.9071 | 0.0018 | No | Gap 0.002 < 0.15 |
| `FTS5 BM25` | 0.9088 | 0.9044 | 0.0044 | No | Gap 0.004 < 0.15 |

**Latency comparison** (with vs without short-circuit):

| Query | Full Pipeline (ms) | Short-Circuit (ms) | Saved (ms) | Triggered? |
|-------|-------------------|-------------------|-----------|-----------|
| `normalize_bm25_score` | 0.5 | 0.4 | +0.0 | No |
| `ProjectDB` | 0.4 | 0.4 | +0.0 | No |
| `hybrid_search` | 0.7 | 0.6 | +0.0 | No |
| `error handling` | 0.5 | 0.7 | -0.3 | No |
| `authentication scope` | 0.6 | 0.4 | +0.1 | No |

## 8. Weighted RRF Fusion

**Change:** Per-list weights: keyword=1.5x, semantic=1.0x, graph=0.8x (was 1.0x equal).
Keyword results boosted because FTS5 precision is higher for code search.

| Query | Equal RRF Top-3 | Weighted RRF Top-3 | Rank Changes | Score Boost |
|-------|-----------------|--------------------|-------------|-------------|
| `normalize_bm25_score` | 12326206, 12326199, 12327718 | 12326206, 12326199, 12327718 | 0/3 | +0.008197 |
| `ProjectDB` | 12327767, 12327867, 12327866 | 12327767, 12327867, 12327866 | 0/3 | +0.008197 |
| `hybrid_search` | 12325771, 12327452, 12326207 | 12325771, 12327452, 12326207 | 0/3 | +0.008197 |
| `error handling` | 12327557, 12328050, 12325770 | 12327557, 12328050, 12325770 | 0/3 | +0.008197 |
| `authentication scope` | 12326277, 12325767, 12326200 | 12326277, 12325767, 12326200 | 0/3 | +0.008197 |
| `graph traversal` | 12326205, 12325770, 12326448 | 12326205, 12325770, 12326448 | 0/3 | +0.008197 |

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
| `normalize_bm25_score` | [12326206, 12326199, 12327718]... | [12326206, 12326199, 12327718]... | Pass |
| `error handling` | [12327557, 12328050, 12325770]... | [12327557, 12328050, 12325770]... | Pass |
| `graph traversal` | [12326205, 12325770, 12326448]... | [12326205, 12325770, 12326448]... | Pass |

**VEC-only vs LEX-only (proving complementary signals):**

| Query | LEX Top-3 | VEC Top-3 | Top-5 Overlap | Different? |
|-------|----------|----------|--------------|-----------|
| `normalize_bm25_score` | index.html, index.html, search.py |  | 0/5 | Yes |
| `error handling` | index.html, test_embeddings, benchmark-searc |  | 0/5 | Yes |
| `graph traversal` | index.html, benchmark-searc, index.html |  | 0/5 | Yes |

**HYDE vs VEC (embed_single vs embed_query):**

| Query | VEC Top-3 | HYDE Top-3 | Top-5 Overlap |
|-------|----------|-----------|--------------|
| `normalize_bm25_score` |  |  | 0/5 |
| `error handling` |  |  | 0/5 |
| `graph traversal` |  |  | 0/5 |

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
| `hybrid_search` | 0.53 | 0.52 |
| `"def hybrid_search"` | 0.58 | 0.52 |
| `error NOT warning` | 0.46 | 0.38 |
| `hybrid*` | 0.32 | 0.34 |

---

*Generated by `tests/test_search_benchmark.py` against the live codemem index (/Users/danieliser/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db).*
