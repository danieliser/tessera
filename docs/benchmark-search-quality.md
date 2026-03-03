# Search Quality Benchmark Report

## 1. BM25 Score Normalization

**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.
Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.

| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |
|-------|----------|------------------------|-------------------------|-----------------|
| `normalize_bm25_score` | 10 | [-7.06, -4.72] | [0.825, 0.876] | Yes |
| `ProjectDB` | 10 | [-5.45, -4.51] | [0.819, 0.845] | Yes |
| `hybrid_search` | 10 | [-5.25, -4.48] | [0.817, 0.840] | Yes |
| `error handling` | 10 | [-9.43, -8.29] | [0.892, 0.904] | Yes |
| `authentication scope` | 10 | [-8.59, -5.57] | [0.848, 0.896] | Yes |
| `graph traversal` | 10 | [-7.88, -5.70] | [0.851, 0.887] | Yes |
| `keyword_search limit` | 10 | [-10.09, -8.29] | [0.892, 0.910] | Yes |
| `create_scope` | 10 | [-6.15, -5.25] | [0.840, 0.860] | Yes |
| `async to_thread` | 10 | [-10.70, -7.73] | [0.885, 0.915] | Yes |
| `FTS5 BM25` | 10 | [-11.01, -7.29] | [0.879, 0.917] | Yes |

**Score distribution:** Higher std = better discrimination between relevant and marginal results.

| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |
|-------|-------|-----------|----------------------|---------|
| `normalize_bm25_score` | 0.8760 | 0.8610 | 0.0509 | 0.0132 |
| `ProjectDB` | 0.8451 | 0.8361 | 0.0265 | 0.0096 |
| `hybrid_search` | 0.8399 | 0.8339 | 0.0225 | 0.0066 |
| `error handling` | 0.9041 | 0.9020 | 0.0118 | 0.0043 |
| `authentication scope` | 0.8957 | 0.8878 | 0.0479 | 0.0135 |
| `graph traversal` | 0.8874 | 0.8819 | 0.0367 | 0.0124 |
| `keyword_search limit` | 0.9098 | 0.9038 | 0.0174 | 0.0053 |
| `create_scope` | 0.8601 | 0.8511 | 0.0201 | 0.0070 |
| `async to_thread` | 0.9145 | 0.9000 | 0.0291 | 0.0085 |
| `FTS5 BM25` | 0.9167 | 0.9037 | 0.0374 | 0.0105 |

## 2. embed_query (Retrieval Prefix)

**Change:** `embed_single(query)` → `embed_query(query)` adds retrieval prefix.

| Query | embed_single Top-3 | embed_query Top-3 | Top-5 Overlap | Score Delta |
|-------|-------------------|------------------|--------------|-------------|
| `normalize_bm25_score` | benchmark-search-quality.md, test_search_benchmark.py, benchmark-search-quality.md | benchmark-search-quality.md, test_search_benchmark.py, benchmark-search-quality.md | 5/5 | +0.0000 |
| `ProjectDB` | test_db_graph.py, conftest.py, test_real_federation.py | test_db_graph.py, conftest.py, test_real_federation.py | 5/5 | -0.0005 |
| `hybrid_search` | benchmark-search-quality.md, spec-v1.md, spec-v2.md | test_search_benchmark.py, spec-v1.md, spec-v2.md | 5/5 | +0.0007 |
| `error handling` | test_embeddings.py, benchmark-search-quality.md, test_search_benchmark.py | test_embeddings.py, benchmark-search-quality.md, test_search_benchmark.py | 5/5 | +0.0000 |
| `authentication scope` | research.md, benchmark-search-quality.md, test_search_benchmark.py | research.md, benchmark-search-quality.md, test_search_benchmark.py | 5/5 | +0.0000 |
| `graph traversal` | benchmark-search-quality.md, test_search_benchmark.py, research.md | intake.md, benchmark-search-quality.md, test_search_benchmark.py | 5/5 | +0.0146 |

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
| `normalize_bm25_score` | test_search_benchmark.py:666 | 8 | 7 | 7/8 (88%) | 4 |
| `normalize_bm25_score` | benchmark-search-quality.md:239 | 18 | 7 | 7/18 (39%) | 6 |
| `ProjectDB` | conftest.py:13 | 14 | 6 | 6/14 (43%) | 2 |
| `ProjectDB` | test_db_graph.py:0 | 5 | 4 | 4/5 (80%) | 0 |
| `hybrid_search` | test_search_benchmark.py:101 | 3 | 3 | 3/3 (100%) | 1 |
| `hybrid_search` | benchmark-search-quality.md:281 | 18 | 5 | 5/18 (28%) | 1 |
| `error handling` | test_embeddings.py:169 | 51 | 5 | 5/51 (10%) | 1 |
| `error handling` | benchmark-search-quality.md:216 | 24 | 7 | 7/24 (29%) | 8 |
| `authentication scope` | research.md:646 | 6 | 6 | 6/6 (100%) | 2 |
| `authentication scope` | benchmark-search-quality.md:45 | 15 | 5 | 5/15 (33%) | 1 |
| `graph traversal` | benchmark-search-quality.md:216 | 24 | 7 | 7/24 (29%) | 9 |
| `graph traversal` | test_search_benchmark.py:666 | 8 | 5 | 5/8 (62%) | 6 |
| `keyword_search limit` | spec-v1.md:819 | 33 | 7 | 7/33 (21%) | 6 |
| `keyword_search limit` | research.md:538 | 32 | 7 | 7/32 (22%) | 9 |
| `create_scope` | _scope.py:0 | 12 | 4 | 4/12 (33%) | 0 |
| `create_scope` | test_auth.py:53 | 74 | 7 | 7/74 (9%) | 3 |
| `async to_thread` | PLAN.md:64 | 4 | 4 | 4/4 (100%) | 2 |
| `async to_thread` | CLAUDE.md:60 | 7 | 5 | 5/7 (71%) | 5 |
| `FTS5 BM25` | search.py:71 | 14 | 5 | 5/14 (36%) | 1 |
| `FTS5 BM25` | CHANGELOG.md:34 | 7 | 5 | 5/7 (71%) | 1 |

**Sample snippets** (showing what agents actually see):

**Query: `hybrid_search`** — test_search_benchmark.py:101 (3 lines → 3 lines)
```
0 | def _search(db, query, limit=10):
1 |     """Run keyword-only hybrid search (enriched results with file_path)."""
2 |     return hybrid_search(query, query_embedding=None, db=db, limit=limit)
```

**Query: `error handling`** — test_embeddings.py:169 (51 lines → 5 lines)
```
0 | class TestEmbeddingErrorHandling:
1 |     """Test error handling."""
2 | 
3 |     @patch("httpx.Client.post")
4 |     def test_embed_request_error(self, mock_post):
```

**Query: `create_scope`** — _scope.py:0 (12 lines → 4 lines)
```
0 | """Scope tools: create_scope, revoke_scope."""
1 | 
2 | import asyncio
3 | import json
```

## 4. Content-Addressable Document IDs

**Change:** `generate_docid` produces a stable 6-char hex hash from content.

| Property | Result |
|----------|--------|
| Deterministic (same content → same ID) | Pass |
| Total unique IDs tested | 73 |
| Collisions (different content, same ID) | 0 |
| ID format | 6-char hex (e.g. `6a491d`) |

## 5. Multi-Format Output

**Change:** `format_results` supports json, csv, markdown, and files output modes.
Before: only `json.dumps`. After: agents can request the format that suits their task.

| Format | Valid | Size | Notes |
|--------|-------|------|-------|
| json | Yes | 4874 chars | 5 items, full metadata |
| csv | Yes | 2911 chars | 5 rows, 21 columns |
| markdown | Yes | 1001 chars | 5 result sections with code blocks |
| files | Yes | 181 chars | 5 unique file paths |

**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.

## 6. Search Latency

**Keyword-only search** (FTS5 + RRF merge + enrichment):

| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |
|-------|-----------|---------------|-------------|-----------|---------|
| `normalize_bm25_score` | 0.57 | 0.35 | 0.01 | 0.93 | 10 |
| `ProjectDB` | 0.41 | 0.39 | 0.01 | 0.81 | 10 |
| `hybrid_search` | 0.83 | 0.31 | 0.01 | 1.16 | 10 |
| `error handling` | 0.50 | 0.32 | 0.01 | 0.83 | 10 |
| `authentication scope` | 0.52 | 0.35 | 0.01 | 0.88 | 10 |
| `graph traversal` | 0.56 | 0.33 | 0.01 | 0.90 | 10 |
| `keyword_search limit` | 0.89 | 0.60 | 0.02 | 1.51 | 10 |
| `create_scope` | 0.68 | 0.42 | 0.01 | 1.11 | 10 |
| `async to_thread` | 1.02 | 0.38 | 0.01 | 1.40 | 10 |
| `FTS5 BM25` | 0.42 | 0.35 | 0.01 | 0.78 | 10 |

**Full hybrid search** (embed + keyword + semantic + RRF + snippet + docid):

| Query | Embed (ms) | Search (ms) | Post-process (ms) | Total (ms) |
|-------|-----------|-----------|-------------------|-----------|
| `normalize_bm25_score` | 0.0 | 8.8 | 0.36 | 9.2 |
| `ProjectDB` | 0.0 | 13.9 | 0.41 | 14.3 |
| `hybrid_search` | 0.0 | 9.8 | 0.36 | 10.1 |
| `error handling` | 0.0 | 9.3 | 0.36 | 9.7 |
| `authentication scope` | 0.0 | 9.6 | 0.38 | 10.0 |

## 7. BM25 Strong-Signal Short-Circuit

**Change:** Skip semantic search + PPR when top BM25 result score >= 0.85 AND gap to #2 >= 0.15.
Saves ~30-50ms per query when keyword match is unambiguous.

| Query | Top-1 Norm | Top-2 Norm | Gap | Short-Circuit? | Reason |
|-------|-----------|-----------|-----|---------------|--------|
| `normalize_bm25_score` | 0.8760 | 0.8671 | 0.0089 | No | Gap 0.009 < 0.15 |
| `ProjectDB` | 0.8451 | 0.8403 | 0.0047 | No | Score 0.845 < 0.85 |
| `hybrid_search` | 0.8399 | 0.8355 | 0.0044 | No | Score 0.840 < 0.85 |
| `error handling` | 0.9041 | 0.9038 | 0.0003 | No | Gap 0.000 < 0.15 |
| `authentication scope` | 0.8957 | 0.8894 | 0.0064 | No | Gap 0.006 < 0.15 |
| `graph traversal` | 0.8874 | 0.8867 | 0.0007 | No | Gap 0.001 < 0.15 |
| `keyword_search limit` | 0.9098 | 0.9065 | 0.0033 | No | Gap 0.003 < 0.15 |
| `create_scope` | 0.8601 | 0.8568 | 0.0032 | No | Gap 0.003 < 0.15 |
| `async to_thread` | 0.9145 | 0.9036 | 0.0109 | No | Gap 0.011 < 0.15 |
| `FTS5 BM25` | 0.9167 | 0.9108 | 0.0059 | No | Gap 0.006 < 0.15 |

**Latency comparison** (with vs without short-circuit):

| Query | Full Pipeline (ms) | Short-Circuit (ms) | Saved (ms) | Triggered? |
|-------|-------------------|-------------------|-----------|-----------|
| `normalize_bm25_score` | 10.6 | 0.9 | +9.7 | No |
| `ProjectDB` | 9.0 | 0.5 | +8.5 | No |
| `hybrid_search` | 11.3 | 1.1 | +10.2 | No |
| `error handling` | 12.0 | 0.6 | +11.4 | No |
| `authentication scope` | 15.8 | 0.7 | +15.1 | No |

## 8. Weighted RRF Fusion

**Change:** Per-list weights: keyword=1.5x, semantic=1.0x, graph=0.8x (was 1.0x equal).
Keyword results boosted because FTS5 precision is higher for code search.

| Query | Equal RRF Top-3 | Weighted RRF Top-3 | Rank Changes | Score Boost |
|-------|-----------------|--------------------|-------------|-------------|
| `normalize_bm25_score` | 12325384, 12323761, 12325629 | 12325629, 12323761, 12325384 | 2/3 | +0.007990 |
| `ProjectDB` | 12325695, 12325435, 12325294 | 12325695, 12325435, 12325535 | 1/3 | +0.008065 |
| `hybrid_search` | 12325612, 12324016, 12324180 | 12325612, 12324016, 12324180 | 0/3 | +0.008197 |
| `error handling` | 12325569, 12324780, 12325715 | 12325569, 12324780, 12325715 | 0/3 | +0.007353 |
| `authentication scope` | 12324473, 12324269, 12323754 | 12324473, 12323754, 12325608 | 2/3 | +0.008197 |
| `graph traversal` | 12324866, 12323766, 12323898 | 12324866, 12323766, 12325629 | 1/3 | +0.007692 |

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
| `normalize_bm25_score` | [12325629, 12323767, 12325387]... | [12325629, 12323767, 12325387]... | Pass |
| `error handling` | [12325715, 12323766, 12325629]... | [12325715, 12323766, 12325629]... | Pass |
| `graph traversal` | [12323766, 12325629, 12323865]... | [12323766, 12325629, 12323865]... | Pass |

**VEC-only vs LEX-only (proving complementary signals):**

| Query | LEX Top-3 | VEC Top-3 | Top-5 Overlap | Different? |
|-------|----------|----------|--------------|-----------|
| `normalize_bm25_score` | test_search_ben, benchmark-searc, search.py | search.py, test_search_ben, benchmark-searc | 1/5 | Yes |
| `error handling` | test_embeddings, benchmark-searc, test_search_ben | test_server.py, spec-v2.md, spec-v2.md | 0/5 | Yes |
| `graph traversal` | benchmark-searc, test_search_ben, research.md | research.md, intake.md, spec-v1.md | 1/5 | Yes |

**HYDE vs VEC (embed_single vs embed_query):**

| Query | VEC Top-3 | HYDE Top-3 | Top-5 Overlap |
|-------|----------|-----------|--------------|
| `normalize_bm25_score` | search.py, test_search_ben, benchmark-searc | test_search_ben, search.py, benchmark-searc | 5/5 |
| `error handling` | test_server.py, spec-v2.md, spec-v2.md | test_server.py, spec-v2.md, auth.py | 3/5 |
| `graph traversal` | research.md, intake.md, spec-v1.md | spec-v1.md, research.md, research-depend | 1/5 |

## 10. FTS5 Advanced Mode

**Safe mode (default):** All FTS5 operators escaped. No syntax errors possible.

| Query | Results (safe) | Error? |
|-------|---------------|--------|
| `error handling (async)` | 5 | No |
| `foo OR bar AND baz` | 5 | No |
| `"already quoted"` | 2 | No |
| `hybrid*` | 5 | No |
| `error NOT warning` | 5 | No |
| `NEAR(search query, 5)` | 3 | No |

**Advanced mode (`advanced_fts=True`):** FTS5 operators active.

| Query Type | Query | Results (adv) | Results (safe) | Fewer? |
|-----------|-------|--------------|---------------|--------|
| phrase | `"def hybrid_search"` | 9 | 10 | Yes |
| negation | `error NOT warning` | 10 | 10 | No |
| prefix | `hybrid*` | 10 | 10 | No |
| unquoted | `hybrid_search` | 10 | 10 | No |

**Phrase precision:** Quoted phrases return fewer, more precise results.

| Phrase | Phrase Results | Unquoted Results | Subset? |
|--------|---------------|-----------------|---------|
| `"def hybrid_search"` | 9 | 20 | No |
| `"keyword search"` | 20 | 20 | No |
| `"error handling"` | 20 | 20 | No |

**FTS5 latency (safe vs advanced):**

| Query | Safe (ms) | Advanced (ms) |
|-------|----------|-------------|
| `hybrid_search` | 0.70 | 0.65 |
| `"def hybrid_search"` | 1.00 | 0.97 |
| `error NOT warning` | 0.68 | 0.39 |
| `hybrid*` | 0.28 | 0.33 |

---

*Generated by `tests/test_search_benchmark.py` against the live codemem index (/Users/danieliser/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db).*
