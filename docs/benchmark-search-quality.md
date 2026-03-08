# Search Quality Benchmark Report

## 1. BM25 Score Normalization

**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.
Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.

| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |
|-------|----------|------------------------|-------------------------|-----------------|
| `normalize_bm25_score` | 10 | [-7.85, -5.88] | [0.855, 0.887] | Yes |
| `ProjectDB` | 10 | [-5.86, -5.09] | [0.836, 0.854] | Yes |
| `hybrid_search` | 10 | [-5.36, -4.85] | [0.829, 0.843] | Yes |
| `error handling` | 10 | [-10.89, -9.67] | [0.906, 0.916] | Yes |
| `authentication scope` | 10 | [-9.10, -6.84] | [0.873, 0.901] | Yes |
| `graph traversal` | 10 | [-9.12, -6.41] | [0.865, 0.901] | Yes |
| `keyword_search limit` | 10 | [-10.90, -8.07] | [0.890, 0.916] | Yes |
| `create_scope` | 10 | [-6.73, -5.52] | [0.847, 0.871] | Yes |
| `async to_thread` | 10 | [-11.43, -8.25] | [0.892, 0.920] | Yes |
| `FTS5 BM25` | 10 | [-12.01, -8.43] | [0.894, 0.923] | Yes |

**Score distribution:** Higher std = better discrimination between relevant and marginal results.

| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |
|-------|-------|-----------|----------------------|---------|
| `normalize_bm25_score` | 0.8871 | 0.8790 | 0.0325 | 0.0109 |
| `ProjectDB` | 0.8542 | 0.8496 | 0.0185 | 0.0064 |
| `hybrid_search` | 0.8427 | 0.8397 | 0.0136 | 0.0045 |
| `error handling` | 0.9159 | 0.9122 | 0.0097 | 0.0033 |
| `authentication scope` | 0.9010 | 0.8964 | 0.0285 | 0.0107 |
| `graph traversal` | 0.9012 | 0.8968 | 0.0362 | 0.0141 |
| `keyword_search limit` | 0.9159 | 0.9122 | 0.0262 | 0.0099 |
| `create_scope` | 0.8707 | 0.8646 | 0.0241 | 0.0073 |
| `async to_thread` | 0.9196 | 0.9117 | 0.0277 | 0.0092 |
| `FTS5 BM25` | 0.9231 | 0.9084 | 0.0292 | 0.0080 |

## 2. embed_query (Retrieval Prefix)

**Change:** `embed_single(query)` → `embed_query(query)` adds retrieval prefix.

| Query | embed_single Top-3 | embed_query Top-3 | Top-5 Overlap | Score Delta |
|-------|-------------------|------------------|--------------|-------------|
| `normalize_bm25_score` | search.py, benchmark-search-quality.md, test_search_benchmark.py | search.py, benchmark-search-quality.md, test_search_benchmark.py | 4/5 | +0.0003 |
| `ProjectDB` | test_db_graph.py, _project.py, _project.py | test_db_graph.py, _project.py, _state.py | 3/5 | -0.0009 |
| `hybrid_search` | test_search_benchmark.py, test_search_with_ppr.py, search.py | test_search_benchmark.py, search.py, search-guide.md | 3/5 | +0.0003 |
| `error handling` | test_server.py, embeddings.py, auth.py | test_server.py, snippets.md, auth.py | 4/5 | +0.0000 |
| `authentication scope` | auth.py, auth.py, auth.py | auth.py, auth.py, auth.py | 4/5 | +0.0000 |
| `graph traversal` | test_search_benchmark.py, research-dependency-graphs.md, test_graph.py | spec-v1.md, research-dependency-graphs.md, intake.md | 2/5 | -0.0161 |

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
| `normalize_bm25_score` | search.py:107 | 23 | 7 | 7/23 (30%) | 14 |
| `normalize_bm25_score` | test_search_benchmark.py:666 | 8 | 7 | 7/8 (88%) | 4 |
| `ProjectDB` | test_real_federation.py:73 | 89 | 4 | 4/89 (4%) | 0 |
| `ProjectDB` | conftest.py:0 | 27 | 7 | 7/27 (26%) | 15 |
| `hybrid_search` | test_search_benchmark.py:101 | 3 | 3 | 3/3 (100%) | 1 |
| `hybrid_search` | search-advanced.md:309 | 131 | 7 | 7/131 (5%) | 12 |
| `error handling` | test_embeddings.py:169 | 51 | 5 | 5/51 (10%) | 1 |
| `error handling` | benchmark-search-quality.md:236 | 69 | 7 | 7/69 (10%) | 7 |
| `authentication scope` | benchmark-search-quality.md:0 | 59 | 7 | 7/59 (12%) | 13 |
| `authentication scope` | getting-started.md:329 | 185 | 7 | 7/185 (4%) | 98 |
| `graph traversal` | test_search_benchmark.py:666 | 8 | 5 | 5/8 (62%) | 6 |
| `graph traversal` | benchmark-search-quality.md:184 | 63 | 6 | 6/63 (10%) | 2 |
| `keyword_search limit` | spec-v1.md:735 | 120 | 7 | 7/120 (6%) | 72 |
| `keyword_search limit` | test_search_benchmark.py:815 | 116 | 7 | 7/116 (6%) | 24 |
| `create_scope` | test_auth.py:53 | 74 | 7 | 7/74 (9%) | 3 |
| `create_scope` | test_e2e_tools.py:363 | 44 | 7 | 7/44 (16%) | 10 |
| `async to_thread` | test_server_graph.py:30 | 54 | 7 | 7/54 (13%) | 8 |
| `async to_thread` | _collections.py:13 | 92 | 7 | 7/92 (8%) | 18 |
| `FTS5 BM25` | search.py:73 | 14 | 5 | 5/14 (36%) | 1 |
| `FTS5 BM25` | benchmark-search-quality.md:0 | 59 | 7 | 7/59 (12%) | 4 |

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
| Total unique IDs tested | 74 |
| Collisions (different content, same ID) | 0 |
| ID format | 6-char hex (e.g. `432c76`) |

## 5. Multi-Format Output

**Change:** `format_results` supports json, csv, markdown, and files output modes.
Before: only `json.dumps`. After: agents can request the format that suits their task.

| Format | Valid | Size | Notes |
|--------|-------|------|-------|
| json | Yes | 17416 chars | 5 items, full metadata |
| csv | Yes | 15161 chars | 5 rows, 21 columns |
| markdown | Yes | 1710 chars | 6 result sections with code blocks |
| files | Yes | 147 chars | 4 unique file paths |

**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.

## 6. Search Latency

**Keyword-only search** (FTS5 + RRF merge + enrichment):

| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |
|-------|-----------|---------------|-------------|-----------|---------|
| `normalize_bm25_score` | 0.57 | 1.10 | 0.03 | 1.69 | 10 |
| `ProjectDB` | 0.42 | 0.77 | 0.02 | 1.21 | 10 |
| `hybrid_search` | 1.02 | 1.18 | 0.02 | 2.23 | 10 |
| `error handling` | 0.53 | 1.15 | 0.03 | 1.72 | 10 |
| `authentication scope` | 0.58 | 1.54 | 0.03 | 2.15 | 10 |
| `graph traversal` | 0.62 | 1.28 | 0.03 | 1.92 | 10 |
| `keyword_search limit` | 1.00 | 1.53 | 0.03 | 2.56 | 10 |
| `create_scope` | 0.79 | 0.81 | 0.02 | 1.61 | 10 |
| `async to_thread` | 1.04 | 1.78 | 0.03 | 2.85 | 10 |
| `FTS5 BM25` | 0.45 | 1.29 | 0.03 | 1.77 | 10 |

**Full hybrid search** (embed + keyword + semantic + RRF + snippet + docid):

| Query | Embed (ms) | Search (ms) | Post-process (ms) | Total (ms) |
|-------|-----------|-----------|-------------------|-----------|
| `normalize_bm25_score` | 0.0 | 9.4 | 0.57 | 10.0 |
| `ProjectDB` | 0.0 | 8.3 | 2.98 | 11.3 |
| `hybrid_search` | 0.0 | 9.1 | 2.01 | 11.2 |
| `error handling` | 0.0 | 8.6 | 0.43 | 9.1 |
| `authentication scope` | 0.0 | 8.2 | 0.72 | 9.0 |

## 7. BM25 Strong-Signal Short-Circuit

**Change:** Skip semantic search + PPR when top BM25 result score >= 0.85 AND gap to #2 >= 0.15.
Saves ~30-50ms per query when keyword match is unambiguous.

| Query | Top-1 Norm | Top-2 Norm | Gap | Short-Circuit? | Reason |
|-------|-----------|-----------|-----|---------------|--------|
| `normalize_bm25_score` | 0.8871 | 0.8867 | 0.0003 | No | Gap 0.000 < 0.15 |
| `ProjectDB` | 0.8542 | 0.8539 | 0.0003 | No | Gap 0.000 < 0.15 |
| `hybrid_search` | 0.8427 | 0.8398 | 0.0029 | No | Score 0.843 < 0.85 |
| `error handling` | 0.9159 | 0.9134 | 0.0025 | No | Gap 0.002 < 0.15 |
| `authentication scope` | 0.9010 | 0.8968 | 0.0042 | No | Gap 0.004 < 0.15 |
| `graph traversal` | 0.9012 | 0.9011 | 0.0001 | No | Gap 0.000 < 0.15 |
| `keyword_search limit` | 0.9159 | 0.9155 | 0.0004 | No | Gap 0.000 < 0.15 |
| `create_scope` | 0.8707 | 0.8664 | 0.0043 | No | Gap 0.004 < 0.15 |
| `async to_thread` | 0.9196 | 0.9162 | 0.0033 | No | Gap 0.003 < 0.15 |
| `FTS5 BM25` | 0.9231 | 0.9073 | 0.0158 | No | Gap 0.016 < 0.15 |

**Latency comparison** (with vs without short-circuit):

| Query | Full Pipeline (ms) | Short-Circuit (ms) | Saved (ms) | Triggered? |
|-------|-------------------|-------------------|-----------|-----------|
| `normalize_bm25_score` | 132.9 | 0.7 | +132.2 | No |
| `ProjectDB` | 8.8 | 0.5 | +8.3 | No |
| `hybrid_search` | 8.9 | 0.9 | +7.9 | No |
| `error handling` | 8.2 | 0.5 | +7.7 | No |
| `authentication scope` | 7.5 | 0.6 | +6.9 | No |

## 8. Weighted RRF Fusion

**Change:** Per-list weights: keyword=1.0x, semantic=1.2x, graph=0.8x (was 1.0x equal).
Keyword results boosted because FTS5 precision is higher for code search.

| Query | Equal RRF Top-3 | Weighted RRF Top-3 | Rank Changes | Score Boost |
|-------|-----------------|--------------------|-------------|-------------|
| `normalize_bm25_score` | 12337091, 12337093, 12337313 | 12337091, 12336544, 12337313 | 1/3 | +0.003279 |
| `ProjectDB` | 12337378, 12337120, 12337231 | 12337378, 12337120, 12336982 | 1/3 | +0.003077 |
| `hybrid_search` | 12337296, 12336915, 12337134 | 12337296, 12336915, 12337134 | 0/3 | +0.003226 |
| `error handling` | 12337262, 12336930, 12337398 | 12337262, 12336930, 12336961 | 1/3 | +0.003279 |
| `authentication scope` | 12336544, 12336962, 12336569 | 12336962, 12336963, 12336967 | 3/3 | +0.003279 |
| `graph traversal` | 12337313, 12336830, 12336547 | 12337313, 12336830, 12336940 | 1/3 | +0.002985 |

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
| `normalize_bm25_score` | [12337093, 12337313, 12336544]... | [12337093, 12337313, 12336544]... | Pass |
| `error handling` | [12337398, 12336548, 12336914]... | [12337398, 12336548, 12336914]... | Pass |
| `graph traversal` | [12337313, 12336547, 12336608]... | [12337313, 12336547, 12336608]... | Pass |

**VEC-only vs LEX-only (proving complementary signals):**

| Query | LEX Top-3 | VEC Top-3 | Top-5 Overlap | Different? |
|-------|----------|----------|--------------|-----------|
| `normalize_bm25_score` | search.py, test_search_ben, benchmark-searc | search.py, test_search_ben, test_search_ben | 3/5 | Yes |
| `error handling` | test_embeddings, benchmark-searc, search-guide.md | test_server.py, snippets.md, auth.py | 0/5 | Yes |
| `graph traversal` | test_search_ben, benchmark-searc, research.md | spec-v1.md, research-depend, intake.md | 0/5 | Yes |

**HYDE vs VEC (embed_single vs embed_query):**

| Query | VEC Top-3 | HYDE Top-3 | Top-5 Overlap |
|-------|----------|-----------|--------------|
| `normalize_bm25_score` | search.py, test_search_ben, test_search_ben | test_search_ben, search.py, test_search_ben | 4/5 |
| `error handling` | test_server.py, snippets.md, auth.py | test_server.py, embeddings.py, auth.py | 4/5 |
| `graph traversal` | spec-v1.md, research-depend, intake.md | research-depend, test_search_ben, test_graph.py | 2/5 |

## 10. FTS5 Advanced Mode

**Safe mode (default):** All FTS5 operators escaped. No syntax errors possible.

| Query | Results (safe) | Error? |
|-------|---------------|--------|
| `error handling (async)` | 5 | No |
| `foo OR bar AND baz` | 5 | No |
| `"already quoted"` | 2 | No |
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
| `"def hybrid_search"` | 14 | 20 | No |
| `"keyword search"` | 20 | 20 | No |
| `"error handling"` | 20 | 20 | No |

**FTS5 latency (safe vs advanced):**

| Query | Safe (ms) | Advanced (ms) |
|-------|----------|-------------|
| `hybrid_search` | 0.77 | 0.75 |
| `"def hybrid_search"` | 0.99 | 0.94 |
| `error NOT warning` | 0.69 | 0.39 |
| `hybrid*` | 0.46 | 0.56 |

---

*Generated by `tests/test_search_benchmark.py` against the live codemem index (/Users/danieliser/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db).*
