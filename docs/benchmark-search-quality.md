# Search Quality Benchmark Report

## 1. BM25 Score Normalization

**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.
Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.

| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |
|-------|----------|------------------------|-------------------------|-----------------|
| `normalize_bm25_score` | 10 | [-6.88, -4.83] | [0.829, 0.873] | Yes |
| `ProjectDB` | 10 | [-6.21, -4.96] | [0.832, 0.861] | Yes |
| `hybrid_search` | 10 | [-5.47, -4.69] | [0.824, 0.845] | Yes |
| `error handling` | 10 | [-10.28, -9.05] | [0.901, 0.911] | Yes |
| `authentication scope` | 9 | [-9.61, -6.02] | [0.857, 0.906] | Yes |
| `graph traversal` | 10 | [-8.95, -6.37] | [0.864, 0.899] | Yes |
| `keyword_search limit` | 10 | [-10.33, -9.02] | [0.900, 0.912] | Yes |
| `create_scope` | 10 | [-6.49, -5.69] | [0.850, 0.867] | Yes |
| `async to_thread` | 10 | [-12.21, -8.51] | [0.895, 0.924] | Yes |
| `FTS5 BM25` | 10 | [-10.04, -7.79] | [0.886, 0.909] | Yes |

**Score distribution:** Higher std = better discrimination between relevant and marginal results.

| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |
|-------|-------|-----------|----------------------|---------|
| `normalize_bm25_score` | 0.8730 | 0.8651 | 0.0445 | 0.0137 |
| `ProjectDB` | 0.8614 | 0.8522 | 0.0292 | 0.0099 |
| `hybrid_search` | 0.8454 | 0.8403 | 0.0212 | 0.0073 |
| `error handling` | 0.9114 | 0.9078 | 0.0109 | 0.0042 |
| `authentication scope` | 0.9058 | 0.8951 | 0.0483 | 0.0181 |
| `graph traversal` | 0.8995 | 0.8859 | 0.0351 | 0.0117 |
| `keyword_search limit` | 0.9117 | 0.9060 | 0.0115 | 0.0037 |
| `create_scope` | 0.8666 | 0.8612 | 0.0161 | 0.0060 |
| `async to_thread` | 0.9243 | 0.9108 | 0.0295 | 0.0084 |
| `FTS5 BM25` | 0.9095 | 0.9052 | 0.0232 | 0.0077 |

## 2. embed_query (Retrieval Prefix)

**Change:** `embed_single(query)` → `embed_query(query)` adds retrieval prefix.

| Query | embed_single Top-3 | embed_query Top-3 | Top-5 Overlap | Score Delta |
|-------|-------------------|------------------|--------------|-------------|
| `normalize_bm25_score` | SPEC.md, 04-search-pipeline.md, test_search_benchmark.py | SPEC.md, 04-search-pipeline.md, test_search_benchmark.py | 5/5 | +0.0000 |
| `ProjectDB` | test_db_graph.py, conftest.py, test_real_federation.py | conftest.py, test_db_graph.py, test_real_federation.py | 5/5 | -0.0157 |
| `hybrid_search` | test_search_benchmark.py, spec-v1.md, spec-v2.md | test_search_benchmark.py, spec-v1.md, spec-v2.md | 5/5 | +0.0010 |
| `error handling` | test_server.py, test_embeddings.py, benchmark-search-quality.md | test_server.py, test_embeddings.py, benchmark-search-quality.md | 5/5 | +0.0000 |
| `authentication scope` | research.md, test_search_benchmark.py, benchmark-search-quality.md | research.md, test_search_benchmark.py, benchmark-search-quality.md | 5/5 | +0.0000 |
| `graph traversal` | research.md, intake.md, test_search_benchmark.py | intake.md, research.md, test_search_benchmark.py | 5/5 | +0.0157 |

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
| `normalize_bm25_score` | SPEC.md:2314 | 28 | 7 | 7/28 (25%) | 5 |
| `normalize_bm25_score` | 04-search-pipeline.md:963 | 28 | 7 | 7/28 (25%) | 5 |
| `ProjectDB` | conftest.py:13 | 14 | 6 | 6/14 (43%) | 2 |
| `ProjectDB` | test_db_graph.py:0 | 6 | 4 | 4/6 (67%) | 0 |
| `hybrid_search` | test_search_benchmark.py:94 | 3 | 3 | 3/3 (100%) | 1 |
| `hybrid_search` | research.md:400 | 2 | 2 | 2/2 (100%) | 0 |
| `error handling` | test_embeddings.py:167 | 51 | 5 | 5/51 (10%) | 1 |
| `error handling` | benchmark-search-quality.md:69 | 24 | 7 | 7/24 (29%) | 4 |
| `authentication scope` | research.md:646 | 6 | 6 | 6/6 (100%) | 2 |
| `authentication scope` | test_search_benchmark.py:40 | 16 | 7 | 7/16 (44%) | 7 |
| `graph traversal` | research.md:112 | 12 | 7 | 7/12 (58%) | 4 |
| `graph traversal` | intake.md:6 | 4 | 4 | 4/4 (100%) | 2 |
| `keyword_search limit` | spec-v1.md:819 | 33 | 7 | 7/33 (21%) | 6 |
| `keyword_search limit` | research.md:538 | 32 | 7 | 7/32 (22%) | 9 |
| `create_scope` | benchmark-search-quality.md:69 | 24 | 7 | 7/24 (29%) | 13 |
| `create_scope` | test_auth.py:53 | 74 | 7 | 7/74 (9%) | 3 |
| `async to_thread` | PLAN.md:64 | 4 | 4 | 4/4 (100%) | 2 |
| `async to_thread` | CLAUDE.md:60 | 7 | 5 | 5/7 (71%) | 5 |
| `FTS5 BM25` | search.py:32 | 14 | 5 | 5/14 (36%) | 1 |
| `FTS5 BM25` | SPEC.md:1493 | 26 | 7 | 7/26 (27%) | 4 |

**Sample snippets** (showing what agents actually see):

**Query: `hybrid_search`** — test_search_benchmark.py:94 (3 lines → 3 lines)
```
def _search(db, query, limit=10):
    """Run keyword-only hybrid search (enriched results with file_path)."""
    return hybrid_search(query, query_embedding=None, db=db, limit=limit)
```

**Query: `error handling`** — test_embeddings.py:167 (51 lines → 5 lines)
```
class TestEmbeddingErrorHandling:
    """Test error handling."""

    @patch("httpx.Client.post")
    def test_embed_request_error(self, mock_post):
```

**Query: `create_scope`** — benchmark-search-quality.md:69 (24 lines → 7 lines)
```
    def test_embed_request_error(self, mock_post):
```

**Query: `create_scope`** — test_auth.py:53 (74 lines → 7 lines)
```
class TestCreateScope:
    """Test session creation."""
```

## 4. Content-Addressable Document IDs

**Change:** `generate_docid` produces a stable 6-char hex hash from content.

| Property | Result |
|----------|--------|
| Deterministic (same content → same ID) | Pass |
| Total unique IDs tested | 74 |
| Collisions (different content, same ID) | 0 |
| ID format | 6-char hex (e.g. `95aa37`) |

## 5. Multi-Format Output

**Change:** `format_results` supports json, csv, markdown, and files output modes.
Before: only `json.dumps`. After: agents can request the format that suits their task.

| Format | Valid | Size | Notes |
|--------|-------|------|-------|
| json | Yes | 4863 chars | 5 items, full metadata |
| csv | Yes | 3042 chars | 5 rows, 19 columns |
| markdown | Yes | 1424 chars | 5 result sections with code blocks |
| files | Yes | 189 chars | 5 unique file paths |

**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.

## 6. Search Latency

**Keyword-only search** (FTS5 + RRF merge + enrichment):

| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |
|-------|-----------|---------------|-------------|-----------|---------|
| `normalize_bm25_score` | 0.68 | 0.46 | 0.02 | 1.15 | 10 |
| `ProjectDB` | 0.53 | 0.54 | 0.02 | 1.08 | 10 |
| `hybrid_search` | 0.97 | 0.22 | 0.01 | 1.20 | 10 |
| `error handling` | 0.56 | 0.28 | 0.01 | 0.85 | 10 |
| `authentication scope` | 0.63 | 0.38 | 0.01 | 1.02 | 9 |
| `graph traversal` | 0.71 | 0.38 | 0.02 | 1.11 | 10 |
| `keyword_search limit` | 1.13 | 0.55 | 0.02 | 1.70 | 10 |
| `create_scope` | 0.98 | 0.51 | 0.02 | 1.51 | 10 |
| `async to_thread` | 1.22 | 0.44 | 0.01 | 1.67 | 10 |
| `FTS5 BM25` | 0.57 | 0.35 | 0.01 | 0.93 | 10 |

**Full hybrid search** (embed + keyword + semantic + RRF + snippet + docid):

| Query | Embed (ms) | Search (ms) | Post-process (ms) | Total (ms) |
|-------|-----------|-----------|-------------------|-----------|
| `normalize_bm25_score` | 0.0 | 11.5 | 0.50 | 12.0 |
| `ProjectDB` | 0.0 | 10.2 | 0.55 | 10.7 |
| `hybrid_search` | 0.0 | 11.0 | 0.25 | 11.3 |
| `error handling` | 0.0 | 12.0 | 0.30 | 12.3 |
| `authentication scope` | 0.0 | 11.5 | 0.43 | 11.9 |

## 7. BM25 Strong-Signal Short-Circuit

**Change:** Skip semantic search + PPR when top BM25 result score >= 0.85 AND gap to #2 >= 0.15.
Saves ~30-50ms per query when keyword match is unambiguous.

| Query | Top-1 Norm | Top-2 Norm | Gap | Short-Circuit? | Reason |
|-------|-----------|-----------|-----|---------------|--------|
| `normalize_bm25_score` | 0.8730 | 0.8730 | 0.0000 | No | Gap 0.000 < 0.15 |
| `ProjectDB` | 0.8614 | 0.8559 | 0.0055 | No | Gap 0.005 < 0.15 |
| `hybrid_search` | 0.8454 | 0.8440 | 0.0014 | No | Score 0.845 < 0.85 |
| `error handling` | 0.9114 | 0.9103 | 0.0011 | No | Gap 0.001 < 0.15 |
| `authentication scope` | 0.9058 | 0.9057 | 0.0001 | No | Gap 0.000 < 0.15 |
| `graph traversal` | 0.8995 | 0.8899 | 0.0096 | No | Gap 0.010 < 0.15 |
| `keyword_search limit` | 0.9117 | 0.9081 | 0.0036 | No | Gap 0.004 < 0.15 |
| `create_scope` | 0.8666 | 0.8664 | 0.0001 | No | Gap 0.000 < 0.15 |
| `async to_thread` | 0.9243 | 0.9139 | 0.0104 | No | Gap 0.010 < 0.15 |
| `FTS5 BM25` | 0.9095 | 0.9074 | 0.0021 | No | Gap 0.002 < 0.15 |

**Latency comparison** (with vs without short-circuit):

| Query | Full Pipeline (ms) | Short-Circuit (ms) | Saved (ms) | Triggered? |
|-------|-------------------|-------------------|-----------|-----------|
| `normalize_bm25_score` | 11.4 | 1.0 | +10.4 | No |
| `ProjectDB` | 11.2 | 0.7 | +10.5 | No |
| `hybrid_search` | 11.2 | 1.3 | +10.0 | No |
| `error handling` | 13.1 | 0.8 | +12.3 | No |
| `authentication scope` | 12.0 | 0.8 | +11.2 | No |

## 8. Weighted RRF Fusion

**Change:** Per-list weights: keyword=1.5x, semantic=1.0x, graph=0.8x (was 1.0x equal).
Keyword results boosted because FTS5 precision is higher for code search.

| Query | Equal RRF Top-3 | Weighted RRF Top-3 | Rank Changes | Score Boost |
|-------|-----------------|--------------------|-------------|-------------|
| `normalize_bm25_score` | 12303157, 12303416, 5584855 | 12303157, 12303416, 5584855 | 0/3 | +0.007576 |
| `ProjectDB` | 12303185, 12303074, 12303266 | 12303185, 12303266, 12303391 | 2/3 | +0.008197 |
| `hybrid_search` | 12303420, 12301794, 12301958 | 12303420, 12301794, 12301958 | 0/3 | +0.008197 |
| `error handling` | 12303451, 12302558, 12303301 | 12303451, 12302558, 12303301 | 0/3 | +0.007692 |
| `authentication scope` | 12302251, 12302047, 12303416 | 12302251, 12303416, 12301543 | 2/3 | +0.008197 |
| `graph traversal` | 12302644, 12301643, 12301676 | 12302644, 12301643, 12303416 | 1/3 | +0.008065 |

**Weight sensitivity:** How per-list weights shift RRF scores.

| Weights (kw, sem) | ID-1 Score | ID-2 Score | ID-3 Score | Rank Change vs Equal |
|-------------------|-----------|-----------|-----------|---------------------|
| (1.0, 1.0) | 0.032266 | 0.032522 | 0.032002 | 0/3 |
| (1.5, 1.0) | 0.040463 | 0.040587 | 0.039939 | 0/3 |
| (2.0, 1.0) | 0.048660 | 0.048652 | 0.047875 | 2/3 |
| (1.0, 2.0) | 0.048139 | 0.048916 | 0.048131 | 0/3 |

---

*Generated by `tests/test_search_benchmark.py` against the live codemem index (/Users/danieliser/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db).*
