# Search Quality Benchmark Report

## 1. BM25 Score Normalization

**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.
Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.

| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |
|-------|----------|------------------------|-------------------------|-----------------|
| `normalize_bm25_score` | 10 | [-6.93, -4.45] | [0.816, 0.874] | Yes |
| `ProjectDB` | 10 | [-6.17, -4.93] | [0.831, 0.861] | Yes |
| `hybrid_search` | 10 | [-5.46, -4.68] | [0.824, 0.845] | Yes |
| `error handling` | 10 | [-10.34, -9.01] | [0.900, 0.912] | Yes |
| `authentication scope` | 9 | [-9.56, -6.00] | [0.857, 0.905] | Yes |
| `graph traversal` | 10 | [-8.89, -6.34] | [0.864, 0.899] | Yes |
| `keyword_search limit` | 10 | [-10.33, -9.02] | [0.900, 0.912] | Yes |
| `create_scope` | 10 | [-6.46, -5.30] | [0.841, 0.866] | Yes |
| `async to_thread` | 10 | [-12.15, -8.48] | [0.895, 0.924] | Yes |
| `FTS5 BM25` | 10 | [-10.15, -7.87] | [0.887, 0.910] | Yes |

**Score distribution:** Higher std = better discrimination between relevant and marginal results.

| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |
|-------|-------|-----------|----------------------|---------|
| `normalize_bm25_score` | 0.8738 | 0.8659 | 0.0575 | 0.0164 |
| `ProjectDB` | 0.8606 | 0.8515 | 0.0292 | 0.0099 |
| `hybrid_search` | 0.8452 | 0.8401 | 0.0212 | 0.0072 |
| `error handling` | 0.9118 | 0.9077 | 0.0118 | 0.0045 |
| `authentication scope` | 0.9053 | 0.8931 | 0.0482 | 0.0183 |
| `graph traversal` | 0.8989 | 0.8851 | 0.0351 | 0.0118 |
| `keyword_search limit` | 0.9117 | 0.9060 | 0.0115 | 0.0037 |
| `create_scope` | 0.8660 | 0.8581 | 0.0248 | 0.0063 |
| `async to_thread` | 0.9239 | 0.9104 | 0.0294 | 0.0083 |
| `FTS5 BM25` | 0.9103 | 0.9061 | 0.0231 | 0.0076 |

## 3. Snippet Extraction

**Change:** `extract_snippet` returns a focused ~7-line window around the best
keyword-matching line, instead of the full chunk (which can be 50+ lines).

| Query | File | Chunk Lines | Snippet Lines | Compression | Best Match Line |
|-------|------|------------|--------------|-------------|----------------|
| `normalize_bm25_score` | SPEC.md:2314 | 28 | 4 | 4/28 (14%) | 0 |
| `normalize_bm25_score` | 04-search-pipeline.md:963 | 28 | 4 | 4/28 (14%) | 0 |
| `ProjectDB` | conftest.py:13 | 14 | 6 | 6/14 (43%) | 2 |
| `ProjectDB` | test_db_graph.py:0 | 6 | 4 | 4/6 (67%) | 5 |
| `hybrid_search` | test_search_benchmark.py:94 | 3 | 3 | 3/3 (100%) | 0 |
| `hybrid_search` | research.md:400 | 2 | 2 | 2/2 (100%) | 0 |
| `error handling` | benchmark-search-quality.md:78 | 24 | 7 | 7/24 (29%) | 10 |
| `error handling` | test_embeddings.py:167 | 51 | 5 | 5/51 (10%) | 1 |
| `authentication scope` | research.md:646 | 6 | 4 | 4/6 (67%) | 0 |
| `authentication scope` | test_search_benchmark.py:40 | 16 | 4 | 4/16 (25%) | 0 |
| `graph traversal` | research.md:112 | 12 | 7 | 7/12 (58%) | 4 |
| `graph traversal` | intake.md:6 | 4 | 4 | 4/4 (100%) | 2 |
| `keyword_search limit` | spec-v1.md:819 | 33 | 4 | 4/33 (12%) | 0 |
| `keyword_search limit` | research.md:538 | 32 | 4 | 4/32 (12%) | 0 |
| `create_scope` | test_auth.py:53 | 74 | 4 | 4/74 (5%) | 0 |
| `create_scope` | test_e2e_tools.py:364 | 44 | 4 | 4/44 (9%) | 0 |
| `async to_thread` | PLAN.md:64 | 4 | 4 | 4/4 (100%) | 2 |
| `async to_thread` | CLAUDE.md:60 | 7 | 6 | 6/7 (86%) | 4 |
| `FTS5 BM25` | search.py:31 | 14 | 5 | 5/14 (36%) | 1 |
| `FTS5 BM25` | SPEC.md:1493 | 26 | 7 | 7/26 (27%) | 4 |

**Sample snippets** (showing what agents actually see):

**Query: `hybrid_search`** — test_search_benchmark.py:94 (3 lines → 3 lines)
```
def _search(db, query, limit=10):
    """Run keyword-only hybrid search (enriched results with file_path)."""
    return hybrid_search(query, query_embedding=None, db=db, limit=limit)
```

**Query: `error handling`** — benchmark-search-quality.md:78 (24 lines → 7 lines)
```
**Query: `error handling`** — :167 (51 lines → 5 lines)
```
class TestEmbeddingErrorHandling:
    """Test error handling."""

    @patch("httpx.Client.post")
    def test_embed_request_error(self, mock_post):
```

**Query: `create_scope`** — test_auth.py:53 (74 lines → 4 lines)
```
class TestCreateScope:
    """Test session creation."""

    def test_create_scope_project_level(self, db_conn):
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
| json | Yes | 4761 chars | 5 items, full metadata |
| csv | Yes | 2923 chars | 5 rows, 19 columns |
| markdown | Yes | 1237 chars | 6 result sections with code blocks |
| files | Yes | 189 chars | 5 unique file paths |

**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.

## 6. Search Latency

**Keyword-only search** (FTS5 + RRF merge + enrichment):

| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |
|-------|-----------|---------------|-------------|-----------|---------|
| `normalize_bm25_score` | 0.81 | 0.15 | 0.02 | 0.98 | 10 |
| `ProjectDB` | 0.66 | 0.18 | 0.02 | 0.87 | 10 |
| `hybrid_search` | 0.86 | 0.07 | 0.01 | 0.94 | 10 |
| `error handling` | 0.58 | 0.10 | 0.01 | 0.69 | 10 |
| `authentication scope` | 0.55 | 0.13 | 0.01 | 0.69 | 9 |
| `graph traversal` | 0.67 | 0.15 | 0.02 | 0.84 | 10 |
| `keyword_search limit` | 1.02 | 0.17 | 0.02 | 1.20 | 10 |
| `create_scope` | 0.75 | 0.15 | 0.01 | 0.92 | 10 |
| `async to_thread` | 1.03 | 0.15 | 0.01 | 1.19 | 10 |
| `FTS5 BM25` | 0.55 | 0.12 | 0.01 | 0.69 | 10 |

---

*Generated by `tests/test_search_benchmark.py` against the live codemem index (/Users/danieliser/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db).*
