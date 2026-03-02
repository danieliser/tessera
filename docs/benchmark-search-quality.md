# Search Quality Benchmark Report

## 1. BM25 Score Normalization

**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.
Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.

| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |
|-------|----------|------------------------|-------------------------|-----------------|
| `normalize_bm25_score` | 10 | [-6.87, -4.83] | [0.829, 0.873] | Yes |
| `ProjectDB` | 10 | [-6.18, -4.93] | [0.831, 0.861] | Yes |
| `hybrid_search` | 10 | [-5.45, -4.69] | [0.824, 0.845] | Yes |
| `error handling` | 10 | [-10.24, -9.01] | [0.900, 0.911] | Yes |
| `authentication scope` | 9 | [-9.56, -6.00] | [0.857, 0.905] | Yes |
| `graph traversal` | 10 | [-8.88, -6.33] | [0.864, 0.899] | Yes |
| `keyword_search limit` | 10 | [-10.30, -9.00] | [0.900, 0.911] | Yes |
| `create_scope` | 10 | [-6.47, -5.67] | [0.850, 0.866] | Yes |
| `async to_thread` | 10 | [-12.10, -8.52] | [0.895, 0.924] | Yes |
| `FTS5 BM25` | 10 | [-10.13, -7.86] | [0.887, 0.910] | Yes |

**Score distribution:** Higher std = better discrimination between relevant and marginal results.

| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |
|-------|-------|-----------|----------------------|---------|
| `normalize_bm25_score` | 0.8726 | 0.8647 | 0.0441 | 0.0136 |
| `ProjectDB` | 0.8609 | 0.8519 | 0.0290 | 0.0099 |
| `hybrid_search` | 0.8450 | 0.8398 | 0.0208 | 0.0069 |
| `error handling` | 0.9110 | 0.9075 | 0.0109 | 0.0042 |
| `authentication scope` | 0.9053 | 0.8945 | 0.0482 | 0.0180 |
| `graph traversal` | 0.8987 | 0.8849 | 0.0352 | 0.0118 |
| `keyword_search limit` | 0.9113 | 0.9056 | 0.0115 | 0.0037 |
| `create_scope` | 0.8661 | 0.8593 | 0.0160 | 0.0050 |
| `async to_thread` | 0.9237 | 0.9101 | 0.0287 | 0.0077 |
| `FTS5 BM25` | 0.9102 | 0.9059 | 0.0231 | 0.0076 |

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
| `error handling` | benchmark-search-quality.md:69 | 29 | 7 | 7/29 (24%) | 7 |
| `authentication scope` | research.md:646 | 6 | 6 | 6/6 (100%) | 2 |
| `authentication scope` | test_search_benchmark.py:40 | 16 | 7 | 7/16 (44%) | 7 |
| `graph traversal` | research.md:112 | 12 | 7 | 7/12 (58%) | 4 |
| `graph traversal` | intake.md:6 | 4 | 4 | 4/4 (100%) | 2 |
| `keyword_search limit` | research.md:538 | 32 | 7 | 7/32 (22%) | 9 |
| `create_scope` | test_auth.py:53 | 74 | 7 | 7/74 (9%) | 3 |
| `create_scope` | test_e2e_tools.py:364 | 44 | 7 | 7/44 (16%) | 10 |
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

**Query: `create_scope`** — test_auth.py:53 (74 lines → 7 lines)
```
class TestCreateScope:
    """Test session creation."""

    def test_create_scope_project_level(self, db_conn):
        """Create a project-level scope."""
        session_id = create_scope(
            db_conn,
```

## 4. Content-Addressable Document IDs

**Change:** `generate_docid` produces a stable 6-char hex hash from content.

| Property | Result |
|----------|--------|
| Deterministic (same content → same ID) | Pass |
| Total unique IDs tested | 73 |
| Collisions (different content, same ID) | 0 |
| ID format | 6-char hex (e.g. `95aa37`) |

## 5. Multi-Format Output

**Change:** `format_results` supports json, csv, markdown, and files output modes.
Before: only `json.dumps`. After: agents can request the format that suits their task.

| Format | Valid | Size | Notes |
|--------|-------|------|-------|
| json | Yes | 4952 chars | 5 items, full metadata |
| csv | Yes | 3111 chars | 5 rows, 19 columns |
| markdown | Yes | 1423 chars | 5 result sections with code blocks |
| files | Yes | 189 chars | 5 unique file paths |

**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.

## 6. Search Latency

**Keyword-only search** (FTS5 + RRF merge + enrichment):

| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |
|-------|-----------|---------------|-------------|-----------|---------|
| `normalize_bm25_score` | 1.22 | 0.47 | 0.02 | 1.71 | 10 |
| `ProjectDB` | 0.73 | 0.54 | 0.02 | 1.30 | 10 |
| `hybrid_search` | 1.86 | 0.26 | 0.01 | 2.14 | 10 |
| `error handling` | 0.86 | 0.29 | 0.01 | 1.16 | 10 |
| `authentication scope` | 1.12 | 0.38 | 0.01 | 1.52 | 9 |
| `graph traversal` | 1.27 | 0.39 | 0.02 | 1.67 | 10 |
| `keyword_search limit` | 2.03 | 0.54 | 0.02 | 2.60 | 10 |
| `create_scope` | 1.69 | 0.51 | 0.02 | 2.22 | 10 |
| `async to_thread` | 2.30 | 0.53 | 0.02 | 2.84 | 10 |
| `FTS5 BM25` | 0.74 | 0.35 | 0.01 | 1.11 | 10 |

---

*Generated by `tests/test_search_benchmark.py` against the live codemem index (/Users/danieliser/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db).*
