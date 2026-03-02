# Search Quality Benchmark Report

**Index:** codemem project — 3,180 chunks, 1,821 symbols, 3,131 FTS entries, 0 embeddings stored.

> **Note:** Embedding tests (Section 2) are skipped because the codemem index has no stored
> embeddings — the project was indexed without the embedding endpoint running. Re-index with
> `--embedding-endpoint` to populate `chunk_embeddings` and enable hybrid (keyword + semantic) search.
> Keyword-only search works fully without embeddings.

> **Bug found:** `hybrid_search` enrichment doesn't join the `files` table, so `file_path` is
> empty in results. The benchmark works around this with a manual join. Should fix in `search.py`.

## 1. BM25 Score Normalization

**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.
Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.

| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |
|-------|----------|------------------------|-------------------------|-----------------|
| `normalize_bm25_score` | 10 | [-6.92, -4.44] | [0.816, 0.874] | Yes |
| `ProjectDB` | 10 | [-6.16, -4.92] | [0.831, 0.860] | Yes |
| `hybrid_search` | 10 | [-5.39, -4.67] | [0.824, 0.844] | Yes |
| `error handling` | 10 | [-10.48, -8.99] | [0.900, 0.913] | Yes |
| `authentication scope` | 8 | [-9.60, -6.02] | [0.858, 0.906] | Yes |
| `graph traversal` | 10 | [-8.84, -6.30] | [0.863, 0.898] | Yes |
| `keyword_search limit` | 10 | [-10.28, -8.98] | [0.900, 0.911] | Yes |
| `create_scope` | 10 | [-6.45, -5.66] | [0.850, 0.866] | Yes |
| `async to_thread` | 10 | [-12.12, -8.47] | [0.894, 0.924] | Yes |
| `FTS5 BM25` | 10 | [-10.13, -7.85] | [0.887, 0.910] | Yes |

**Score distribution:** Higher std = better discrimination between relevant and marginal results.

| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |
|-------|-------|-----------|----------------------|---------|
| `normalize_bm25_score` | 0.8737 | 0.8661 | 0.0574 | 0.0163 |
| `ProjectDB` | 0.8604 | 0.8512 | 0.0292 | 0.0099 |
| `hybrid_search` | 0.8436 | 0.8380 | 0.0199 | 0.0063 |
| `error handling` | 0.9129 | 0.9078 | 0.0130 | 0.0047 |
| `authentication scope` | 0.9057 | 0.8937 | 0.0481 | 0.0188 |
| `graph traversal` | 0.8984 | 0.8860 | 0.0353 | 0.0115 |
| `keyword_search limit` | 0.9114 | 0.9056 | 0.0115 | 0.0037 |
| `create_scope` | 0.8659 | 0.8585 | 0.0160 | 0.0049 |
| `async to_thread` | 0.9238 | 0.9103 | 0.0294 | 0.0083 |
| `FTS5 BM25` | 0.9101 | 0.9059 | 0.0231 | 0.0076 |

## 3. Snippet Extraction

**Change:** `extract_snippet` returns a focused ~7-line window around the best
keyword-matching line, instead of the full chunk (which can be 50+ lines).

| Query | File | Chunk Lines | Snippet Lines | Compression | Best Match Line |
|-------|------|------------|--------------|-------------|----------------|
| `normalize_bm25_score` | SPEC.md:2314 | 28 | 4 | 4/28 (14%) | 0 |
| `normalize_bm25_score` | 04-search-pipeline.md:963 | 28 | 4 | 4/28 (14%) | 0 |
| `ProjectDB` | conftest.py:13 | 14 | 6 | 6/14 (43%) | 2 |
| `ProjectDB` | test_db_graph.py:0 | 6 | 4 | 4/6 (67%) | 5 |
| `hybrid_search` | research.md:400 | 2 | 2 | 2/2 (100%) | 0 |
| `hybrid_search` | spec-v1.md:1394 | 1 | 1 | 1/1 (100%) | 0 |
| `error handling` | benchmark-search-quality.md:70 | 20 | 7 | 7/20 (35%) | 6 |
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

**Query: `hybrid_search`** — research.md:400 (2 lines → 2 lines)
```
- [Advanced RAG — Understanding Reciprocal Rank Fusion in Hybrid Search](https://glaforge.dev/posts/2026/02/10/advanced-rag-understanding-reciprocal-rank-fusion-in-hybrid-search/) — RRF + reranking pipeline
- [Azure AI Search: Hybrid retrieval and reranking](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/azure-ai-search-outperforming-vector-search-with-hybrid-retrieval-and-reranking/3929167) — Enterprise hybrid search architecture
```

**Query: `error handling`** — benchmark-search-quality.md:70 (20 lines → 7 lines)
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
| Total unique IDs tested | 72 |
| Collisions (different content, same ID) | 0 |
| ID format | 6-char hex (e.g. `95aa37`) |

## 5. Multi-Format Output

**Change:** `format_results` supports json, csv, markdown, and files output modes.
Before: only `json.dumps`. After: agents can request the format that suits their task.

| Format | Valid | Size | Notes |
|--------|-------|------|-------|
| json | Yes | 5417 chars | 5 items, full metadata |
| csv | Yes | 3554 chars | 5 rows, 19 columns |
| markdown | Yes | 1405 chars | 6 result sections with code blocks |
| files | Yes | 189 chars | 5 unique file paths |

**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.

## 6. Search Latency

**Keyword-only search** (FTS5 + RRF merge + enrichment):

| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |
|-------|-----------|---------------|-------------|-----------|---------|
| `normalize_bm25_score` | 0.82 | 0.15 | 0.02 | 0.98 | 10 |
| `ProjectDB` | 0.64 | 0.18 | 0.02 | 0.85 | 10 |
| `hybrid_search` | 1.16 | 0.08 | 0.01 | 1.25 | 10 |
| `error handling` | 0.71 | 0.10 | 0.01 | 0.81 | 10 |
| `authentication scope` | 0.78 | 0.11 | 0.01 | 0.90 | 8 |
| `graph traversal` | 0.93 | 0.15 | 0.02 | 1.10 | 10 |
| `keyword_search limit` | 1.30 | 0.17 | 0.01 | 1.48 | 10 |
| `create_scope` | 1.78 | 0.16 | 0.02 | 1.96 | 10 |
| `async to_thread` | 1.56 | 0.15 | 0.01 | 1.73 | 10 |
| `FTS5 BM25` | 0.64 | 0.12 | 0.01 | 0.78 | 10 |

---

*Generated by `tests/test_search_benchmark.py` against the live codemem index (/Users/danieliser/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db).*
