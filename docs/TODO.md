# Tessera — TODO

## Search Quality

### Embedding-based snippet scoring

**Context:** `extract_snippet` currently scores lines by keyword overlap (set intersection of lowercased tokens). This works well for identifier-heavy queries (`normalize_bm25_score`, `class ProjectDB`) but fails on semantic queries like "error handling logic" where relevant lines use `try/except/raise` without matching any query words.

**Proposal:** Score candidate lines (or small line windows) using cosine similarity against the query embedding instead of keyword overlap. The embedding client is already available at search time — `embed_query` produces the query vector. For each chunk, embed a sliding window of ~3 lines and pick the window with highest similarity to the query embedding.

**Trade-offs:**
- Accuracy: Dramatically better for semantic/conceptual queries. Keyword overlap is essentially BM25-at-line-level — embedding similarity captures meaning.
- Cost: Each chunk would need N embedding calls for N sliding windows, or a single batch call with N inputs. For a 30-line chunk with 3-line windows, that's ~28 embeddings per chunk, times K results. At 10 results, ~280 embedding calls per search. With local models (nomic-embed on llama.cpp) latency is ~1-5ms per embedding, so ~0.3-1.4s added latency.
- Optimization: Could use a two-pass approach — keyword overlap first as a fast filter, then embedding similarity only on the top 3-5 chunks. Or pre-compute line-level embeddings at index time (storage cost but zero search-time overhead).
- Fallback: Keep keyword overlap as the fallback when embedding endpoint is unavailable.

**Priority:** Medium — improves agent experience on conceptual queries but current keyword approach handles the common case (searching for identifiers) well.

---

### ~~Symbol-aware snippet expansion (`expand_context`)~~ — SHIPPED in v0.6.0

Implemented as collapsed ancestry snippets with `expand_context` ("lines" / "full") and `max_depth` params.
