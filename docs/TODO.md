# Tessera — TODO

## Search Quality

### ~~Embedding-based snippet scoring~~ — SHIPPED in v0.6.0

Implemented as semantic sliding-window scoring in `_find_best_match_line` / `_semantic_best_line`. 3-line windows are batch-embedded and scored by cosine similarity against the query embedding. Falls back to keyword overlap when embeddings are unavailable.

---

### ~~Symbol-aware snippet expansion (`expand_context`)~~ — SHIPPED in v0.6.0

Implemented as collapsed ancestry snippets with `expand_context` ("lines" / "full") and `max_depth` params.
