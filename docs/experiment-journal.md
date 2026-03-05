# Experiment Journal

Systematic log of every retrieval optimization tested against the PM benchmark. Each entry records the hypothesis, model context, result, and whether the finding is model-dependent.

**Baseline**: All results compared against the unmodified pipeline for each model tier. "Before" numbers reflect parser-fixed codebase (v0.7.x) unless noted.

---

## Model Context Windows

Critical context: all BERT-family models share a 512-token (~2K char) input limit regardless of embedding dimension. Optimizations that consume input tokens (prefixes, metadata) have identical cost across BGE-small, BGE-base, and GTE-base.

| Model | Dimensions | Context Window | Architecture |
|-------|-----------|---------------|--------------|
| BGE-small-en-v1.5 | 384d | 512 tokens | BERT |
| BGE-base-en-v1.5 | 768d | 512 tokens | BERT |
| GTE-base | 768d | 512 tokens | BERT |
| Nomic-embed-text-v1.5 | 768d | 8192 tokens | RoPE/extended |
| Jina-embeddings-v2-small | 512d | 8192 tokens | ALiBi |
| Jina-embeddings-v2-base-code | 768d | 8192 tokens | ALiBi |
| BGE-M3 | 1024d | 8192 tokens | Extended |

**Implication**: Any optimization that prepends tokens to chunk content will behave identically on all 512-token models. Long-context models (Nomic, Jina-v2, BGE-M3) need separate testing — the prefix cost is negligible there.

---

## Experiments

### EXP-001: Structured metadata prefix (REJECTED)

**Date**: 2026-02-28
**Hypothesis**: Prepending `// File: X, Class: Y, Package: Z` to chunks before embedding improves retrieval by giving the model file/scope context.
**Models tested**: BGE-small (512 tokens)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| VEC+rerank MRR | 0.609 | 0.536 | **-0.073 (-12%)** |
| VEC+rerank (with reranker) | 0.739 | 0.735 | -0.004 |

**Finding**: Net negative. Comment-style structured prefix (`// File:`) consumes tokens without semantic value. The embedding model treats it as noise. The rigid format doesn't create useful semantic associations.

**Model-dependent?** Likely yes — 512-token models lose proportionally more content. Untested on long-context models.

---

### EXP-002: Oversized chunk splitting (NEUTRAL — infrastructure kept)

**Date**: 2026-03-03
**Hypothesis**: Splitting chunks >N characters into method-level or sliding-window sub-chunks improves retrieval by fitting more content into the embedding window.

**Models tested**: BGE-small (512 tokens)
**Thresholds tested**: 4K, 6K, 16K, 50K (no splitting)

| Threshold | Chunk Count | VEC+rerank MRR | HYB+rerank MRR |
|-----------|-------------|----------------|-----------------|
| 4K | 9,669 | 0.570 | 0.620 |
| 6K | 8,549 | 0.595 | 0.645 |
| 16K | 5,799 | 0.634 | 0.678 |
| 50K (no split) | 3,932 | 0.765 | 0.785 |
| 999K (no split) | 3,932 | 0.766 | — |

**Finding**: More chunks = more FAISS noise = worse retrieval. The MRR degradation scales linearly with chunk count. Isolation test (EXP-002c) confirmed the parser fix was the sole source of MRR gain — splitting contributed nothing.

**Root cause**: FAISS ANN search returns K nearest neighbors. More chunks means more false-positive candidates competing with the true match. The reranker helps but can't fully compensate.

**Model-dependent?** The noise effect is model-independent (it's a retrieval-stage problem), but long-context models might benefit from splitting differently since they can embed more content per chunk.

**Status**: Splitting infrastructure kept at 50K default (effectively disabled). Available for future use with enrichment strategies that reduce per-chunk noise.

---

### EXP-003: PHP parser fix (CONFIRMED — shipped v0.7.x)

**Date**: 2026-03-03
**Hypothesis**: Fixing `tree_sitter_php.language()` → `language_php()` and adding `method_declaration` to definition types would improve PHP chunking quality.

**Models tested**: BGE-small (512 tokens)

| Metric | Before (broken PHP) | After (fixed PHP) | Delta |
|--------|--------------------|--------------------|-------|
| VEC+rerank MRR | 0.739 | 0.766 | **+0.027 (+3.6%)** |
| Chunk count (Core) | ~1,147 | ~3,112 | +171% |

**Finding**: PHP was silently falling back to single-chunk-per-file. Fix enabled proper method-level chunking. Isolation test confirmed this is the sole source of MRR improvement in the v0.7.x cycle.

**Model-dependent?** No — parser fix affects chunking, not embedding. All models benefit equally.

---

### EXP-004: Natural language scope-context prefix

**Date**: 2026-03-04
**Hypothesis**: Prepending natural language scope context (e.g., `"Cookies class, set_cookie method in Cookies.php: "`) to chunk text before embedding creates semantic bridges between queries and code, unlike the rejected structured prefix (EXP-001).

Based on cAST (CMU 2025, +1.8-5.6 Recall@5) and Anthropic contextual retrieval (35-49% failure reduction).

**Models tested**: BGE-small (512 tokens)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| VEC+rerank MRR | 0.766 | 0.736 | **-0.030 (-3.9%)** |
| HYB+rerank MRR | 0.785 | 0.800 | **+0.015 (+1.9%)** |
| FULL MRR | — | 0.800 | — |

**Per-query changes (HYB+rerank)**:
- Q17 (DI container): MISS → rank 1 (major win)
- Q5 (exit intent): held at rank 1
- Q7 (popup conditionals): rank 1 → MISS (regression)
- Q14 (analytics): rank 1 → rank 2 (minor regression)

**Finding**: Mixed. The natural language prefix helps hybrid search (keyword component benefits from the added terms) but hurts pure vector search (prefix tokens consume BGE-small's 512-token budget, diluting actual code content). Net positive for the production path (HYB+rerank) but modest.

**Model-dependent?** Almost certainly. The prefix consumes ~10-20 tokens out of 512 (2-4% of budget). On 8192-token models this would be <0.25% — negligible. **Must test on Nomic/Jina-v2 before drawing conclusions.**

**BGE-base results** (same 512-token window, 768d vectors):

| Metric | Before (no prefix) | After (scope prefix) | Delta |
|--------|--------------------|-----------------------|-------|
| VEC+rerank MRR | 0.766 | 0.802 | **+0.036 (+4.7%)** |
| HYB+rerank MRR | — | 0.802 | — |

The 768d vectors have enough capacity to encode prefix semantics without losing code signal. This directly contradicts the "same 512-token window = same behavior" assumption. **Vector dimensionality matters for prefix tolerance**, not just context window.

Per-query highlights vs no-prefix:
- Q5 (exit intent): MISS → rank 1 (all modes)
- Q1 (frontend rendering): MISS → rank 3
- Q15 (admin settings): MISS → rank 2 (VEC)
- Q7 (popup conditionals): rank 1 held in VEC+rerank, MISS in HYB+rerank (regression)

**GTE-base results** (512-token window, 768d vectors, Jina-turbo reranker):

| Metric | Before (no prefix) | After (scope prefix) | Delta |
|--------|--------------------|-----------------------|-------|
| VEC+rerank MRR | 0.825 | 0.657 | **-0.168 (-20.4%)** |
| HYB+rerank MRR | — | 0.620 | — |

Catastrophic regression. The scope prefix destroys GTE-base quality despite having the same 768d dimensions as BGE-base (which *gained* +4.7%). This suggests the interaction is model-specific, not just dimension-dependent. GTE and BGE encode the natural language prefix differently — GTE may weight the prefix disproportionately, swamping the code signal.

**Revised conclusion**: Scope prefix behavior is model-specific, not predictable from dimensions or context window alone. Each model needs individual benchmarking. The profile `scope_prefix` flag must be set empirically per model, not derived from specs.

**Status**: Long-context model testing still needed. Current evidence:
- BGE-small 384d: scope prefix **hurts** (-3.9% VEC, +1.9% HYB)
- BGE-base 768d: scope prefix **helps** (+4.7% VEC)
- GTE-base 768d: scope prefix **destroys** (-20.4% VEC)

---

## Planned Experiments

### EXP-005: Scope-context on long-context models
Test EXP-004 on Nomic-embed-text (8192 tokens) and Jina-v2-base-code (8192 tokens) where the prefix cost is negligible. If MRR improves significantly, the scope prefix should be model-gated — on for long-context, off for 512-token models.

### EXP-006: Hybrid mode RRF weight retuning
Lower keyword weight in hybrid mode. Currently 1.0 — FTS5 tokenization doesn't align with NL queries against code. Test 0.5, 0.3, 0.1 weights.

### EXP-007: Multi-scale indexing
Dual FAISS index: method-level + file-level. RRF merge at search time. Addresses the noise problem from EXP-002 — fewer, more targeted chunks at each scale.

### EXP-008: Query-adaptive mode selection
Detect keyword-style vs NL-style queries, route to optimal search mode automatically.
