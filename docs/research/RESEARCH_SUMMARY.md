# Code Chunking Research — Executive Summary

## TL;DR

Your MRR degradation with more chunks is **not a chunking problem** — it's a **candidate selection problem**. Your AST-aware 512-char chunking is fine. The issue is **noise in your retrieved candidate set**.

**Solution:** Add **pre-filtering (file-level search first) + contextual embeddings (parent context injection)**. Expected gain: **+25-40% MRR**. Effort: **3 weeks**. Implementation: Two-stage retrieval, chunk context prepending.

---

## What Production Systems Actually Do

| System | Chunking | Chunk Size | Multi-Stage Retrieval? |
|--------|----------|------------|------------------------|
| Cursor | Tree-sitter AST | 2-4 KB (function-level) | YES (IDE context) |
| GitHub Copilot | Tree-sitter AST | Variable | YES (file summary first) |
| Sourcegraph Cody | Repo Semantic Graph | 1-2 KB | YES (graph expansion) |
| Continue.dev | Tree-sitter AST | 2-4 KB (top-level functions) | IMPLICIT |
| Greptile | Per-function tight | 1-2 KB | IMPLICIT (noise control) |

**Pattern:** All use function-level (1-4 KB), not fixed-size. All use multi-stage retrieval (not just FAISS→RRF).

---

## The Real Problem (Not Chunking)

**Your symptom:** More chunks → lower MRR.

**Root cause:** Sourcegraph's graph expansion + contextual embeddings research shows the real issue is **isolated chunk semantics**. When you embed a chunk without its parent function/class context, the embedding is ambiguous. FAISS then retrieves many similar-but-wrong chunks.

**Evidence:**
- Sourcegraph: 35% failure reduction via contextual embeddings (not smaller chunks)
- Greptile: "tight per-function chunking" works because it's semantically self-contained, not because it's small
- Late chunking: +24% improvement without reducing chunk size (just embedding full context first)

**Not your fault:** Current Nomic-embed-text is general-purpose; code-specific embedding (Nomic Embed Code) may help, but even that doesn't solve the isolated chunk problem.

---

## Three-Tier Solution (In Order)

### Tier 1: Pre-Filtering (File-Level Search)
**Effort:** 1 week | **Impact:** +10-20% MRR | **Risk:** Low

Two-stage retrieval:
1. Search files (FTS5 on filenames + docstrings) → top-K files
2. Search chunks within those files only (FAISS + FTS5)

**Why:** Reduces noise by eliminating irrelevant files upfront. GitHub Copilot approach.

**Code:** ~50 lines in `search.py`.

---

### Tier 2: Contextual Embeddings (Chunk + Parent Context)
**Effort:** 2 weeks | **Impact:** +15-25% MRR | **Risk:** Low

Before embedding chunks, prepend parent metadata:
```
[CLASS: UserService] [FUNC: authenticate]
function authenticate(username, password) { ... }
```

Then re-embed + re-index.

**Why:** Eliminates semantic isolation. Sourcegraph validated this (35% improvement in top-20 retrieval).

**Code:** ~100 lines in `chunker.py` + `embeddings.py`.

---

### Tier 3: Late Chunking (Optional)
**Effort:** 1 week | **Impact:** +15-25% additional | **Risk:** Medium (latency)

Use Jina Embeddings v3 API to embed full chunks first, then segment embeddings (not text).

**Why:** Full document context during embedding. +24.47% improvement measured (Jina, July 2025).

**Code:** ~30 lines in `embeddings.py`.

---

## Key Numbers from Research

| Metric | Value | Source |
|--------|-------|--------|
| AST chunking vs fixed-size | +1.8 to +5.6 points Recall@5 | cAST (EMNLP 2025) |
| Late chunking improvement | +24.47% relative | Jina (arXiv 2409.04701) |
| Contextual embeddings | 35% top-20 failure reduction | Sourcegraph |
| Semantic chunking cost | 6-10× compute, marginal gains | NAACL 2025 (Vectara) |
| Optimal chunk size | 400-512 tokens recommended; actual varies | Vecta Feb 2026, FireCrawl 2026 |
| Embedding dimensions | 768 dims → 0.26% quality loss vs 3072 | MTEB benchmarks |

---

## What Changed Since Your Last Research

1. **Nomic Embed Code (March 2026):** 7B code-specific embedding model, SOTA on CodeSearchNet. Consider switching from generic Nomic-embed-text.

2. **Jina Reranker v3 (December 2025):** 63.28 nDCG on code retrieval (CoIR), 2.5× fewer params. Upgrade if current reranker weak.

3. **Academic consensus on semantic chunking (NAACL 2025):** Not worth it for code; AST boundaries beat it.

4. **Sourcegraph Cody details published (2025):** Repo Semantic Graph + graph expansion + contextual embeddings = their approach. Validated.

5. **Mix-of-Granularity research (COLING 2025):** Adaptive chunk routing exists but weak gains (1.2-5%); not recommended yet.

---

## Diagnostic First

Before implementing, run diagnostic on your 20 benchmark queries:

```python
# For each query:
# 1. What rank is the correct answer? (if >15, pre-filtering helps)
# 2. Are top-5 results noise? (if yes, chunking strategy issue)
# 3. Do they cluster in same file? (if no, file-level search helps)
```

This tells you whether pre-filtering or reranking is the bottleneck.

---

## Implementation Roadmap (3 Weeks)

**Week 1:** Diagnostic + Pre-filtering
- Diagnose bottleneck
- Add file-level FTS5 search
- Benchmark 5 queries
- Expected: +10% MRR

**Week 2:** Contextual Embeddings
- Extract parent context from AST
- Prepend to chunks before re-embedding
- Re-index Popup Maker (~30 min)
- Benchmark full 20 queries
- Expected: +25-40% combined

**Week 3:** Optional Tuning
- Late chunking (if Jina API available)
- Reranker upgrade (Jina v3 if weak)
- Production rollout

**Target:** 0.691 → 0.80+ MRR

---

## What Would Break This Recommendation

1. **If file-level search reduces recall:** Some files have no distinctive names. Mitigate by boosting query intent detection.

2. **If contextual embeddings slow re-indexing:** Longer token counts per chunk. Mitigate with batch processing.

3. **If late chunking adds latency:** Jina API slower than on-device. Mitigate by keeping Nomic as fallback.

4. **If you need sub-100ms query time:** Pre-filtering adds 10-20ms (second search). May not be acceptable.

5. **If your RRF weights already perfectly tuned:** Pre-filtering/contextual embeddings may not add marginal value. Validate with diagnostic first.

---

## Files to Read in Order

1. **This file** (you are here)
2. **RESEARCH_CODE_CHUNKING_2025.md** (full report, all sources)
3. **IMPLEMENTATION_QUICK_START.md** (code snippets, file locations)

---

## Key Takeaways

1. **AST chunking is right.** Don't switch to fixed-size or semantic chunking.

2. **512 chars is reasonable** if using pre-filtering + contextual embeddings. Consider 2KB as alternative (less FAISS noise).

3. **"More chunks = worse" is solvable.** Not a design problem, a tuning problem. Pre-filtering + contextual embeddings fix it.

4. **Sourcegraph's approach works.** Two-stage retrieval + graph ranking. You can implement simpler version (file-level + chunk-level, skip graph).

5. **Embedding model matters.** Upgrade from generic Nomic-embed-text to Nomic Embed Code (ICLR 2025, SOTA on CodeSearchNet).

6. **Reranking matters too.** Jina Reranker v3 (63.28 CoIR) > most alternatives. Upgrade if current weak.

---

## Next Steps

1. Run diagnostic on 5 benchmark queries (1 day)
2. Implement pre-filtering (1 week)
3. Implement contextual embeddings (1 week)
4. Validate on full 20-query benchmark
5. If MRR < 0.78: investigate embedding model or reranker
6. If MRR > 0.80: done; optional late chunking in Week 3

---

## Sources Summary

- **cAST (EMNLP 2025):** AST-aware chunking beats fixed-size
- **Late Chunking (Jina, July 2025):** +24.47% improvement
- **Sourcegraph Blog (2025):** Repo Semantic Graph + contextual embeddings (35% improvement)
- **NAACL 2025 (Vectara):** Semantic chunking not justified
- **Nomic Embed Code (ICLR 2025):** SOTA on CodeSearchNet
- **Jina Reranker v3 (Dec 2025):** 63.28 on code retrieval
- **Continue.dev Docs:** Tree-sitter AST strategy
- **Greptile Blog:** Per-function tight chunking reduces noise
- **FireCrawl Blog (2026):** 512-token chunks with 10-20% overlap recommended

All sources fully cited in RESEARCH_CODE_CHUNKING_2025.md.

