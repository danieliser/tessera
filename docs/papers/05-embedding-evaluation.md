# Paper 5: Model-Agnostic Embedding Evaluation for Practical Code Retrieval

**Status:** IDEA
**Target venue:** EMNLP 2026 (short paper) or ACL Findings

## Abstract (Draft)

Code embedding models are typically evaluated on benchmarks like CodeSearchNet and MTEB, which test natural language to code retrieval using docstring-derived queries. These benchmarks have driven the development of increasingly large, code-specialized models — Jina-Code-v2 (560MB), CodeRankEmbed, Nomic-Code-7B — under the assumption that code-specific pre-training is necessary for code retrieval quality.

We challenge this assumption with a controlled evaluation methodology: identical index, identical queries, identical fusion pipeline — swap only the embedding model. Using a curated benchmark of 38 real-world code navigation queries against a production codebase (3,677 chunks), we find that BGE-small-en-v1.5 (67MB, 384 dimensions, general-purpose) achieves a blend MRR of 0.87, outperforming Jina-Code-v2 (560MB, 768d, code-specialized) at 0.45 blend MRR.

Our analysis reveals why: in a multi-signal retrieval system with keyword and graph ranking, the embedding model's primary role is semantic re-ranking of candidates already surfaced by structural signals. A smaller, faster model that provides adequate semantic discrimination is more valuable than a larger model with superior standalone retrieval but identical fusion-stage behavior. This finding has practical implications: teams deploying code search systems should evaluate embedding models within their full retrieval pipeline, not on standalone benchmarks.

## Key Claims

1. **General-purpose embeddings can outperform code-specialized embeddings** when combined with structural signals (keyword + graph).

2. **Standalone embedding benchmarks are misleading** for system-level retrieval quality — a model's MTEB score does not predict its contribution to a fusion pipeline.

3. **Model size and dimensionality are poor predictors** of retrieval quality in multi-signal systems. 67MB/384d matched or beat 560MB/768d.

4. **Practical recommendation:** evaluate embeddings within the deployment pipeline, not on standalone benchmarks.

## Required Literature Review

- [ ] MTEB benchmark and methodology
- [ ] CodeSearchNet model evaluations
- [ ] Code embedding model papers (CodeBERT, GraphCodeBERT, UniXcoder, StarEncoder)
- [ ] BGE model family papers
- [ ] Jina-Code-v2 technical report
- [ ] Nomic Embed papers
- [ ] Fusion-stage re-ranking literature
- [ ] Embedding model efficiency studies

## Data / Experiments Needed

- BGE-small vs. Jina-Code-v2 comparison (already have)
- Nomic-embed-text comparison (already have)
- CodeRankEmbed comparison (partially complete)
- Additional models: StarEncoder, UniXcoder, CodeSage
- Per-query-category breakdown (code vs doc vs cross-ref per model)
- Latency comparison (embedding time, search time, total pipeline)
- Standalone retrieval comparison (embeddings only, no fusion) to demonstrate the gap
- Multiple codebases (not just PM) for generalization

## Risks

- "Your benchmark is too small" — same risk as Paper 3
- "The general-purpose model might just be better in general" — need standalone comparison to show it's specifically the fusion that changes the picture
- Fast-moving field — new models may obsolete specific comparisons before publication
- Need to be careful about making claims that generalize beyond our evaluation
