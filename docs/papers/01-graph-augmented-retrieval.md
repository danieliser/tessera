# Paper 1: Deterministic Graph-Augmented Retrieval for Multi-Source Code Intelligence

**Status:** IDEA
**Target venue:** ICSE 2027 / FSE 2027 (full paper, 12 pages)

## Abstract (Draft)

Code retrieval systems face a fundamental tension: lexical methods excel at identifier lookup but fail on conceptual queries, while neural methods capture semantic similarity but lose structural precision. We present Tessera, a federated code search system that resolves this tension through weighted Reciprocal Rank Fusion (RRF) of three orthogonal signals: FTS5 keyword matching, FAISS vector similarity, and PageRank-based graph ranking. Critically, the graph signal is derived deterministically from tree-sitter AST analysis — not learned from training data — making it language-agnostic, reproducible, and free of training distribution bias.

We evaluate Tessera on a curated benchmark of 38 real-world code navigation queries against a production Next.js codebase (3,677 chunks, 1,729 files), spanning code retrieval, documentation retrieval, and cross-reference queries. Our deterministic graph-augmented approach achieves a blend MRR of 0.87, with the graph signal providing the decisive ranking boost on structural queries (call chains, inheritance, hook dependencies) where both lexical and neural signals produce equivalent candidates.

We further demonstrate that the deterministic graph approach generalizes across programming languages (PHP, TypeScript, Python, JavaScript) without retraining, and that a 67MB general-purpose embedding model combined with graph re-ranking outperforms 560MB code-specialized models that lack structural signals.

## Key Claims

1. **Deterministic AST graphs beat learned graph representations** for code navigation tasks where structural relationships (calls, imports, inheritance, events) are the primary retrieval signal.

2. **Three-signal RRF fusion** (keyword + semantic + graph) provides robust retrieval across query types that no single signal handles well alone.

3. **Graph signal is the tiebreaker** — when keyword and semantic signals produce equivalent candidates (which happens frequently in codebases with consistent naming), PageRank from the call graph promotes structurally central results.

4. **Language-agnostic extraction** via tree-sitter means no per-language model training. Adding a new language requires only a grammar and an extractor module (~200 lines).

## Novelty vs. Prior Art

| Prior work | What they do | What we add |
|---|---|---|
| GraphCodeBERT (Guo et al., 2021) | Learns data flow graphs during pre-training | Deterministic graphs, no training required |
| CodeBERT (Feng et al., 2020) | Bimodal pre-training on NL-code pairs | Graph signal absent entirely |
| Sourcegraph | Lexical search + precise code intelligence | No semantic signal, no fusion ranking |
| GitHub code search | Lexical + limited semantic | No graph signal, no RRF fusion |
| Astra (Microsoft) | Graph neural networks for code | Requires training per-language |

**Gap:** No published system combines deterministic graph ranking with neural retrieval via RRF fusion for code search.

## Required Literature Review

- [ ] RRF in information retrieval (Cormack et al., 2009)
- [ ] Graph-based code representations (Allamanis et al., 2018)
- [ ] GraphCodeBERT and successors
- [ ] Tree-sitter applications in research
- [ ] PageRank for software dependency analysis
- [ ] Code search evaluation methodologies

## Data / Experiments Needed

- PM20/PM38 benchmark results (already have)
- Cross-model comparison data (BGE-small, Jina, Nomic — already have)
- Ablation study: RRF with/without graph signal
- Ablation study: deterministic graph vs. no graph vs. learned graph
- Latency benchmarks
- Multi-language evaluation (PHP + TS + Python on same codebase)

## Risks

- Reviewers may argue the benchmark is too small / single-project
- Need to address: does this generalize beyond the PM ecosystem?
- Tree-sitter coverage gaps (languages without grammars) could be raised
