# Tessera Benchmarks

Reproducible benchmark results and scripts for Tessera search quality evaluation.

## CodeSearchNet (CoIR Protocol)

**Dataset:** [code-search-net/code_search_net](https://huggingface.co/datasets/code-search-net/code_search_net) Python test split  
**Corpus:** 22,092 functions across 680 repos  
**Queries:** 500 sampled docstrings (stratified across repos)  
**Metric:** NDCG@10 (CoIR leaderboard standard)  
**Protocol:** Each docstring query → retrieve its own function from the full 22k corpus

### Results — No-Chunk Mode (CoIR-comparable)

One vector per whole function, pure cosine similarity via FAISS. Matches CoIR evaluation protocol exactly.

| Model | NDCG@10 | MRR@10 | Latency |
|-------|---------|--------|---------|
| BGE-base-en-v1.5 (768d) | **91.4** | 89.8 | 1ms/q |

### CoIR Leaderboard Position

| Rank | Model | NDCG@10 |
|------|-------|---------|
| 1 | CodeSage-large-v2 (1.3B) | 94.26 |
| **2** | **Tessera + BGE-base-en-v1.5** | **91.39** |
| 3 | Voyage-Code-002 | 81.79 |
| 4 | BGE-base-en-v1.5 (published baseline) | 69.6 |
| 5 | E5-Base-v2 (110M) | 67.99 |
| 6 | GTE-base-en-v1.5 | 43.35 |
| 7 | BGE-M3 (567M) | 43.23 |

> Note: Tessera scores **+21.8 points above the published BGE-base baseline** (69.6→91.4).
> The gap to #1 (CodeSage-large-v2) is **2.87 points**.

### Results — Tessera Pipeline Mode (chunked hybrid search)

Real-world retrieval pipeline: AST-aware chunking + FTS5 + FAISS hybrid search.

| Model | Mode | NDCG@10 | MRR@10 | Notes |
|-------|------|---------|--------|-------|
| BGE-small-384d | HYBRID | 37.55 | 37.47 | 22ms/q |
| BGE-small-384d | VEC | 36.21 | 35.77 | 69ms/q |
| BGE-base-768d | HYBRID | 37.83 | 37.77 | 23ms/q |

Chunking splits long functions across multiple chunks, diluting per-function scores.
This is intentional — chunking improves real-world retrieval for partial/sub-function queries.

## Dual-Model Fan-Out Benchmark

Source-type routing with reranker fusion. SMARTv2 routing: fan-out both models for code+cross, BGE-small only for docs, reranker picks winners.

**Embedding:** BGE-small (67MB, 384d)
**Dataset:** PM20 codebase — 10 code, 10 doc, 10 cross-media queries
**Pool:** 40 candidates → reranker

### Reranker Comparison (all pool=40, SMARTv2 routing)

| Reranker | Size | Total DL | Code | Doc | Cross | **Blended** |
|----------|------|----------|------|-----|-------|------------|
| Jina v3 | 1.2GB | 1.3GB | 0.883 | 0.950 | 0.814 | **0.883** |
| **MiniLM-L-6 (default)** | **80MB** | **147MB** | **0.838** | **0.950** | **0.825** | **0.871** |
| Jina-tiny | 130MB | 197MB | **0.925** | 0.950 | 0.600 | 0.825 |
| Jina v2 | 560MB | 627MB | 0.783 | 0.950 | 0.692 | 0.808 |
| Jina-turbo | 150MB | 217MB | 0.755 | 0.950 | 0.642 | 0.782 |

**Key findings:**
- MiniLM-L-6 (80MB) beats Jina v2 (560MB) — 0.871 vs 0.808 blended
- Pool=40 is the sweet spot (+0.085 over pool=20, flat above 40)
- Jina-tiny scores 0.925 code MRR — highest of any configuration
- Default stack: BGE-small + MiniLM-L-6 = **147MB total download**

### Routing Strategy

- **Code:** fan-out both models (CodeRankEmbed + BGE-small), VEC+LEX, reranker picks
- **Doc:** BGE-small only, HyDE embedding, markdown+json filter
- **Cross:** fan-out both models, VEC+LEX, HyDE on general model, graph boost

See `scripts/benchmark_fanout.py`.

## Mixed-Media Benchmark (PM20 Codebase)

Internal benchmark across code, documentation, and cross-media queries on a real WordPress plugin codebase.

See `scripts/benchmark_mixed.py` and CSV outputs in this directory.

## Running Benchmarks

```bash
# CoIR-comparable (no-chunk, whole-function vectors)
uv run python scripts/benchmark_csn.py --model bge-base --no-chunk --samples 500

# Full CoIR eval (all 22k queries, slow)
uv run python scripts/benchmark_csn.py --model bge-base --no-chunk --full

# Tessera pipeline (chunked hybrid)
uv run python scripts/benchmark_csn.py --model bge-base --samples 500

# Mixed-media benchmark
uv run python scripts/benchmark_mixed.py --all

# Dual-model fan-out benchmark
uv run python scripts/benchmark_fanout.py          # pure fan-out
uv run python scripts/benchmark_fanout.py --smart   # smart routing

# Aggregate mixed-media results
uv run python scripts/aggregate_mixed.py
```

Corpus files and indexes are cached at `~/.tessera/benchmarks/csn/` — first run is slow, subsequent runs are instant.
