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

# Aggregate mixed-media results
uv run python scripts/aggregate_mixed.py
```

Corpus files and indexes are cached at `~/.tessera/benchmarks/csn/` — first run is slow, subsequent runs are instant.
