# Benchmarks

Real-world search quality measurements on a production PHP codebase: [Popup Maker](https://wppopupmaker.com/) core + Pro (611 files, 2,574 chunks across two repositories).

## Test Suite

20 ground-truth queries spanning the full codebase — popup lifecycle, trigger system, conditions, cookies, forms, REST API, admin, DI architecture, and Pro features. Each query has 1-2 expected files that a developer would navigate to when investigating that topic.

**Metrics:**

- **MRR** (Mean Reciprocal Rank): How high the correct file ranks on average. 1.0 = always rank 1.
- **Top-K Accuracy**: Percentage of queries where the correct file appears in the top K results.

## Embedding Model Comparison

All models tested with VEC+code mode (semantic search, code files only) and VEC+rerank where a reranker is available.

### With Reranker (Best Mode)

| Setup | Embedding | Reranker | Download | MRR | Top-1 | Top-3 | Top-10 |
|-------|-----------|---------|----------|-----|-------|-------|--------|
| Gateway + rerank | Nomic 768d | Jina reranker | Server required | **0.854** | **75%** | **95%** | **100%** |
| Gateway + rerank | Qwen3 1024d | Jina reranker | Server required | 0.825 | 70% | 100% | 100% |
| Local ~200MB | BGE-small 384d | Jina-tiny reranker | ~197MB | 0.739 | 60% | 85% | 95% |
| Local ~100MB | Arctic-XS 384d | MiniLM reranker | ~103MB | 0.649 | 50% | 65% | 95% |

### Without Reranker (Embedding Only)

| Setup | Embedding | Download | MRR | Top-1 | Top-3 | Top-10 |
|-------|-----------|----------|-----|-------|-------|--------|
| Gateway | Nomic 768d | Server required | 0.696 | 55% | 75% | 100% |
| Local ~67MB | BGE-small 384d | ~67MB | 0.609 | 45% | 80% | 95% |
| Gateway | Qwen3 1024d | Server required | 0.615 | 45% | 70% | 100% |
| Local ~23MB | Arctic-XS 384d | ~23MB | 0.626 | 45% | 75% | 90% |

**Key findings:**

- Cross-encoder reranking is the single biggest quality lever (+0.158 MRR over VEC-only).
- The 200MB local stack (BGE-small + Jina-tiny) achieves 86% of the gateway's best score with zero server setup.
- Higher embedding dimensions (1024d Qwen3) do not beat lower dimensions (768d Nomic) — model training quality matters more than vector size.
- The 100MB ultra-compact tier drops too far — the reranker cannot compensate for weak embeddings.

## Search Mode Comparison

All modes tested with Nomic 768d via gateway.

| Mode | Description | MRR | Top-1 | Top-3 | Top-10 | Avg Latency |
|------|-------------|-----|-------|-------|--------|-------------|
| VEC+rerank | Semantic + cross-encoder reranking | **0.854** | **75%** | **95%** | **100%** | 298ms |
| HYB+rerank | Hybrid + cross-encoder reranking | 0.817 | 75% | 90% | 90% | 202ms |
| VEC+code | Semantic only, code files | 0.696 | 55% | 75% | 100% | 13ms |
| VEC+PPR | Semantic + PageRank graph | 0.647 | 45% | 80% | 95% | 16ms |
| HYBRID+code | Keyword + semantic, code files | 0.550 | 40% | 60% | 80% | 9ms |
| LEX-only | FTS5 keyword only | 0.307 | 15% | 25% | 35% | 5ms |

**Key findings:**

- Semantic search (VEC) dominates keyword search (LEX) for natural language queries against code.
- PPR graph ranking helps structural queries (e.g., "singleton registry for popup trigger types" jumps from rank 2 to rank 1) but is gated to only fire when query terms match actual symbol names — preventing noise on conceptual queries.
- HYBRID mode underperforms VEC-only because FTS5 tokenization doesn't align well with natural language queries against PHP code.
- Reranking adds ~200-300ms latency but the quality gain is substantial.

## PPR Graph Ranking

PageRank-based ranking uses the code's reference graph (who-calls-what) to boost structurally important files. Tessera gates PPR activation on symbol name matching — it only fires when query terms match actual symbol names in the index.

| Query | Without PPR | With PPR | Change |
|-------|-------------|----------|--------|
| Frontend rendering (`Popups.php`) | rank 5 | **rank 1** | Structural hub |
| Trigger registry (`Triggers.php`) | rank 2 | **rank 1** | High fan-in symbol |
| Newsletter AJAX (`Subscribe.php`) | rank 9 | rank 7 | Weak structural signal |
| Scheduling (`scheduling.php`) | rank 4 | rank 6 | PPR noise (gated in current build) |

## Recommended Configuration

### Zero-Config (pip install)

```bash
pip install tessera-idx[embed]
tessera index /path/to/project
```

Uses BGE-small-en-v1.5 (67MB, 384d) + Jina-reranker-v1-tiny (130MB). Total ~200MB downloaded on first run. No GPU, no server, no config.

**Expected quality:** 0.739 MRR, 85% Top-3, 95% Top-10.

### Maximum Quality (model server)

Run an embedding + reranker server (e.g., LM Studio, vLLM, or a local gateway):

```bash
tessera index /path/to/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings \
  --embedding-model nomic-embed
```

With Nomic-embed-text (768d) + Jina cross-encoder reranking.

**Expected quality:** 0.854 MRR, 95% Top-3, 100% Top-10.

## Reproducing

```bash
# Self-benchmark (indexes Tessera's own codebase)
uv run python scripts/benchmark_quick.py

# Full PM benchmark (requires Popup Maker source)
uv run python scripts/benchmark_pm.py --all              # gateway models
uv run python scripts/benchmark_pm.py --provider fastembed --all  # local models
```
