# Benchmarks

Real-world search quality measurements on a production PHP codebase: [Popup Maker](https://wppopupmaker.com/) core + Pro (611 files, 2,574 chunks across two repositories).

## Test Suite

20 ground-truth queries spanning the full codebase — popup lifecycle, trigger system, conditions, cookies, forms, REST API, admin, DI architecture, and Pro features. Each query has 1-2 expected files that a developer would navigate to when investigating that topic.

**Metrics:**

- **MRR** (Mean Reciprocal Rank): How high the correct file ranks on average. 1.0 = always rank 1.
- **Top-K Accuracy**: Percentage of queries where the correct file appears in the top K results.

## Recommended Configurations

### Zero-Config Default (~200MB)

```bash
pip install tessera-idx[embed]
tessera index /path/to/project
```

Uses **BGE-small-en-v1.5** (67MB, 384d) + **Jina-reranker-v1-tiny** (130MB). Total ~200MB downloaded on first run. No GPU, no server, no config.

**Expected quality:** 0.739 MRR, 85% Top-3, 95% Top-10.

### Quality Local (~340MB)

```bash
tessera index /path/to/project \
  --embedding-model BAAI/bge-base-en-v1.5
```

Uses **BGE-base-en-v1.5** (210MB, 768d) + **Jina-reranker-v1-tiny** (130MB).

**Expected quality:** 0.766 MRR, 85% Top-3, 100% Top-10.

### Maximum Local (~590MB)

```bash
tessera index /path/to/project \
  --embedding-model thenlper/gte-base \
  --reranking-model jinaai/jina-reranker-v1-turbo-en
```

Uses **GTE-base** (440MB, 768d) + **Jina-reranker-v1-turbo** (150MB). Matches gateway-quality scores with zero server setup.

**Expected quality:** 0.825 MRR, 90% Top-3, 100% Top-10.

### Maximum Quality (model server)

Run an embedding + reranker server (e.g., LM Studio, vLLM, or a local gateway):

```bash
tessera index /path/to/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings \
  --embedding-model nomic-embed
```

With **Nomic-embed-text** (768d) + Jina cross-encoder reranking via HTTP.

**Expected quality:** 0.854 MRR, 95% Top-3, 100% Top-10.

---

## Full Embedding Model Comparison

All 12 fastembed-compatible models tested with VEC+code mode (semantic search, code files only), sorted by reranked MRR. Reranker: Jina-reranker-v1-tiny (130MB) unless noted.

| Model | Size | Dim | Index Time | VEC MRR | Top-1 | Top-3 | Top-10 | +Rerank MRR | Top-1 | Top-3 | Top-10 |
|-------|------|-----|-----------|---------|-------|-------|--------|-------------|-------|-------|--------|
| **GTE-base** | 440MB | 768d | 183s | 0.696 | 55% | 80% | 100% | 0.743 | 60% | 85% | 100% |
| **BGE-small** | 67MB | 384d | 64s | 0.609 | 45% | 80% | 95% | 0.739 | 60% | 85% | 95% |
| **BGE-base** | 210MB | 768d | 187s | 0.621 | 45% | 85% | 90% | **0.766** | 65% | 85% | 100% |
| Jina-Code | 640MB | 768d | 1109s | 0.475 | 35% | 50% | 70% | 0.716 | 55% | 85% | 95% |
| Arctic-XS | 90MB | 384d | 32s | 0.626 | 45% | 75% | 90% | 0.704 | 60% | 70% | 95% |
| Arctic-S | 130MB | 384d | 59s | 0.623 | 40% | 85% | 100% | 0.704 | 55% | 80% | 100% |
| Arctic-M | 430MB | 768d | 182s | 0.490 | 35% | 55% | 80% | 0.645 | 55% | 65% | 85% |
| MxBAI-large | 640MB | 1024d | 706s | 0.592 | 40% | 80% | 90% | 0.625 | 45% | 75% | 95% |
| Jina-small | 120MB | 512d | 195s | 0.440 | 30% | 55% | 75% | 0.617 | 50% | 70% | 80% |
| MiniLM-L6 | 90MB | 384d | 21s | 0.422 | 25% | 50% | 80% | 0.609 | 45% | 70% | 90% |
| Nomic-full | 520MB | 768d | 1011s | 0.401 | 30% | 40% | 65% | 0.468 | 35% | 55% | 70% |
| Nomic-Q | 130MB | 768d | 790s | 0.346 | 25% | 40% | 50% | 0.399 | 30% | 45% | 60% |

Self-hosted gateway models (HTTP endpoint required):

| Model | Dim | VEC MRR | Top-1 | Top-3 | Top-10 | +Rerank MRR | Top-1 | Top-3 | Top-10 |
|-------|-----|---------|-------|-------|--------|-------------|-------|-------|--------|
| **Nomic-embed-text** | 768d | 0.696 | 55% | 75% | 100% | **0.854** | 75% | 95% | 100% |
| Qwen3-embed | 1024d | 0.615 | 45% | 70% | 100% | 0.825 | 70% | 100% | 100% |

Cloud API models (paid, per-token pricing):

| Model | Dim | Cost | VEC MRR | Top-1 | Top-3 | Top-10 | +Rerank MRR | Top-1 | Top-3 | Top-10 |
|-------|-----|------|---------|-------|-------|--------|-------------|-------|-------|--------|
| OpenAI text-embedding-3-large | 1024d | $0.13/1M | 0.566 | 35% | 75% | 100% | 0.722 | 60% | 80% | 100% |
| OpenAI text-embedding-3-large | 3072d | $0.13/1M | 0.571 | 35% | 75% | 100% | 0.687 | 55% | 80% | 100% |
| OpenAI text-embedding-3-small | 1536d | $0.02/1M | 0.558 | 40% | 65% | 80% | 0.668 | 60% | 65% | 85% |
| OpenAI text-embedding-3-small | 512d | $0.02/1M | 0.491 | 35% | 60% | 80% | 0.627 | 55% | 70% | 85% |

!!! warning "OpenAI embeddings underperform on code search"
    OpenAI's general-purpose embeddings score significantly below code-trained local models on this benchmark. Their best configuration (text-embedding-3-large at 1024d + local reranker, 0.722 MRR) loses to the free 67MB BGE-small (0.739 MRR). OpenAI's models are optimized for general text retrieval, not code search.

## Cross-Test: Embedder x Reranker Matrix

The top 4 local embedders tested against all 4 rerankers. The best reranker depends on which embedder you use.

| Embedder | Reranker | Total Size | MRR | Top-1 | Top-3 | Top-10 |
|----------|----------|------------|-----|-------|-------|--------|
| GTE-base 768d | Jina-turbo (150MB) | 590MB | **0.825** | 70% | 90% | 100% |
| GTE-base 768d | MiniLM-L12 (120MB) | 560MB | 0.806 | 70% | 95% | 100% |
| GTE-base 768d | MiniLM-L6 (80MB) | 520MB | 0.795 | 70% | 85% | 100% |
| BGE-base 768d | Jina-tiny (130MB) | 340MB | 0.766 | 65% | 85% | 100% |
| GTE-base 768d | Jina-tiny (130MB) | 570MB | 0.743 | 60% | 85% | 100% |
| BGE-small 384d | Jina-tiny (130MB) | 197MB | 0.739 | 60% | 85% | 95% |
| BGE-base 768d | Jina-turbo (150MB) | 360MB | 0.731 | 55% | 95% | 100% |
| BGE-base 768d | MiniLM-L12 (120MB) | 330MB | 0.726 | 60% | 90% | 100% |
| BGE-small 384d | MiniLM-L12 (120MB) | 187MB | 0.721 | 60% | 85% | 95% |
| BGE-base 768d | MiniLM-L6 (80MB) | 290MB | 0.718 | 60% | 85% | 100% |
| BGE-small 384d | Jina-turbo (150MB) | 217MB | 0.708 | 55% | 85% | 95% |
| BGE-small 384d | MiniLM-L6 (80MB) | 147MB | 0.703 | 60% | 75% | 95% |

!!! note "Reranker interaction matters"
    Jina-tiny is the best reranker for BGE models (0.739, 0.766) but the *worst* for GTE-base (0.743 vs 0.825 with Jina-turbo). Always cross-test your specific combination.

## Reranker Comparison

All rerankers tested with GTE-base (768d) embeddings.

| Reranker | Size | MRR | Top-1 | Top-3 | Top-10 |
|----------|------|-----|-------|-------|--------|
| **Jina-turbo** | 150MB | **0.825** | 70% | 90% | 100% |
| MiniLM-L12 | 120MB | 0.806 | 70% | 95% | 100% |
| MiniLM-L6 | 80MB | 0.795 | 70% | 85% | 100% |
| Jina-tiny | 130MB | 0.743 | 60% | 85% | 100% |

## Key Findings

- **Cross-encoder reranking is the single biggest quality lever.** +0.13-0.16 MRR over embedding-only search across all models.
- **Free local models beat paid cloud APIs for code search.** BGE-small (67MB, free) scores 0.739 MRR vs OpenAI text-embedding-3-large ($0.13/1M) at 0.722. General-purpose cloud embeddings aren't trained for code retrieval.
- **Bigger is NOT better for local ONNX models.** Nomic-full (520MB) and MxBAI-large (640MB) scored worse than BGE-small (67MB). ONNX quantization and model architecture matter more than parameter count.
- **The 200MB default stack (BGE-small + Jina-tiny) is the sweet spot for zero-config.** 87% of the gateway's best score, zero server setup, fast indexing (64s).
- **590MB gets you gateway-level quality locally.** GTE-base + Jina-turbo hits 0.825 MRR — matching the Qwen3 gateway setup.
- **Higher dimensions don't guarantee better results.** BGE-small (384d) outperforms MxBAI-large (1024d) and OpenAI's 3072d model. Model training quality dominates.
- **Nomic ONNX quantized variants perform poorly.** The fastembed Nomic-Q (0.399 MRR) is dramatically worse than Nomic via HTTP gateway (0.854). The quantization destroys quality for this model.
- **Reranker-embedder interaction is real.** Jina-tiny pairs best with BGE models; Jina-turbo pairs best with GTE. Always cross-test.
- **OpenAI dimension reduction helps with reranking.** text-embedding-3-large at 1024d (0.722) outperforms full 3072d (0.687) when paired with a cross-encoder reranker — denser representations give the reranker more signal.

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

## Reproducing

All benchmarks are fully reproducible using the scripts in the `scripts/` directory.

```bash
# Self-benchmark (indexes Tessera's own codebase)
uv run python scripts/benchmark_quick.py

# Full PM benchmark — single model (requires Popup Maker source)
uv run python scripts/benchmark_pm.py --all              # gateway models (HTTP endpoint)
uv run python scripts/benchmark_pm.py --provider fastembed --all  # local fastembed models

# Batch benchmark — all 12 local embedding models + 4 rerankers
uv run python scripts/benchmark_all_models.py              # embedding models only
uv run python scripts/benchmark_all_models.py --rerankers   # + reranker comparison

# Cross-test matrix — top embedders × all rerankers
uv run python scripts/benchmark_cross.py

# Cloud API benchmark — OpenAI (requires OPENAI_API_KEY)
uv run python scripts/benchmark_cloud.py                    # all OpenAI variants
uv run python scripts/benchmark_cloud.py --voyage           # + Voyage (needs VOYAGE_API_KEY)
```

### Requirements

- **Local benchmarks**: Requires `pip install tessera-idx[embed]` (installs fastembed). Models auto-download on first run.
- **Gateway benchmarks**: Requires an OpenAI-compatible embedding endpoint (e.g., LM Studio, vLLM) at `http://localhost:8800/v1/embeddings`
- **Cloud benchmarks**: Requires `OPENAI_API_KEY` env var. Optional `VOYAGE_API_KEY` for Voyage models. API costs are minimal (~$0.50 for all 4 OpenAI model variants).
- **PM benchmark**: Requires Popup Maker core + Pro source code at `~/Projects/ProContent/ProductCode/popup-maker{,-pro}`
