# Embedding Model Testing Plan for Tessera

**Based on:** Research Report `docs/research/embedding-models.md`
**Date:** 2026-03-02
**Owner:** Tessera Team

## Overview

This document outlines a minimal validation plan to confirm Jina Code Embeddings 1.5B is production-ready for Tessera's MLX embedding server before full deployment.

---

## Critical Unknowns

Three assumptions must be validated before production use:

1. **Matryoshka Truncation Quality:** Does Jina 1.5B's 1536d → 384d truncation retain >95% CoIR quality?
2. **Apple Silicon Latency:** Does Jina 1.5B inference on M-series hardware meet <100ms SLA?
3. **RRF Fusion Effectiveness:** Does FTS5 keyword + vector (384d) RRF match or exceed Jina at full dimensions?

---

## Test Phase 1: Matryoshka Truncation (1–2 days)

### Objective
Validate that Jina 1.5B's Matryoshka truncation from 1536d → 384d doesn't degrade code retrieval quality.

### Method

1. **Setup:**
   ```bash
   # Download Jina 1.5B
   from transformers import AutoModel
   model = AutoModel.from_pretrained("jinaai/jina-code-embeddings-1.5b", trust_remote_code=True)
   ```

2. **Benchmark:** Use CodeSearchNet code search dataset subset (100–500 queries)
   - Baseline: Full 1536d embeddings
   - Test: 384d truncated embeddings (via Matryoshka)
   - Metric: NDCG@5, NDCG@10, MRR

3. **Acceptance Criteria:**
   - NDCG@10 degradation < 5% (acceptable; RRF compensates for remainder)
   - MRR stable (within 2%)
   - If >5% degradation: fallback to Nomic 1.5 + stronger RRF

### Expected Outcome
Quality retention ~95–99% (typical for Matryoshka). If validated, proceed to Phase 2.

---

## Test Phase 2: Apple Silicon Latency (1 day)

### Objective
Measure inference latency on target M-series hardware (M1, M2, M3) to ensure <100ms per embedding.

### Method

1. **Setup:**
   - Test hardware: M1/M2 MacBook Pro, M3 Max if available
   - Framework: MLX or ONNX Runtime with CoreML backend
   - Batch sizes: 1 (single embedding), 32 (batch processing)

2. **Benchmarks:**
   ```
   For single embedding (batch_size=1):
   - Latency (ms): record P50, P95, P99
   - Memory: peak heap usage
   - Throughput: embeddings/second

   For batch (batch_size=32):
   - Throughput: embeddings/second
   - Memory: peak heap usage
   ```

3. **Acceptance Criteria:**
   - P50 latency < 50ms (target for single embedding)
   - P95 latency < 100ms (fallback threshold)
   - Throughput > 20 embeddings/sec on M1 (or proportional to hardware)
   - Memory footprint < 3GB on M1 (INT8 quantized)

### Expected Outcome
Latency 20–50ms on M1 (estimated from BERT-scale benchmarks). If >100ms: evaluate Nomic 1.5 as alternative.

---

## Test Phase 3: RRF Fusion Effectiveness (2 days)

### Objective
Confirm that keyword (FTS5) + vector (384d) RRF fusion maintains retrieval quality on code search.

### Method

1. **Setup:**
   - Indexing: 500–1000 code chunks from target languages (Python, TypeScript, PHP, Swift)
   - FTS5 index: on code comments + identifiers
   - FAISS vector index: 384d Jina embeddings (Matryoshka truncated)

2. **Queries:** 50 natural language code search queries (e.g., "find function that parses JSON")

3. **Metrics:**
   - RRF (FTS5 + 384d Jina) NDCG@10
   - Pure 384d Jina NDCG@10
   - Pure FTS5 NDCG@10
   - Combined RRF ranking

4. **Acceptance Criteria:**
   - RRF NDCG@10 ≥ 75% (absolute quality)
   - RRF NDCG > pure vector by ≥ 5% (fusion benefit demonstrated)
   - No keyword queries regress >10% vs. FTS5 baseline

### Expected Outcome
RRF fusion recovers 10–15% NDCG from 384d dimension reduction. If combined quality < 70%, escalate to Phase 4.

---

## Test Phase 4: Fallback Evaluation (Optional, 1 day)

### Objective
If Phases 1–3 fail acceptance criteria, evaluate Nomic Embed Text V1.5 as replacement.

### Method

Repeat Phases 1–3 with Nomic 1.5:
- Matryoshka truncation: 768d → 384d (already proven in literature)
- Latency: Expected <10ms (10x faster than Jina)
- RRF fusion: Apply stronger keyword weight (0.7 × FTS5 + 0.3 × vector) to compensate for lower code quality

### Acceptance Criteria
- Nomic 1.5 NDCG@10 + RRF > 70%
- Latency P50 < 10ms

If Nomic also fails: escalate to CodeXEmbed 2B evaluation or API fallback (Voyage Code 3).

---

## Timeline & Resources

| Phase | Duration | Hardware | Effort | Blocker? |
|-------|----------|----------|--------|----------|
| Phase 1: Matryoshka | 1–2 days | M1/M2 + CodeSearchNet dataset | 4–8h | Yes |
| Phase 2: Latency | 1 day | M1, M2, M3 | 4–6h | Yes |
| Phase 3: RRF Fusion | 2 days | M1 + indexing setup | 8–12h | Yes |
| Phase 4: Fallback (if needed) | 1 day | M1 | 4–6h | No |

**Total:** 4–7 days for full validation (dependent on Phase 1–3 outcomes).

---

## Deployment Gate

**GO:** All Phase 1–3 acceptance criteria passed
- Deploy Jina 1.5B + MLX server
- Configure Matryoshka truncation to 384d
- Monitor production NDCG weekly

**NO-GO:** Phase 1 or 2 fails
- Switch to Phase 4 (Nomic 1.5 evaluation)
- If Nomic fails: revisit architecture (increase FAISS dims, explore CodeXEmbed 2B, or API fallback)

---

## Success Definition

Tessera embedding system is **production-ready** when:
1. Matryoshka 384d truncation achieves > 95% quality retention
2. Inference latency P95 < 100ms on M1
3. RRF fusion (FTS5 + 384d) achieves NDCG@10 > 75%
4. Memory footprint < 3GB on M1 (with INT8 quantization)
5. All tests pass on Python, TypeScript, PHP, Swift codebases

---

## Appendix: Test Code Skeleton

### Phase 1 Validation
```python
from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained("jinaai/jina-code-embeddings-1.5b",
                                 trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-code-embeddings-1.5b")

# Full dimensionality
inputs = tokenizer(["def hello(): pass"], return_tensors="pt", padding=True)
embeddings_full = model(**inputs).pooler_output  # 1536d

# Truncated (384d)
embeddings_truncated = embeddings_full[:, :384]

# Compare NDCG on CodeSearchNet queries
# (detailed benchmark code omitted; use MTEB evaluation framework)
```

### Phase 2 Latency Measurement
```python
import time
import mlx.core as mx

# MLX inference
model = mx.loading.load_model("jinaai/jina-code-embeddings-1.5b")

texts = ["def hello(): pass"] * 32
latencies = []

for _ in range(10):
    start = time.time()
    embeddings = model.encode(texts)
    latencies.append((time.time() - start) / len(texts))

print(f"Mean latency: {sum(latencies) / len(latencies):.2f}ms")
print(f"P95: {sorted(latencies)[int(0.95 * len(latencies))]:.2f}ms")
```

### Phase 3 RRF Fusion
```python
from scipy.stats import rankdata
import numpy as np

# FTS5 results (ranked by relevance score)
fts5_ranks = [1, 3, 5, 10, 15]  # positions of relevant docs

# Vector search results (ranked by cosine similarity)
vector_ranks = [2, 4, 6, 8, 12]

# RRF fusion: 1/rank for each, combine, re-rank
rrf_scores = {}
for rank in fts5_ranks:
    rrf_scores[rank] = rrf_scores.get(rank, 0) + 1/rank

for rank in vector_ranks:
    rrf_scores[rank] = rrf_scores.get(rank, 0) + 1/rank

fused_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
print(f"RRF ranking: {fused_ranking}")
```

---

## References
- Phase 1: MTEB CodeSearchNet benchmark ([github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb))
- Phase 2: Apple MLX documentation & hardware specs
- Phase 3: Tessera Phase 5 research (RRF validation)
- Fallback: Nomic Embed Text V1.5 docs
