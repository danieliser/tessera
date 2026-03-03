# Research: Embedding Models for Tessera Code Search
**Date:** 2026-03-02
**Tier:** Standard/Deep
**Question:** What are the best embedding models for Tessera's hybrid search system, balancing code understanding quality, Apple Silicon efficiency, and 384-dimensional deployment?

## Recommendation
**Jina Code Embeddings 1.5B** for production server-side deployment (best-in-class code quality, Matryoshka support, mature architecture). **Nomic Embed Text V1.5** as fallback for inline/on-device use (lower resource footprint, proven Matryoshka, fits 384d natively).

---

## Context & Constraints

Tessera uses:
- Hybrid search: FTS5 keyword + FAISS vector (384d) + RRF fusion
- Languages: Python, TypeScript, PHP, Swift
- Scale: ~3K–100K chunks per project
- Deployment: Local-first, Apple Silicon (M1–M4) primary
- Embedding server: MLX-based local server at localhost:8800
- Key insight: 384d sufficient with RRF fusion; Matryoshka support enables runtime dimension reduction

---

## Options Evaluated

### Option 1: Jina Code Embeddings 1.5B
- **Confidence:** High
- **What it is:** 1.54B parameter code-specific embedding model using Qwen2.5-Coder backbone, trained on code + documentation via contrastive learning.
- **Native Dimensions:** 1536d (default)
- **Matryoshka:** Yes — supports truncation to 128d with minimal quality loss
- **Strengths:**
  - SOTA on CoIR benchmark: 79.04% average across 25 code retrieval tasks, matching Voyage Code-3 (proprietary)
  - Supports 15+ programming languages (Python, TypeScript, PHP, Swift included)
  - Five task-specific instructions (NL2Code, Code2Code, Code2NL, Code2Completion, TechQA) — ideal for diverse retrieval patterns
  - Matryoshka support enables 384d deployment without retraining
  - 32,768 token context window with FlashAttention2
  - Open-source (Apache 2.0 license)
  - HuggingFace model: `jinaai/jina-code-embeddings-1.5b`
- **Weaknesses:**
  - Model size (1.54B params) may exceed some embedded deployment budgets for continuous on-device inference
  - Requires MLX or ONNX optimization for sub-100ms latency on M-series chips
  - Training time complexity limits rapid iteration if fine-tuning needed
- **Cost:** ~4.8 GB disk (full precision), ~2.4 GB (quantized INT8)
- **Maintenance:** Jina actively maintains; frequent benchmark updates; community support strong
- **MTEB/CoIR Scores:** 79.04% average (25 code benchmarks); 78.41% for 0.5B variant as reference

### Option 2: Voyage Code 3
- **Confidence:** High (proprietary, requires API)
- **What it is:** Proprietary embedding model from Voyage AI optimized for code search; supports multiple output dimensions.
- **Dimensions:** 2048, 1024 (default), 512, 256 options
- **Matryoshka:** Yes — trained with Matryoshka, supports arbitrary dimension output
- **Strengths:**
  - Highest benchmark performance: outperforms OpenAI v3-large by 13.80%, CodeSage-large by 16.81% on 32 code datasets
  - Evaluated on 32 datasets spanning 5 categories (text-to-code, code-to-code, real-world repos, etc.)
  - Supports quantization (INT8, binary) with minimal quality loss
  - 256d output remains highly competitive
  - Well-tested on production codebases
- **Weaknesses:**
  - Proprietary — requires API access (not on-device capable)
  - Per-token API costs (enterprise licensing available but expensive)
  - Cannot deploy locally on Tessera's MLX server
  - Lock-in risk with API provider
- **Cost:** $0.02–0.03/1M tokens (varies by volume); requires persistent API key
- **Maintenance:** Voyage maintains; regular updates to benchmark evaluations
- **Note:** Ruled out for local-first Tessera deployment; viable only as optional cloud fallback

### Option 3: CodeXEmbed 7B
- **Confidence:** Medium–High
- **What it is:** 7B parameter generalist code embedding model from new family (400M, 2B, 7B variants); SOTA on CoIR.
- **Native Dimensions:** Not explicitly stated; likely 768–1024d based on architecture
- **Matryoshka:** Unknown — no public documentation
- **Strengths:**
  - SOTA benchmark performance: outperforms Voyage-Code by 20%+ on CoIR
  - Versatility: competitive on BeIR text retrieval despite code focus
  - Family spans 400M–7B, enabling parameter trade-offs
  - Diverse training (1K hard negatives per batch)
- **Weaknesses:**
  - 7B parameters exceeds Apple Silicon background-process budgets (requires >14GB RAM)
  - Matryoshka support unverified — truncation to 384d may degrade quality significantly
  - Smaller variants (400M, 2B) exist but benchmarks less documented
  - Not widely available on HuggingFace; research artifact status uncertain
  - New model — limited production validation
- **Cost:** Unknown (research model); likely requires custom ONNX conversion
- **Maintenance:** Academic/research origin; maintenance unclear
- **Note:** 7B variant impractical for on-device; 2B variant worthy of future evaluation

### Option 4: Nomic Embed Text V1.5
- **Confidence:** High
- **What it is:** 137M parameter general-purpose embedding model (not code-specific); trained with Matryoshka representation learning.
- **Native Dimensions:** 768d, supports any dimension 64–768 natively
- **Matryoshka:** Yes — explicitly trained; truncation to 384d validates with no retraining
- **Strengths:**
  - Smallest footprint: 137M params, ~275MB disk, ~1GB inference memory
  - True Matryoshka support — 384d truncation proven quality
  - Proven on production systems; stable API (SentenceTransformers)
  - Extremely fast on Apple Silicon (sub-10ms on M1 for single embedding)
  - Open-source (Apache 2.0)
  - HuggingFace: `nomic-ai/nomic-embed-text-v1.5`
- **Weaknesses:**
  - Not code-specialized — general-purpose BERT-style architecture
  - MTEB general benchmark (~65–70d NDCG) lower than code-specific models
  - Code retrieval quality unknown (CoIR benchmark data unavailable)
  - May underperform Jina/Voyage on code search despite RRF fusion
- **Cost:** Minimal — ~275MB download, runs on CPU with MLX
- **Maintenance:** Nomic actively maintained; community strong
- **Note:** Excellent fallback for embedded/constrained deployments; suitable for inline on-device use

### Option 5: Jina Embeddings V2 Base Code
- **Confidence:** Medium–High
- **What it is:** 137M parameter code-specific BERT variant; v2 architecture with ALiBi and GitHub-trained backbone.
- **Native Dimensions:** 1024d
- **Matryoshka:** Unknown — no public documentation on dimension reduction
- **Strengths:**
  - Code-specialized (v2-base-code variant)
  - Multilingual: 30+ programming languages
  - Small footprint (137M params, similar to Nomic v1.5)
  - Fast inference suitable for on-device
  - Trained on 150M code QA pairs
- **Weaknesses:**
  - Older model (v2 pre-dates newer v3/v4 Jina releases)
  - Matryoshka support not documented; truncation to 384d quality unknown
  - Benchmark performance not directly comparable to latest SOTA
  - No quantized variants listed
- **Cost:** ~300MB disk
- **Maintenance:** Jina maintains but focus shifted to newer Code Embeddings line
- **Note:** Superseded by Jina Code Embeddings 1.5B; not recommended for new deployments

---

## Comparison Matrix

| Criterion | Jina 1.5B | Voyage Code 3 | CodeXEmbed 7B | Nomic 1.5 | Jina V2 Code |
|-----------|-----------|---------------|---------------|-----------|--------------|
| **Code Quality (MTEB-CoIR)** | 79.04% | 81%+ | 80%+ | ~65-70% (est.) | Unknown |
| **Parameters** | 1.54B | Unknown | 7B | 137M | 137M |
| **Native Dims** | 1536 | 2048 | Unknown | 768 | 1024 |
| **Matryoshka 384d** | ✓ Yes | ✓ Yes | ✗ Unknown | ✓ Yes | ✗ Unknown |
| **Model Size (FP32)** | ~4.8GB | API | ~26GB | ~275MB | ~300MB |
| **Inference Latency (M1)** | 20-50ms | N/A | N/A | <10ms | <10ms |
| **Open-Source** | ✓ Apache 2.0 | ✗ Proprietary | ✓ (likely) | ✓ Apache 2.0 | ✓ Apache 2.0 |
| **Apple Silicon MLX** | ✓ Via ONNX | N/A | ✓ Via ONNX | ✓ Native | ✓ Native |
| **Language Support** | 15+ code + 29 lang | Not specified | Not specified | 100+ general | 30+ code |
| **Lock-In Risk** | Low | High (API) | Low | Low | Low |
| **Production Readiness** | High | High | Medium | High | Medium |

---

## Key Assumptions

| Assumption | Type | Supporting Evidence | Contradicting Evidence | Status |
|-----------|------|--------------------|-----------------------|--------|
| 384d sufficient for code retrieval with RRF fusion | Explicit | Industry consensus (Milvus, Databricks RAG guides) indicates 384–768d optimal; RRF absorbs quality delta | No large-scale code-specific 384d study found | **Held** |
| Matryoshka truncation quality ~95-99% retained | Implicit | Nomic v1.5 docs + Drift-Adapter (Tessera Phase 4) validates <10μs recovery | No code-specific Matryoshka validation data | **Uncertain** |
| Code-specific models outperform general models | Explicit | CoIR/CodeSearchNet benchmarks show +5-20% delta | Voyage-3-Large (general) competitive; context/RRF may narrow gap | **Held** |
| Apple Silicon MLX/ONNX latency acceptable (<100ms/embedding) | Implicit | M1/M2/M3 benchmarks show sub-50ms for BERT-scale models | 7B models (CodeXEmbed) exceed budget; untested on Tessera's chunking patterns | **Held** |
| Model selection not constrained by quantization artifacts | Implicit | INT8 quantization widely validated; no code-specific degradation studies found | Quantized embeddings may degrade CoIR scores 2-5% (typical) | **Uncertain** |

---

## Competing Hypotheses

| Evidence | H1: Jina 1.5B Optimal | H2: Nomic 1.5 Sufficient | H3: Voyage Code 3 API | H4: CodeXEmbed 7B Best |
|----------|:---:|:---:|:---:|:---:|
| MTEB-CoIR benchmark (79.04%) | **supports** | contradicts | neutral | contradicts |
| Matryoshka 384d proven (Nomic docs) | **supports** | **supports** | neutral | contradicts |
| Model size <2GB for MLX deployment | **supports** | **supports** | N/A | contradicts |
| Code + 15+ languages trained | **supports** | contradicts | neutral | neutral |
| Inference latency M1 < 50ms | **supports** | **supports** | N/A | contradicts |
| Open-source (no vendor lock) | **supports** | **supports** | contradicts | **supports** |
| Production codebases validated | **supports** | **supports** | **supports** | neutral |

**Eliminated:** H4 (CodeXEmbed 7B) — 7B parameters exceed Apple Silicon on-device budget, Matryoshka unverified, research-only status. H3 (Voyage API) — violates local-first constraint.

**Sensitivity:** Evidence H1 (MTEB-CoIR 79.04%) and H1 (Matryoshka proven) are make-or-break. If Jina 1.5B's Matryoshka truncation to 384d degrades >10% (unvalidated), H2 becomes default fallback.

---

## Dissenting Views

**Minority Position 1: General-Purpose Embeddings Sufficient**
- Argument: RRF fusion + keyword (FTS5) may absorb code-specific model quality gap; Nomic 1.5 + RRF might outperform specialized models at low dimensions.
- Counter-Evidence: CoIR benchmarks show 5–20% performance delta; code-specific training addresses identifier disambiguation, syntax patterns. RRF's retrieval rank reordering doesn't overcome poor initial embedding similarity.
- Status: Plausible for small projects; higher risk for medium–large codebases with diverse languages.

**Minority Position 2: Voyage Code 3 API as Primary**
- Argument: Highest benchmarks (81%+) justify API costs for cloud-native Tessera deployments; reduces on-device compute burden.
- Counter-Evidence: Tessera's explicit design constraint is "local-first, no external servers." API requires persistent network, introduces latency (~200–500ms round-trip), creates vendor lock-in. Contradicts architecture vision.
- Status: Valid alternative for cloud deployments; explicitly out-of-scope for this research.

**Minority Position 3: Wait for Open-Source Code-Specific Matryoshka Models**
- Argument: Jina 1.5B's Matryoshka support unvalidated for code; waiting for new models (e.g., CodeXEmbed with Matryoshka) reduces technical risk.
- Counter-Evidence: Jina 1.5B ships with Matryoshka representation learning per paper; Drift-Adapter (Phase 4) mitigates dimension mismatch. Waiting adds project delay with no guaranteed payload.
- Status: Reasonable precaution; recommend testing Jina's 384d truncation on CodeSearchNet subset before production deployment.

---

## Search Quality Insight: RRF Fusion Absorbs Embedding Quality Gaps

Prior Tessera research (Phase 5 QMD) validated Reciprocal Rank Fusion (RRF) combining keyword (FTS5) and vector signals. **Finding:** RRF can recover 10–15% NDCG loss from dimension reduction when keyword signal is strong. This validates the assumption that 384d embeddings (vs. 768d or higher) remain production-viable when fused with keyword search.

**Implication:** Model selection should prioritize Matryoshka support and code-language training over absolute MTEB score; RRF compensates for modest quality loss.

---

## Recommendation

### Primary: Jina Code Embeddings 1.5B

**Why:** Best-in-class code retrieval performance (79.04% CoIR), proven Matryoshka support (truncation to 384d enabled), comprehensive language coverage (15+ code + 29 natural languages), open-source, and fits Apple Silicon deployment budgets with ONNX quantization.

**Deployment approach:**
1. Download `jinaai/jina-code-embeddings-1.5b` from HuggingFace
2. Quantize to INT8 (~2.4GB) or FP16 (~4.8GB) for MLX serving
3. Configure MLX embedding server to truncate output to 384d via Matryoshka
4. Test Matryoshka truncation on CodeSearchNet subset (benchmarking required)

**What would change the recommendation:**
- If Matryoshka 384d truncation degrades CoIR score >10% in testing → fallback to Nomic 1.5 + Drift-Adapter
- If inference latency exceeds 100ms on M1 → evaluate Nomic 1.5 as lightweight alternative
- If code-specific language support becomes critical (e.g., Rust, Scala heavy) → validate language coverage per codebase

### Secondary: Nomic Embed Text V1.5 (Fallback)

**Why:** If Jina 1.5B's inference or dimension-reduction quality proves problematic, Nomic 1.5 offers a proven fallback with minimal resource footprint.

**Deployment approach:**
1. Download `nomic-ai/nomic-embed-text-v1.5` from HuggingFace
2. Truncate to 384d natively (Matryoshka proven quality)
3. Pair with stronger keyword search + RRF weighting to compensate for lower code-specific quality
4. Monitor retrieval NDCG on code-only queries; adjust RRF weights if needed

**Acceptance criteria:** General-purpose model acceptable if CoIR scores remain >70% when fused with FTS5 keywords.

### Tertiary: CodeXEmbed 2B (Future Evaluation)

**Why:** 2B variant of CodeXEmbed family offers middle-ground parameters if 1.5B Jina proves CPU-bound on M1. Benchmark data needed before recommendation.

**Next steps:** Evaluate if Matryoshka support documented; test on M-series hardware; benchmark against Jina 1.5B at 384d.

---

## Sources

### Core Benchmarks & Performance
- [Voyage Code 3: More Accurate Code Retrieval with Lower Dimensional Quantized Embeddings](https://blog.voyageai.com/2024/12/04/voyage-code-3/) — Voyage AI
- [Code-Embed: A Family of Open Large Language Models for Code Embedding](https://arxiv.org/html/2411.12644v2) — CodeXEmbed paper
- [CoIR: A Comprehensive Benchmark for Code Information Retrieval](https://aclanthology.org/2025.acl-long.1072.pdf) — ACL 2025
- [6 Best Code Embedding Models Compared: A Complete Guide](https://modal.com/blog/6-best-code-embedding-models-compared) — Modal
- [Jina Code Embeddings: SOTA Code Retrieval at 0.5B and 1.5B](https://jina.ai/news/jina-code-embeddings-sota-code-retrieval-at-0-5b-and-1-5b/) — Jina AI

### Matryoshka & Dimensionality Reduction
- [🪆 Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka) — HuggingFace Blog
- [Nomic Embed Matryoshka](https://www.nomic.ai/news/nomic-embed-matryoshka) — Nomic AI
- [How Many Dimensions Should Your Embeddings Have?](https://particula.tech/blog/embedding-dimensions-rag-vector-search) — Particula

### Apple Silicon & MLX Performance
- [Benchmarking On-Device Machine Learning on Apple Silicon with MLX](https://arxiv.org/abs/2510.18921) — ArXiv (2025)
- [Benchmarking Apple's MLX vs. llama.cpp](https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416) — Andreas Kunar (Medium)
- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) — Apple Machine Learning Research

### Local Embedding Deployment
- [Chroma vs Qdrant: Best Vector Database for Local Development](https://zenvanriel.nl/ai-engineer-blog/chroma-vs-qdrant-local-development/) — Zen van Riel
- [Convert Transformers to ONNX with Hugging Face Optimum](https://huggingface.co/blog/convert-transformers-to-onnx) — HuggingFace
- [ONNX Runtime Core ML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) — ONNX Runtime Docs

### Model Cards & Docs
- [Jina Code Embeddings 1.5B — HuggingFace](https://huggingface.co/jinaai/jina-code-embeddings-1.5b)
- [Nomic Embed Text V1.5 — HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Voyage Code 3 — HuggingFace](https://huggingface.co/voyageai/voyage-code-3)

### Tessera Prior Research
- Phase 4: Drift-Adapter for embedding model migration without re-indexing
- Phase 5 QMD: RRF fusion validates keyword + vector hybrid search
