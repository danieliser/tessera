"""Model profiles: embedding model metadata and optimization flags.

Each profile captures the model's capabilities and the retrieval optimizations
known to help or hurt, based on empirical benchmarks (docs/experiment-journal.md).
Profiles drive hook registration — no hardcoded conditionals in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


RERANKER_JINA_TINY = "jinaai/jina-reranker-v1-tiny-en"
RERANKER_JINA_TURBO = "jinaai/jina-reranker-v1-turbo-en"
RERANKER_JINA_V3 = "jinaai/jina-reranker-v3"


@dataclass(frozen=True)
class ModelProfile:
    """Embedding model metadata and optimization flags."""

    key: str
    model_id: str
    display_name: str
    dimensions: int
    max_tokens: int
    size_mb: int
    architecture: str             # "bert", "alibi", "rope", "flash"
    provider: str                 # "fastembed", "http"

    # Optimization flags — set empirically per model from benchmarks
    scope_prefix: bool = False
    recommended_reranker: str | None = None
    hybrid_keyword_weight: float = 1.0
    max_chunk_size: int = 50000

    # Benchmark reference
    baseline_mrr: float | None = None
    best_mrr: float | None = None
    notes: str = ""


PROFILES: dict[str, ModelProfile] = {}


def _register(*profiles: ModelProfile) -> None:
    for p in profiles:
        PROFILES[p.key] = p


# ---- 512-token BERT family ----

_register(
    ModelProfile(
        key="bge-small",
        model_id="BAAI/bge-small-en-v1.5",
        display_name="BGE-small-384d",
        dimensions=384, max_tokens=512, size_mb=67,
        architecture="bert", provider="fastembed",
        scope_prefix=False,   # EXP-004: -3.9% VEC. 384d can't absorb prefix.
        recommended_reranker=RERANKER_JINA_TINY,
        baseline_mrr=0.766, best_mrr=0.800,
        notes="Default tier. Prefix hurts VEC, helps HYB slightly.",
    ),
    ModelProfile(
        key="bge-base",
        model_id="BAAI/bge-base-en-v1.5",
        display_name="BGE-base-768d",
        dimensions=768, max_tokens=512, size_mb=210,
        architecture="bert", provider="fastembed",
        scope_prefix=True,    # EXP-004: +4.7% VEC. 768d absorbs prefix well.
        recommended_reranker=RERANKER_JINA_TINY,
        baseline_mrr=0.766, best_mrr=0.802,
        notes="Scope prefix +0.036 MRR despite 512-token window.",
    ),
    ModelProfile(
        key="gte-base",
        model_id="thenlper/gte-base",
        display_name="GTE-base-768d",
        dimensions=768, max_tokens=512, size_mb=440,
        architecture="bert", provider="fastembed",
        scope_prefix=False,   # EXP-004: -20.4% VEC. Catastrophic regression.
        recommended_reranker=RERANKER_JINA_TURBO,
        baseline_mrr=0.825,
        notes="DO NOT enable prefix. Jina-turbo pairing critical.",
    ),
    ModelProfile(
        key="arctic-xs",
        model_id="Snowflake/snowflake-arctic-embed-xs",
        display_name="Arctic-XS-384d",
        dimensions=384, max_tokens=512, size_mb=90,
        architecture="bert", provider="fastembed",
        recommended_reranker=RERANKER_JINA_TINY,
        baseline_mrr=0.704,
    ),
)

# ---- Long-context models (8K+ tokens) ----

_register(
    ModelProfile(
        key="nomic-v1.5",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        display_name="Nomic-768d",
        dimensions=768, max_tokens=8192, size_mb=520,
        architecture="rope", provider="fastembed",
        scope_prefix=True,    # 8K window — prefix cost negligible. Untested.
        recommended_reranker=RERANKER_JINA_TURBO,
        baseline_mrr=0.468,   # ONNX quantized — much worse than HTTP
        notes="ONNX quantized variant poor (0.468 vs 0.854 HTTP). Prefix untested.",
    ),
    ModelProfile(
        key="jina-code",
        model_id="jinaai/jina-embeddings-v2-base-code",
        display_name="Jina-Code-768d",
        dimensions=768, max_tokens=8192, size_mb=640,
        architecture="alibi", provider="fastembed",
        scope_prefix=True,    # 8K window. Untested.
        recommended_reranker=RERANKER_JINA_TINY,
        baseline_mrr=0.716,
        notes="Code-specialized. Long context. Prefix untested.",
    ),
)

# ---- Code-specialized models ----

_register(
    ModelProfile(
        key="coderank",
        model_id="nomic-ai/CodeRankEmbed",
        display_name="CodeRankEmbed-137M",
        dimensions=768, max_tokens=8192, size_mb=270,
        architecture="bert", provider="fastembed",
        scope_prefix=True,    # Uses "Represent this query for searching relevant code:" prefix
        recommended_reranker=RERANKER_JINA_TURBO,
        notes="Code-specific 137M bi-encoder. SOTA at size on CSN (77.9 MRR). MIT license.",
    ),
)

# ---- HTTP gateway models ----

_register(
    ModelProfile(
        key="nomic-http",
        model_id="nomic-embed",
        display_name="Nomic-768d (gateway)",
        dimensions=768, max_tokens=8192, size_mb=0,
        architecture="rope", provider="http",
        scope_prefix=True,    # EXP-006: +0.071 VEC+PPR. 8K window absorbs prefix easily.
        recommended_reranker=RERANKER_JINA_TURBO,
        hybrid_keyword_weight=0.0,  # FTS5 returns 0 for NL queries against code
        baseline_mrr=0.547, best_mrr=0.662,
        notes="Best overall. Full-precision via HTTP gateway. PM20 benchmark.",
    ),
    ModelProfile(
        key="qwen3-http",
        model_id="qwen3-embed",
        display_name="Qwen3-1024d (gateway)",
        dimensions=1024, max_tokens=8192, size_mb=0,
        architecture="flash", provider="http",
        scope_prefix=True,
        recommended_reranker=RERANKER_JINA_TURBO,
        baseline_mrr=0.825,
    ),
    ModelProfile(
        key="nomic-code-http",
        model_id="nomic-embed-code",
        display_name="Nomic-Code-7B (gateway)",
        dimensions=768, max_tokens=8192, size_mb=0,
        architecture="rope", provider="http",
        scope_prefix=True,    # Uses "Represent this query for searching relevant code:"
        recommended_reranker=RERANKER_JINA_V3,
        hybrid_keyword_weight=0.0,
        notes="7B code-specific embedder via gateway. CSN: 72.3 PHP.",
    ),
    ModelProfile(
        key="coderank-http",
        model_id="code-rank-embed",
        display_name="CodeRankEmbed-137M (gateway)",
        dimensions=768, max_tokens=8192, size_mb=0,
        architecture="bert", provider="http",
        scope_prefix=True,
        recommended_reranker=RERANKER_JINA_V3,
        hybrid_keyword_weight=0.0,
        notes="137M code bi-encoder via gateway. SOTA at size on CSN (77.9).",
    ),
)


# ---- Preset tiers ----

PRESETS: dict[str, dict] = {
    "compact": {
        "description": "Minimal footprint (~200MB). Fast indexing.",
        "embedding": "bge-small",
        "reranker": RERANKER_JINA_TINY,
    },
    "balanced": {
        "description": "Best quality per MB (~340MB). Recommended.",
        "embedding": "bge-base",
        "reranker": RERANKER_JINA_TINY,
    },
    "quality": {
        "description": "Maximum local quality (~590MB).",
        "embedding": "gte-base",
        "reranker": RERANKER_JINA_TURBO,
    },
    "deep": {
        "description": "Long-context code model (~770MB).",
        "embedding": "jina-code",
        "reranker": RERANKER_JINA_TURBO,
    },
}


def get_profile(model_id: str) -> ModelProfile | None:
    """Look up a profile by key or model_id."""
    if model_id in PROFILES:
        return PROFILES[model_id]
    for profile in PROFILES.values():
        if profile.model_id == model_id:
            return profile
    return None


def get_preset(name: str) -> dict | None:
    """Look up a preset tier by name."""
    return PRESETS.get(name)


def resolve_profile(
    model_id: str | None = None,
    preset: str | None = None,
) -> ModelProfile | None:
    """Resolve a model profile from preset name or model ID.

    Precedence: explicit model_id > preset > None.
    """
    if model_id:
        return get_profile(model_id)
    if preset:
        tier = get_preset(preset)
        if tier:
            return get_profile(tier["embedding"])
    return None
