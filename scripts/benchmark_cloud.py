"""Benchmark cloud embedding APIs (OpenAI, Voyage) against the PM test suite.

Tests paid cloud embedding services to compare against local fastembed models
and self-hosted gateway models.

Usage:
    uv run python scripts/benchmark_cloud.py                    # OpenAI models
    uv run python scripts/benchmark_cloud.py --voyage           # + Voyage (needs VOYAGE_API_KEY)
    uv run python scripts/benchmark_cloud.py --models openai-small openai-large

Requires:
    OPENAI_API_KEY env var for OpenAI models
    VOYAGE_API_KEY env var for Voyage models (optional)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx
import numpy as np

from tessera.db import ProjectDB
from tessera.embeddings import FastembedReranker
from tessera.indexer import IndexerPipeline
from tessera.search import SearchType, hybrid_search

PM_CORE = os.path.expanduser("~/Projects/ProContent/ProductCode/popup-maker")
PM_PRO = os.path.expanduser("~/Projects/ProContent/ProductCode/popup-maker-pro")

QUERIES = [
    ("popup rendering in WordPress footer hook", ["Popups.php"], "Frontend rendering"),
    ("register popup and popup_theme custom post types", ["PostTypes.php"], "Post type registration"),
    ("popup data model with triggers conditions and settings", ["Popup.php"], "Popup model"),
    ("singleton registry for popup trigger types", ["Triggers.php"], "Trigger registry"),
    ("exit intent mouse detection trigger", ["Triggers.php", "entry--plugin-init.php"], "Exit intent trigger"),
    ("condition callback evaluation for page targeting", ["ConditionCallbacks.php", "Conditions.php"], "Condition callbacks"),
    ("check if popup is loadable on current page", ["conditionals.php"], "Popup conditionals"),
    ("popup cookie expiration session hours days", ["Cookies.php", "PopupCookie.php"], "Cookie system"),
    ("newsletter subscription form AJAX submission handler", ["Newsletters.php", "Subscribe.php"], "Newsletter AJAX"),
    ("Gravity Forms integration for popup form submission", ["GravityForms.php"], "Gravity Forms integration"),
    ("shortcode to trigger popup on click element", ["PopupTrigger.php"], "Trigger shortcode"),
    ("pum_subscribe email subscription shortcode", ["Subscribe.php"], "Subscribe shortcode"),
    ("REST API endpoint for license validation", ["RestAPI.php", "License.php"], "License REST API"),
    ("AJAX analytics tracking popup open and conversion events", ["Analytics.php"], "Analytics tracking"),
    ("admin settings page for global plugin options", ["Settings.php"], "Admin settings"),
    ("Gutenberg block editor integration for popups", ["BlockEditor.php"], "Block editor"),
    ("Pimple dependency injection container service registration", ["Container.php", "Core.php"], "DI container"),
    ("FluentCRM add tag to contact on popup conversion", ["AddTag.php", "FluentCRM.php"], "FluentCRM tagging"),
    ("popup schedule date range recurring daily weekly", ["scheduling.php", "scheduled-actions.php"], "Scheduling"),
    ("attribution model calculation for conversion tracking", ["Attribution.php", "Conversions.php"], "Attribution service"),
]


class CloudEmbeddingClient:
    """Embedding client for cloud APIs with auth support."""

    def __init__(self, endpoint: str, model: str, api_key: str, timeout: float = 60.0,
                 dimensions: int | None = None):
        self.endpoint = endpoint
        self.model = model
        self.dimensions = dimensions
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._cache: dict[str, list[float]] = {}
        self._cache_max = 10000

    def _truncate(self, text: str, max_chars: int = 16000) -> str:
        """Truncate text to stay within token limits (~2-3 chars/token for code, 8191 token limit)."""
        return text[:max_chars] if len(text) > max_chars else text

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = 20  # conservative for cloud APIs with long texts
        all_results = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]

            results: list[list[float] | None] = []
            uncached_texts: list[str] = []
            uncached_indices: list[int] = []

            for text in batch:
                truncated = self._truncate(text)
                if truncated in self._cache:
                    results.append(self._cache[truncated])
                else:
                    uncached_indices.append(len(results))
                    results.append(None)
                    uncached_texts.append(truncated)

            if uncached_texts:
                payload: dict = {"input": uncached_texts, "model": self.model}
                if self.dimensions:
                    payload["dimensions"] = self.dimensions

                response = self._client.post(self.endpoint, json=payload)
                if response.status_code == 400 and "maximum context length" in response.text:
                    # Retry with more aggressive truncation
                    uncached_texts = [t[:8000] for t in uncached_texts]
                    payload["input"] = uncached_texts
                    response = self._client.post(self.endpoint, json=payload)
                if response.status_code != 200:
                    print(f"    API error {response.status_code}: {response.text[:200]}")
                    response.raise_for_status()
                data = response.json()

                for item in data.get("data", []):
                    idx_in_uncached = item["index"]
                    embedding = item["embedding"]
                    cache_key = uncached_texts[idx_in_uncached]
                    self._cache[cache_key] = embedding
                    if len(self._cache) > self._cache_max:
                        oldest = next(iter(self._cache))
                        del self._cache[oldest]
                    results[uncached_indices[idx_in_uncached]] = embedding

            all_results.extend(results)

        return all_results  # type: ignore[return-value]

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed query text. No instruction prefix for OpenAI/Voyage models."""
        return self.embed_single(text)

    def close(self):
        self._client.close()


# Cloud models to benchmark: (key, endpoint, model_id, label, dimensions, provider)
CLOUD_MODELS = [
    (
        "openai-small",
        "https://api.openai.com/v1/embeddings",
        "text-embedding-3-small",
        "OpenAI-3-small",
        1536,
        "openai",
        "$0.02/1M",
    ),
    (
        "openai-small-512",
        "https://api.openai.com/v1/embeddings",
        "text-embedding-3-small",
        "OpenAI-3-small-512d",
        512,
        "openai",
        "$0.02/1M",
    ),
    (
        "openai-large",
        "https://api.openai.com/v1/embeddings",
        "text-embedding-3-large",
        "OpenAI-3-large",
        3072,
        "openai",
        "$0.13/1M",
    ),
    (
        "openai-large-1024",
        "https://api.openai.com/v1/embeddings",
        "text-embedding-3-large",
        "OpenAI-3-large-1024d",
        1024,
        "openai",
        "$0.13/1M",
    ),
]

# Optional Voyage models (need VOYAGE_API_KEY)
VOYAGE_MODELS = [
    (
        "voyage-code-3",
        "https://api.voyageai.com/v1/embeddings",
        "voyage-code-3",
        "Voyage-code-3",
        1024,
        "voyage",
        "$0.18/1M",
    ),
    (
        "voyage-3-lite",
        "https://api.voyageai.com/v1/embeddings",
        "voyage-3-lite",
        "Voyage-3-lite",
        512,
        "voyage",
        "$0.02/1M",
    ),
]


def evaluate_hits(hit_files, expected_files):
    top1 = any(exp in hit_files[0] for exp in expected_files) if hit_files else False
    top3 = any(any(exp in f for exp in expected_files) for f in hit_files[:3])
    top5 = any(any(exp in f for exp in expected_files) for f in hit_files[:5])
    top10 = any(any(exp in f for exp in expected_files) for f in hit_files[:10])
    mrr = 0.0
    for rank, f in enumerate(hit_files, 1):
        if any(exp in f for exp in expected_files):
            mrr = 1.0 / rank
            break
    return {"top1": top1, "top3": top3, "top5": top5, "top10": top10, "mrr": mrr}


def run_queries(client, db_core, db_pro, reranker=None):
    results = []
    for query, expected_files, _desc in QUERIES:
        raw = client.embed_query(query)
        query_embedding = np.array(raw, dtype=np.float32)

        hits_core = hybrid_search(
            query, query_embedding, db_core, graph=None, limit=10,
            source_type=["code"], search_types=[SearchType.VEC],
            advanced_fts=False, rrf_weights=None,
        )
        hits_pro = hybrid_search(
            query, query_embedding, db_pro, graph=None, limit=10,
            source_type=["code"], search_types=[SearchType.VEC],
            advanced_fts=False, rrf_weights=None,
        )

        all_hits = hits_core + hits_pro
        all_hits.sort(key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)

        if reranker and all_hits:
            docs = [h.get("content", "")[:512] for h in all_hits[:20]]
            reranked = reranker.rerank(query, docs, top_k=10)
            if reranked and reranked[0][1] > 0:
                reordered = []
                for orig_idx, _score in reranked:
                    if orig_idx < len(all_hits):
                        reordered.append(all_hits[orig_idx])
                all_hits = reordered

        hit_files = [os.path.basename(h.get("file_path", "")) for h in all_hits]
        scores = evaluate_hits(hit_files, expected_files)
        results.append(scores)
    return results


def aggregate(results):
    n = len(results)
    return {
        "top1": sum(1 for r in results if r["top1"]) / n * 100,
        "top3": sum(1 for r in results if r["top3"]) / n * 100,
        "top5": sum(1 for r in results if r["top5"]) / n * 100,
        "top10": sum(1 for r in results if r["top10"]) / n * 100,
        "mrr": sum(r["mrr"] for r in results) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voyage", action="store_true", help="Also test Voyage models (needs VOYAGE_API_KEY)")
    parser.add_argument("--models", nargs="*", help="Specific model keys to test")
    args = parser.parse_args()

    for path, label in [(PM_CORE, "Core"), (PM_PRO, "Pro")]:
        if not os.path.isdir(path):
            print(f"ERROR: {label} not found at {path}")
            return

    openai_key = os.environ.get("OPENAI_API_KEY")
    voyage_key = os.environ.get("VOYAGE_API_KEY")

    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    models_to_test = list(CLOUD_MODELS)
    if args.voyage:
        if not voyage_key:
            print("WARNING: VOYAGE_API_KEY not set, skipping Voyage models")
        else:
            models_to_test.extend(VOYAGE_MODELS)

    if args.models:
        models_to_test = [m for m in models_to_test if m[0] in args.models]

    # Load local reranker for comparison
    print("Loading local reranker (jina-reranker-v1-tiny-en)...")
    reranker = FastembedReranker(model_name="jinaai/jina-reranker-v1-tiny-en")

    all_rows = []

    for model_key, endpoint, model_id, label, dims, provider, cost in models_to_test:
        print(f"\n{'='*80}")
        print(f"  {label} ({model_id}, {dims}d, {cost})")
        print(f"{'='*80}")

        api_key = voyage_key if provider == "voyage" else openai_key
        if not api_key:
            print(f"  SKIPPED: no API key for {provider}")
            continue

        try:
            client = CloudEmbeddingClient(
                endpoint=endpoint,
                model=model_id,
                api_key=api_key,
                dimensions=dims,
            )
            test_vec = client.embed_single("test")
            actual_dim = len(test_vec)
            print(f"  Connected: {actual_dim}d vectors")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        base_dir = tempfile.mkdtemp(prefix=f"tessera_cloud_{model_key}_")

        # Index both repos
        core_base = os.path.join(base_dir, "core")
        os.makedirs(core_base, exist_ok=True)
        ProjectDB.base_dir = core_base
        pipeline_core = IndexerPipeline(project_path=PM_CORE, embedding_client=client)
        pipeline_core.register()
        t0 = time.perf_counter()
        stats_core = pipeline_core.index_project()

        pro_base = os.path.join(base_dir, "pro")
        os.makedirs(pro_base, exist_ok=True)
        ProjectDB.base_dir = pro_base
        pipeline_pro = IndexerPipeline(project_path=PM_PRO, embedding_client=client)
        pipeline_pro.register()
        stats_pro = pipeline_pro.index_project()
        index_time = time.perf_counter() - t0
        total_chunks = stats_core.chunks_embedded + stats_pro.chunks_embedded
        print(f"  Indexed: {total_chunks} chunks in {index_time:.0f}s")

        # Open DBs
        ProjectDB.base_dir = core_base
        db_core = ProjectDB(PM_CORE)
        ProjectDB.base_dir = pro_base
        db_pro = ProjectDB(PM_PRO)

        # VEC-only
        vec_results = run_queries(client, db_core, db_pro)
        vec_agg = aggregate(vec_results)

        # VEC + local reranker
        rr_results = run_queries(client, db_core, db_pro, reranker)
        rr_agg = aggregate(rr_results)

        row = {
            "key": model_key, "label": label, "dim": actual_dim,
            "cost": cost, "provider": provider,
            "index_s": index_time,
            "vec_mrr": vec_agg["mrr"], "vec_top1": vec_agg["top1"],
            "vec_top3": vec_agg["top3"], "vec_top10": vec_agg["top10"],
            "rr_mrr": rr_agg["mrr"], "rr_top1": rr_agg["top1"],
            "rr_top3": rr_agg["top3"], "rr_top10": rr_agg["top10"],
        }
        all_rows.append(row)

        print(f"  VEC-only:  MRR={vec_agg['mrr']:.3f}  Top1={vec_agg['top1']:.0f}%  Top3={vec_agg['top3']:.0f}%  Top10={vec_agg['top10']:.0f}%")
        print(f"  +rerank:   MRR={rr_agg['mrr']:.3f}  Top1={rr_agg['top1']:.0f}%  Top3={rr_agg['top3']:.0f}%  Top10={rr_agg['top10']:.0f}%")

        db_core.close()
        db_pro.close()
        client.close()
        ProjectDB.base_dir = None
        shutil.rmtree(base_dir, ignore_errors=True)

    # Summary
    print(f"\n\n{'='*120}")
    print("CLOUD EMBEDDING MODEL COMPARISON (sorted by VEC+rerank MRR)")
    print(f"{'='*120}")
    print(f"{'Model':<22} {'Dim':>5} {'Cost':>10} {'Index':>7} | {'VEC MRR':>8} {'Top1':>5} {'Top3':>5} {'T10':>5} | {'+ Rerank':>8} {'Top1':>5} {'Top3':>5} {'T10':>5}")
    print("-" * 120)

    for row in sorted(all_rows, key=lambda r: r["rr_mrr"], reverse=True):
        print(f"{row['label']:<22} {row['dim']:>4}d {row['cost']:>10} {row['index_s']:>5.0f}s "
              f"| {row['vec_mrr']:>8.3f} {row['vec_top1']:>4.0f}% {row['vec_top3']:>4.0f}% {row['vec_top10']:>4.0f}% "
              f"| {row['rr_mrr']:>8.3f} {row['rr_top1']:>4.0f}% {row['rr_top3']:>4.0f}% {row['rr_top10']:>4.0f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
