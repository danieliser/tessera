"""CodeSearchNet benchmark for Tessera — comparable to CoIR leaderboard.

Protocol (matches CoIR CSN evaluation):
  - Corpus: all func_code_string entries from Python test split (22,176 functions)
  - Queries: corresponding func_documentation_string (natural language docstrings)
  - Metric: NDCG@10 and MRR@10
  - Ground truth: each query has exactly one correct function (its own code)

Published baselines (CoIR leaderboard, NDCG@10):
  BGE-base-en-v1.5:  69.6
  GTE-base-en-v1.5:  43.35
  BGE-M3:            43.23
  BM25 (lexical):    ~35 (estimated)

Usage:
    uv run python scripts/benchmark_csn.py                        # bge-small, sample 500
    uv run python scripts/benchmark_csn.py --model bge-base       # bge-base
    uv run python scripts/benchmark_csn.py --full                 # all 22k queries (slow)
    uv run python scripts/benchmark_csn.py --provider http        # gateway model
    uv run python scripts/benchmark_csn.py --lang python --full   # full python eval
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tessera.db import ProjectDB
from tessera.embeddings import EmbeddingClient, FastembedClient
from tessera.indexer import IndexerPipeline
from tessera.search import hybrid_search, SearchType

# ---- Models ----

HTTP_MODELS = {
    "nomic": ("nomic-embed", "Nomic-768d"),
    "coderank": ("code-rank-embed", "CodeRankEmbed-137M"),
    "qwen3": ("qwen3-embed", "Qwen3-1024d"),
}

FASTEMBED_MODELS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "BGE-small-384d", 67),
    "bge-base": ("BAAI/bge-base-en-v1.5", "BGE-base-768d", 210),
    "gte-base": ("thenlper/gte-base", "GTE-base-768d", 440),
}

# Published CoIR NDCG@10 baselines for reference
COIR_BASELINES = {
    "BGE-base-en-v1.5": 69.6,
    "GTE-base-en-v1.5": 43.35,
    "BGE-M3 (567M)": 43.23,
    "E5-Base-v2 (110M)": 67.99,
    "Voyage-Code-002": 81.79,
    "CodeSage-large-v2 (1.3B)": 94.26,
}


def ndcg_at_k(ranked_ids: list[str], relevant_id: str, k: int = 10) -> float:
    """Compute NDCG@k for a single query with one relevant document."""
    for i, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id == relevant_id:
            # DCG = 1/log2(rank+1), IDCG = 1/log2(2) = 1.0
            return 1.0 / math.log2(i + 1)
    return 0.0


def mrr_at_k(ranked_ids: list[str], relevant_id: str, k: int = 10) -> float:
    """Compute MRR@k for a single query with one relevant document."""
    for i, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id == relevant_id:
            return 1.0 / i
    return 0.0


def write_functions_as_files(functions: list[dict], out_dir: str) -> None:
    """Write each function as a .py file for Tessera to index.

    File path: out_dir/{repo}/{func_path} (grouped by actual repo structure).
    Each file contains all functions from that path, appended.
    """
    # Group functions by (repo, file_path)
    from collections import defaultdict
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for fn in functions:
        repo = fn["repository_name"].replace("/", "__")
        path = fn["func_path_in_repository"]
        groups[(repo, path)].append(fn)

    for (repo, path), fns in groups.items():
        # Sanitize path
        safe_path = path.lstrip("/")
        full_path = os.path.join(out_dir, repo, safe_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "a") as f:
            for fn in fns:
                f.write(fn["whole_func_string"])
                f.write("\n\n")


def load_csn_dataset(lang: str = "python", split: str = "test", max_samples: int | None = None) -> list[dict]:
    """Load CodeSearchNet functions, filtering to those with docstrings."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"  Loading CSN {lang}/{split}...", flush=True)
    ds = load_dataset("code-search-net/code_search_net", lang, split=split)

    # Filter: must have a non-empty docstring (our query)
    rows = [
        r for r in ds
        if r["func_documentation_string"] and len(r["func_documentation_string"].strip()) > 10
        and r["whole_func_string"] and r["func_code_string"]
    ]

    if max_samples:
        # Stratified sample across repos for diversity
        from collections import defaultdict
        import random
        random.seed(42)
        by_repo: dict[str, list] = defaultdict(list)
        for r in rows:
            by_repo[r["repository_name"]].append(r)
        # Round-robin across repos until we hit max_samples
        sampled = []
        repos = list(by_repo.keys())
        random.shuffle(repos)
        i = 0
        while len(sampled) < max_samples:
            repo = repos[i % len(repos)]
            if by_repo[repo]:
                sampled.append(by_repo[repo].pop(0))
            i += 1
        rows = sampled[:max_samples]

    print(f"  {len(rows)} functions ({lang}/{split})")
    return rows


def run_csn_nochunk(
    queries: list[dict],
    client,
    model_label: str,
    model_key: str,
    lang: str,
    k: int = 10,
    corpus: list[dict] | None = None,
    reindex: bool = False,
) -> dict:
    """CoIR-protocol benchmark: embed each function as one vector, no chunking.

    Builds an in-memory FAISS index over whole_func_string embeddings.
    This matches CoIR's evaluation exactly — one vector per function.
    """
    import faiss
    import json

    index_functions = corpus if corpus is not None else queries
    cache_dir = os.path.expanduser(f"~/.tessera/benchmarks/csn/{model_key}-nochunk/{lang}")
    os.makedirs(cache_dir, exist_ok=True)
    vecs_path = os.path.join(cache_dir, "corpus_vecs.npy")
    ids_path = os.path.join(cache_dir, "corpus_ids.json")

    # --- Embed corpus (cached) ---
    if reindex or not os.path.exists(vecs_path):
        print(f"  Embedding {len(index_functions)} functions (whole, no chunking)...", flush=True)
        t0 = time.perf_counter()
        texts = [fn["whole_func_string"] for fn in index_functions]
        batch_size = 64
        all_vecs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vecs = [client.embed_single(t) for t in batch]
            all_vecs.extend(vecs)
            if (start // batch_size + 1) % 10 == 0:
                print(f"  [{start + len(batch)}/{len(texts)}] embedded...", flush=True)
        corpus_vecs = np.array(all_vecs, dtype=np.float32)
        faiss.normalize_L2(corpus_vecs)
        np.save(vecs_path, corpus_vecs)
        corpus_ids = [
            f"{fn['repository_name'].replace('/', '__')}/{fn['func_path_in_repository'].lstrip('/')}/{fn['func_name']}"
            for fn in index_functions
        ]
        with open(ids_path, "w") as f:
            json.dump(corpus_ids, f)
        print(f"  Embedded {len(corpus_vecs)} functions in {time.perf_counter()-t0:.1f}s", flush=True)
    else:
        print(f"  Corpus vectors cached ({cache_dir})", flush=True)
        corpus_vecs = np.load(vecs_path)
        with open(ids_path) as f:
            corpus_ids = json.load(f)

    # Build FAISS index
    dim = corpus_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
    index.add(corpus_vecs)

    # Run queries
    n_queries = len(queries)
    ndcg_scores, mrr_scores = [], []
    total_ms = 0.0
    print(f"  Running {n_queries} queries against {len(index_functions)}-function corpus...", flush=True)
    for i, fn in enumerate(queries):
        query = fn["func_documentation_string"].strip()
        correct_id = f"{fn['repository_name'].replace('/', '__')}/{fn['func_path_in_repository'].lstrip('/')}/{fn['func_name']}"

        raw = client.embed_query(query)
        q_vec = np.array([raw], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        t0 = time.perf_counter()
        _, indices = index.search(q_vec, k)
        total_ms += (time.perf_counter() - t0) * 1000

        ranked_ids = [corpus_ids[idx] for idx in indices[0]]
        ndcg_scores.append(ndcg_at_k(ranked_ids, correct_id, k))
        mrr_scores.append(mrr_at_k(ranked_ids, correct_id, k))

        if (i + 1) % 100 == 0:
            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) * 100
            avg_mrr = sum(mrr_scores) / len(mrr_scores) * 100
            print(f"  [{i+1}/{n_queries}] NDCG@{k}: {avg_ndcg:.1f}  MRR@{k}: {avg_mrr:.1f}  "
                  f"avg {total_ms/(i+1):.1f}ms/q", flush=True)

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) * 100
    avg_mrr = sum(mrr_scores) / len(mrr_scores) * 100
    return {
        "model": model_label,
        "n_queries": n_queries,
        "ndcg_at_10": avg_ndcg,
        "mrr_at_10": avg_mrr,
        "avg_ms": total_ms / n_queries,
        "index_time_s": 0.0,
    }


def get_corpus_dir(lang: str) -> str:
    """Persistent directory for the written .py corpus files (shared across models)."""
    return os.path.expanduser(f"~/.tessera/benchmarks/csn/corpus/{lang}")


def get_index_dir(model_key: str, lang: str) -> str:
    """Persistent directory for the Tessera index (per model)."""
    return os.path.expanduser(f"~/.tessera/benchmarks/csn/{model_key}/{lang}")


def run_csn_benchmark(
    queries: list[dict],
    client,
    model_label: str,
    model_key: str,
    lang: str,
    use_hyde: bool = False,
    k: int = 10,
    corpus: list[dict] | None = None,
    search_types: list | None = None,
    reindex: bool = False,
) -> dict:
    """Index corpus into Tessera and evaluate retrieval for queries.

    corpus: all functions to index (full test split for CoIR-comparable eval)
    queries: subset to run queries for (can be smaller sample for speed)
    Each query matches docstring → its own code in the corpus.

    Corpus files and indexes are cached persistently under ~/.tessera/benchmarks/csn/.
    Corpus files are written once (shared across models). Index is per-model.
    """
    index_functions = corpus if corpus is not None else queries
    corpus_dir = get_corpus_dir(lang)
    index_dir = get_index_dir(model_key, lang)
    # ProjectDB stores index at {base_dir}/{slug}/index.db
    corpus_slug = corpus_dir.replace(os.sep, "-")
    index_db = os.path.join(index_dir, corpus_slug, "index.db")

    # --- Write corpus files (once, shared across models) ---
    corpus_stamp = os.path.join(corpus_dir, ".written")
    if not os.path.exists(corpus_stamp):
        print(f"  Writing {len(index_functions)} functions to {corpus_dir}...", flush=True)
        os.makedirs(corpus_dir, exist_ok=True)
        write_functions_as_files(index_functions, corpus_dir)
        open(corpus_stamp, "w").close()
    else:
        print(f"  Corpus files cached ({corpus_dir})", flush=True)

    # --- Index (per model, skip if cached) ---
    os.makedirs(index_dir, exist_ok=True)
    ProjectDB.base_dir = index_dir

    index_time = 0.0
    if reindex or not os.path.exists(index_db):
        print(f"  Indexing into {index_dir}...", flush=True)
        pipeline = IndexerPipeline(
            project_path=corpus_dir,
            embedding_client=client,
        )
        pipeline.register()
        t0 = time.perf_counter()
        stats = pipeline.index_project_sync()
        index_time = time.perf_counter() - t0
        print(f"  Indexed: {stats.files_processed} files, {stats.chunks_created} chunks, "
              f"{stats.chunks_embedded} embedded in {index_time:.1f}s", flush=True)
        print(f"  Index saved to {index_db}", flush=True)
    else:
        print(f"  Index cached — skipping re-index", flush=True)

    db = ProjectDB(corpus_dir)

    # Run queries
    n_queries = len(queries)
    ndcg_scores = []
    mrr_scores = []
    total_ms = 0.0
    print(f"  Running {n_queries} queries against {len(index_functions)}-function corpus...", flush=True)
    for i, fn in enumerate(queries):
        query = fn["func_documentation_string"].strip()
        correct_file = os.path.join(
            fn["repository_name"].replace("/", "__"),
            fn["func_path_in_repository"].lstrip("/"),
        )
        correct_func = fn["func_name"]

        if use_hyde:
            raw = client.embed_single(query)
        else:
            raw = client.embed_query(query)
        query_embedding = np.array(raw, dtype=np.float32)

        t0 = time.perf_counter()
        hits = hybrid_search(
            query, query_embedding, db,
            graph=None, limit=k * 2,
            source_type=["code"],
            search_types=search_types,
            advanced_fts=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        ranked_ids = []
        for h in hits[:k]:
            hit_path = h.get("file_path", "")
            hit_content = h.get("content", "")
            if correct_file in hit_path and correct_func in hit_content:
                ranked_ids.append("correct")
            else:
                ranked_ids.append(f"wrong_{len(ranked_ids)}")

        ndcg_scores.append(ndcg_at_k(ranked_ids, "correct", k))
        mrr_scores.append(mrr_at_k(ranked_ids, "correct", k))

        if (i + 1) % 100 == 0:
            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) * 100
            avg_mrr = sum(mrr_scores) / len(mrr_scores) * 100
            print(f"  [{i+1}/{n_queries}] NDCG@{k}: {avg_ndcg:.1f}  MRR@{k}: {avg_mrr:.1f}  "
                  f"avg {total_ms/(i+1):.0f}ms/q", flush=True)

    db.close()
    ProjectDB.base_dir = None

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) * 100
    avg_mrr = sum(mrr_scores) / len(mrr_scores) * 100

    return {
        "model": model_label,
        "n_queries": n_queries,
        "ndcg_at_10": avg_ndcg,
        "mrr_at_10": avg_mrr,
        "avg_ms": total_ms / n_queries,
        "index_time_s": index_time,
    }


def print_results(result: dict, use_hyde: bool = False, vec_only: bool = False):
    mode_tag = "HYDE" if use_hyde else ("VEC" if vec_only else "HYBRID")
    print(f"\n{'=' * 80}")
    print(f"CodeSearchNet Benchmark — {result['model']} ({mode_tag})")
    print(f"{'=' * 80}")
    print(f"  Queries:      {result['n_queries']:,}")
    print(f"  NDCG@10:      {result['ndcg_at_10']:.2f}  ← primary metric (CoIR comparable)")
    print(f"  MRR@10:       {result['mrr_at_10']:.2f}")
    print(f"  Avg latency:  {result['avg_ms']:.0f}ms/query")
    print(f"  Index time:   {result['index_time_s']:.1f}s")

    print(f"\n{'─' * 80}")
    print("CoIR Leaderboard Comparison (NDCG@10 on CSN Python test):")
    our_score = result['ndcg_at_10']
    entries = [(our_score, f"  *** Tessera ({result['model']}) ***", True)]
    for name, score in COIR_BASELINES.items():
        entries.append((score, f"  {name}", False))
    entries.sort(reverse=True)
    for score, label, is_ours in entries:
        marker = " ◄" if is_ours else ""
        print(f"  {score:>6.1f}  {label}{marker}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="CodeSearchNet benchmark for Tessera")
    parser.add_argument("--provider", choices=["http", "fastembed"], default="fastembed")
    parser.add_argument("--model", default="bge-small")
    parser.add_argument("--embed-endpoint", default="http://localhost:8800/v1/embeddings")
    parser.add_argument("--lang", default="python", choices=["python", "java", "go", "javascript", "php", "ruby"])
    parser.add_argument("--full", action="store_true", help="Run all test queries (slow)")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of queries for quick run (default: 500)")
    parser.add_argument("--hyde", action="store_true", help="Use HyDE embedding (embed_single)")
    parser.add_argument("--k", type=int, default=10, help="Top-K for NDCG/MRR")
    parser.add_argument("--vec-only", action="store_true", help="VEC-only search (no FTS/hybrid)")
    parser.add_argument("--reindex", action="store_true", help="Force re-index even if cached")
    parser.add_argument("--no-chunk", action="store_true", help="CoIR protocol: embed whole functions, no chunking")
    args = parser.parse_args()

    print("=" * 80)
    print("Tessera × CodeSearchNet Benchmark")
    print("=" * 80)

    # Load dataset
    # Always index full test split (CoIR protocol: all 22k compete)
    # Only run queries on a sample for speed (--full to query all)
    print("  Loading full corpus (all test functions — needed for CoIR-comparable eval)...")
    full_corpus = load_csn_dataset(lang=args.lang, split="test", max_samples=None)
    if args.full:
        query_functions = full_corpus
    else:
        import random; random.seed(42)
        query_functions = random.sample(full_corpus, min(args.samples, len(full_corpus)))
        print(f"  Query sample: {len(query_functions)} of {len(full_corpus)} total")

    # Load embedding client
    if args.provider == "fastembed":
        model_key = args.model
        if model_key not in FASTEMBED_MODELS:
            print(f"ERROR: Unknown model '{model_key}'. Options: {', '.join(FASTEMBED_MODELS)}")
            return
        model_name, model_label, size_mb = FASTEMBED_MODELS[model_key]
        print(f"  Loading embedding: {model_label} ({model_name}, ~{size_mb}MB)", flush=True)
        client = FastembedClient(model_name=model_name)
    else:
        model_key = args.model
        if model_key not in HTTP_MODELS:
            print(f"ERROR: Unknown model '{model_key}'. Options: {', '.join(HTTP_MODELS)}")
            return
        model_name, model_label = HTTP_MODELS[model_key]
        client = EmbeddingClient(endpoint=args.embed_endpoint, model=model_name)

    # Verify embedding works
    try:
        test_vec = client.embed_single("test")
        print(f"  Embedding: {model_label} — {len(test_vec)}d vectors", flush=True)
    except Exception as e:
        print(f"  ERROR: Embedding unavailable ({e})")
        return

    # Run benchmark
    if args.no_chunk:
        result = run_csn_nochunk(
            query_functions, client, model_label,
            model_key=model_key, lang=args.lang,
            k=args.k, corpus=full_corpus,
            reindex=args.reindex,
        )
        print_results(result, vec_only=True)
    else:
        search_types = [SearchType.VEC] if args.vec_only else None
        result = run_csn_benchmark(
            query_functions, client, model_label,
            model_key=model_key, lang=args.lang,
            use_hyde=args.hyde, k=args.k,
            corpus=full_corpus,
            search_types=search_types,
            reindex=args.reindex,
        )
        print_results(result, use_hyde=args.hyde, vec_only=args.vec_only)

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
