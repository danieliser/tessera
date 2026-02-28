"""Query interface: keyword search, semantic search, and reciprocal rank fusion.

Phase 1: Search API design and result merging.
Will implement:
  - Keyword search (SQLite FTS5 against symbols and references)
  - Semantic search (LanceDB vector queries)
  - Reciprocal rank fusion (RRF) merging keyword + semantic results
  - Scope-filtered queries (respect project/collection/global scope)
  - Pagination and result ranking

CTO Condition C5: Phase 1 latency gate criteria:
  - Keyword search <20ms
  - Semantic search <30ms
  - RRF merge <10ms
  - Total <100ms p95 (validated via pytest benchmarks)
"""

import json
import logging
from typing import Optional
import numpy as np
import faiss

from .graph import ProjectGraph, ppr_to_ranked_list

logger = logging.getLogger(__name__)

# Over-fetch multiplier for source_type post-filtering.
# When source_type filter is active, fetch this many times the limit from FAISS,
# then post-filter. 3x validated as sufficient for most corpora; increase to 5x
# if >5% of queries return insufficient results (CTO Condition C4).
SEMANTIC_SEARCH_OVER_FETCH_MULTIPLIER = 3


def rrf_merge(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    Combines multiple ranked lists using RRF scoring:
    score(d) = sum over sources of 1 / (k + rank(d, source))

    Args:
        ranked_lists: List of ranked result lists, each containing dicts with 'id' field
        k: RRF constant (default 60)

    Returns:
        Merged list sorted by RRF score descending, with 'rrf_score' added to each item
    """
    if not ranked_lists:
        return []

    # Build score map: {id -> sum of reciprocal ranks}
    score_map = {}
    item_map = {}  # Keep track of items for later enrichment

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            reciprocal_rank = 1.0 / (k + rank)

            if item_id not in score_map:
                score_map[item_id] = 0.0
                item_map[item_id] = item
            else:
                # Update item_map if this result has more fields
                item_map[item_id].update(item)

            score_map[item_id] += reciprocal_rank

    # Sort by RRF score descending
    sorted_results = sorted(
        score_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build output with rrf_score
    output = []
    for item_id, score in sorted_results:
        result = item_map[item_id].copy()
        result["rrf_score"] = score
        output.append(result)

    return output


def cosine_search(
    query_embedding: np.ndarray,
    chunk_ids: list[int],
    embeddings: np.ndarray,
    limit: int = 10
) -> list[dict]:
    """
    Vector similarity search using FAISS IndexFlatIP (inner product on normalized vectors).

    Args:
        query_embedding: 1D float32 array (e.g. 768 dims)
        chunk_ids: list of chunk IDs corresponding to rows in embeddings
        embeddings: 2D float32 array (N x dim)
        limit: max results

    Returns:
        list of dicts with 'id', 'score' fields, sorted by score descending
    """
    if len(embeddings) == 0:
        return []

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []

    # Normalize for cosine similarity via inner product
    query_normalized = (query_embedding / query_norm).reshape(1, -1).astype(np.float32)
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_norms[embedding_norms == 0] = 1
    normalized_embeddings = (embeddings / embedding_norms).astype(np.float32)

    # Build FAISS index and search
    d = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(normalized_embeddings))

    k = min(limit, len(chunk_ids))
    scores, indices = index.search(query_normalized, k)

    results = []
    for i in range(k):
        idx = int(indices[0][i])
        if idx >= 0:
            results.append({
                "id": chunk_ids[idx],
                "score": float(scores[0][i])
            })

    return results


def hybrid_search(
    query: str,
    query_embedding: Optional[np.ndarray],
    db,
    graph: Optional["ProjectGraph"] = None,
    limit: int = 10,
    source_type: Optional[list[str]] = None,
) -> list[dict]:
    """
    Hybrid search combining keyword (FTS5) and semantic (vector) results.

    Algorithm:
    1. Run FTS5 keyword search via db.keyword_search()
    2. If query_embedding provided: run cosine_search on db.get_all_embeddings()
    3. NEW: Personalized PageRank search (if graph provided and not sparse)
    4. Merge with RRF
    5. Enrich results with file_path, start_line, end_line from chunk_meta

    Args:
        query: Search query string
        query_embedding: Optional query embedding (1D array)
        db: ProjectDB instance with methods: keyword_search(), get_all_embeddings(), get_chunk()
        graph: Optional ProjectGraph for PPR-based ranking
        limit: Max results to return
        source_type: Optional list of source types to filter by (e.g., ['code', 'markdown'])

    Returns:
        list of SearchResult dicts with:
            id, file_path, start_line, end_line, content, score, rank_sources,
            source_type, trusted, section_heading, key_path, page_number, parent_section, graph_version
    """
    ranked_lists = []
    ppr_results = []

    # 1. Keyword search (SQL-level filtering when source_type is active)
    try:
        keyword_results = db.keyword_search(query, limit=limit, source_type=source_type)
        if keyword_results:
            ranked_lists.append(keyword_results)
    except Exception:
        keyword_results = []

    # 2. Semantic search
    semantic_results = []
    if query_embedding is not None:
        try:
            all_embeddings = db.get_all_embeddings()
            if all_embeddings and len(all_embeddings) > 0:
                chunk_ids, embedding_vectors = all_embeddings
                fetch_limit = limit * SEMANTIC_SEARCH_OVER_FETCH_MULTIPLIER if source_type else limit
                semantic_results = cosine_search(
                    query_embedding,
                    chunk_ids,
                    embedding_vectors,
                    limit=fetch_limit
                )
                if source_type and semantic_results:
                    # Post-filter by source_type using chunk_meta lookup
                    filtered = []
                    for result in semantic_results:
                        chunk = db.get_chunk(result["id"])
                        if chunk and (chunk.get("source_type") or "code") in source_type:
                            filtered.append(result)
                    semantic_results = filtered[:limit]
                if semantic_results:
                    ranked_lists.append(semantic_results)
        except Exception:
            semantic_results = []

    # 3. NEW: PPR search (graph-aware ranking)
    if graph and not graph.is_sparse_fallback():
        try:
            # Identify seed symbols from keyword/semantic results
            seed_symbol_ids = set()

            # Extract symbol IDs from keyword results
            for result in keyword_results:
                chunk = db.get_chunk(result["id"])
                if chunk and chunk.get("symbol_ids"):
                    try:
                        sids = json.loads(chunk["symbol_ids"])
                        seed_symbol_ids.update(sids)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Extract symbol IDs from semantic results
            if semantic_results:
                for result in semantic_results:
                    chunk = db.get_chunk(result["id"])
                    if chunk and chunk.get("symbol_ids"):
                        try:
                            sids = json.loads(chunk["symbol_ids"])
                            seed_symbol_ids.update(sids)
                        except (json.JSONDecodeError, TypeError):
                            pass

            if seed_symbol_ids:
                # Compute PPR from seeds
                ppr_scores = graph.personalized_pagerank(list(seed_symbol_ids))

                # Map PPR scores back to chunk IDs via symbol→chunk mapping
                symbol_to_chunks = db.get_symbol_to_chunks_mapping()
                ppr_chunk_scores = {}
                for symbol_id, ppr_score in ppr_scores.items():
                    for chunk_id in symbol_to_chunks.get(symbol_id, []):
                        # Use max() to prevent dilution from low-scoring symbols
                        ppr_chunk_scores[chunk_id] = max(
                            ppr_chunk_scores.get(chunk_id, 0), ppr_score
                        )

                # Convert to ranked list format
                ppr_results = ppr_to_ranked_list(ppr_chunk_scores)
                ppr_results = ppr_results[:limit]

                if ppr_results:
                    ranked_lists.append(ppr_results)
        except Exception as e:
            logger.warning(f"PPR computation failed, falling back to 2-way RRF: {e}")

    # 4. Merge with RRF
    if not ranked_lists:
        return []

    merged = rrf_merge(ranked_lists)

    # 5. Enrich with chunk metadata
    results = []
    rank_sources = ["keyword", "semantic"]
    if ppr_results:
        rank_sources.append("graph")

    for item in merged[:limit]:
        chunk_id = item["id"]
        try:
            meta = db.get_chunk(chunk_id)
            chunk_source_type = meta.get("source_type", "code")
            enriched = {
                "id": chunk_id,
                "file_path": meta.get("file_path", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "content": meta.get("content", ""),
                "score": item.get("rrf_score", item.get("score", 0.0)),
                "rank_sources": rank_sources,
                "source_type": chunk_source_type,
                "trusted": chunk_source_type == "code",
                "section_heading": meta.get("section_heading", ""),
                "key_path": meta.get("key_path", ""),
                "page_number": meta.get("page_number"),
                "parent_section": meta.get("parent_section", ""),
                "graph_version": graph.loaded_at if graph else None,
            }
        except Exception:
            enriched = {
                "id": chunk_id,
                "file_path": "",
                "start_line": 0,
                "end_line": 0,
                "content": "",
                "score": item.get("rrf_score", item.get("score", 0.0)),
                "rank_sources": rank_sources,
                "source_type": "code",
                "trusted": True,
                "section_heading": "",
                "key_path": "",
                "page_number": None,
                "parent_section": "",
                "graph_version": graph.loaded_at if graph else None,
            }

        results.append(enriched)

    return results


def doc_search(
    query: str,
    query_embedding: Optional[np.ndarray],
    db,
    graph: Optional["ProjectGraph"] = None,
    limit: int = 10,
    formats: Optional[list[str]] = None,
) -> list[dict]:
    """Search non-code documents only.

    Convenience wrapper for hybrid_search with source_type filter
    set to document types only. Note: graph parameter is accepted for
    API compatibility but not used (PPR doesn't apply to documents).

    Args:
        query: Search query string
        query_embedding: Optional query embedding
        db: ProjectDB instance
        graph: Optional ProjectGraph (ignored for doc_search)
        limit: Max results
        formats: Document formats to search. Defaults to all document types.

    Returns:
        List of document search results
    """
    if formats is None:
        formats = [
            'markdown', 'pdf', 'yaml', 'json',
            'html', 'xml', 'text',
            'txt', 'rst', 'csv', 'tsv', 'log',
            'ini', 'cfg', 'toml', 'conf',
        ]
    return hybrid_search(query, query_embedding, db, graph=None, limit=limit, source_type=formats)


if __name__ == "__main__":
    # Test RRF merge with mock data
    print("=== RRF Merge Test ===")

    ranked_list_1 = [
        {"id": 1, "score": 0.9},
        {"id": 2, "score": 0.8},
        {"id": 3, "score": 0.7},
    ]

    ranked_list_2 = [
        {"id": 2, "score": 0.95},
        {"id": 1, "score": 0.85},
        {"id": 4, "score": 0.75},
    ]

    merged = rrf_merge([ranked_list_1, ranked_list_2], k=60)
    print("Merged results:")
    for item in merged:
        print(f"  ID: {item['id']}, RRF Score: {item['rrf_score']:.4f}")

    # Test cosine search with random vectors
    print("\n=== Cosine Search Test ===")

    query_embedding = np.random.randn(768).astype(np.float32)
    chunk_ids = [101, 102, 103, 104, 105]
    embeddings = np.random.randn(5, 768).astype(np.float32)

    results = cosine_search(query_embedding, chunk_ids, embeddings, limit=3)
    print("Cosine search results:")
    for result in results:
        print(f"  Chunk ID: {result['id']}, Similarity: {result['score']:.4f}")

    # Test hybrid search would require a mock db
    print("\n=== Hybrid Search (requires db) ===")
    print("Skipped (requires ProjectDB instance)")
