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

import csv
import hashlib
import io
import json
import logging
import re
from enum import StrEnum
from typing import Optional

import faiss
import numpy as np

from .graph import ProjectGraph, ppr_to_ranked_list

logger = logging.getLogger(__name__)


class SearchType(StrEnum):
    """Controls which search lists fire in hybrid_search."""
    LEX = "lex"     # FTS5 keyword only
    VEC = "vec"     # FAISS vector (uses embed_query with retrieval prefix)
    HYDE = "hyde"    # FAISS vector (uses embed_single, no retrieval prefix)


# Regex: optional comma-separated prefixes followed by colon, then the query
_STRUCTURED_QUERY_RE = re.compile(
    r"^((?:lex|vec|hyde)(?:\s*,\s*(?:lex|vec|hyde))*)\s*:\s*(.+)$",
    re.IGNORECASE,
)


def parse_structured_query(query: str) -> tuple[str, list[SearchType]]:
    """Parse structured query prefix from query string.

    Formats:
        'lex:actual query'       → keyword only
        'vec:actual query'       → semantic only (embed_query)
        'hyde:actual query'      → semantic only (embed_single, no prefix)
        'lex,vec:actual query'   → keyword + semantic
        'actual query'           → [LEX, VEC] (backward compat)

    Returns:
        (clean_query, search_types)
    """
    m = _STRUCTURED_QUERY_RE.match(query.strip())
    if not m:
        return query, [SearchType.LEX, SearchType.VEC]

    prefix_str = m.group(1).lower()
    clean_query = m.group(2).strip()
    types = [SearchType(t.strip()) for t in prefix_str.split(",")]
    return clean_query, list(dict.fromkeys(types))  # dedupe preserving order


def normalize_bm25_score(raw_score: float) -> float:
    """Normalize FTS5 BM25 score to [0, 1] range.

    FTS5 bm25() returns negative values where more negative = better match.
    This normalizes via: norm = -raw / (1 + abs(raw))

    Args:
        raw_score: Raw FTS5 bm25() score (typically negative)

    Returns:
        Normalized score in [0, 1] where 1 = best match
    """
    normalized = -raw_score / (1.0 + abs(raw_score))
    return max(0.0, min(1.0, normalized))


# Over-fetch multiplier for source_type post-filtering.
# When source_type filter is active, fetch this many times the limit from FAISS,
# then post-filter. 3x validated as sufficient for most corpora; increase to 5x
# if >5% of queries return insufficient results (CTO Condition C4).
SEMANTIC_SEARCH_OVER_FETCH_MULTIPLIER = 3

# BM25 short-circuit thresholds: skip semantic/PPR when keyword match is very confident
BM25_STRONG_SIGNAL_THRESHOLD = 0.85
BM25_STRONG_SIGNAL_GAP = 0.15

# Per-list weights for weighted RRF fusion
DEFAULT_RRF_WEIGHTS = {
    "keyword": 1.5,
    "semantic": 1.0,
    "graph": 0.8,
}


def bm25_strong_signal_check(
    keyword_results: list[dict],
    threshold: float = BM25_STRONG_SIGNAL_THRESHOLD,
    gap: float = BM25_STRONG_SIGNAL_GAP,
) -> bool:
    """Check if BM25 results are confident enough to skip semantic/PPR search.

    Returns True when the top result's normalized score >= threshold AND
    the gap between #1 and #2 >= gap, indicating a high-confidence keyword
    match where expensive operations can be skipped.
    """
    if len(keyword_results) < 1:
        return False

    top_score = normalize_bm25_score(keyword_results[0].get("score", 0))
    if top_score < threshold:
        return False

    if len(keyword_results) < 2:
        return True

    second_score = normalize_bm25_score(keyword_results[1].get("score", 0))
    return (top_score - second_score) >= gap


def _find_best_match_line(lines: list[str], query: str) -> int:
    """Find the line with the highest keyword overlap with the query."""
    _tokenize = re.compile(r"[a-z0-9]+", re.IGNORECASE).findall
    query_terms = set(t.lower() for t in _tokenize(query))

    best_line_idx = 0
    best_overlap = 0
    for i, line in enumerate(lines):
        overlap = len(query_terms & set(t.lower() for t in _tokenize(line)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_line_idx = i
    return best_line_idx


def _render_line(abs_line: int, text: str, line_width: int) -> str:
    """Render a single line with its line number."""
    return f"{abs_line:>{line_width}} | {text}"


def _render_collapse(hidden_lines: int, indent: str, line_width: int) -> str:
    """Render a collapse marker showing how many lines are hidden."""
    padding = " " * (line_width + 3)  # width + " | "
    return f"{padding}{indent}...  ({hidden_lines} lines)"


def extract_snippet(
    chunk_content: str,
    query: str,
    context_lines: int = 3,
    ancestors: list[dict] | None = None,
    chunk_start_line: int = 0,
    mode: str = "lines",
    max_depth: int | None = None,
) -> dict:
    """Extract best-matching snippet from chunk content with optional ancestry.

    In 'lines' mode (default), shows the match window surrounded by a collapsed
    nesting skeleton — each ancestor's definition line with collapsed markers
    between. In 'full' mode, expands all ancestor content without collapsing.

    Args:
        chunk_content: Full chunk text
        query: Search query (for keyword overlap scoring)
        context_lines: Number of lines before/after best match
        ancestors: List of symbol dicts with line, end_line, signature, name, kind
                   ordered outermost first. If None, returns flat snippet.
        chunk_start_line: Absolute file line where the chunk starts (1-based)
        mode: 'lines' (collapsed ancestry) or 'full' (expand everything)
        max_depth: Max ancestor levels to show (None = all)

    Returns:
        Dict with snippet, line positions, best_match_line, and ancestors metadata
    """
    empty = {
        "snippet": "", "snippet_start_line": 0,
        "snippet_end_line": 0, "best_match_line": 0, "ancestors": [],
    }
    if not chunk_content:
        return empty

    lines = chunk_content.split("\n")
    best_line_idx = _find_best_match_line(lines, query)

    # Match window (chunk-relative indices)
    win_start = max(0, best_line_idx - context_lines)
    win_end = min(len(lines), best_line_idx + context_lines + 1)

    # Absolute line numbers
    abs_match = chunk_start_line + best_line_idx
    abs_win_start = chunk_start_line + win_start
    abs_win_end = chunk_start_line + win_end - 1

    # No ancestors — return flat snippet with line numbers
    if not ancestors:
        line_width = len(str(abs_win_end))
        rendered = []
        for i in range(win_start, win_end):
            abs_ln = chunk_start_line + i
            rendered.append(_render_line(abs_ln, lines[i], line_width))
        return {
            "snippet": "\n".join(rendered),
            "snippet_start_line": abs_win_start,
            "snippet_end_line": abs_win_end,
            "best_match_line": abs_match,
            "ancestors": [],
        }

    # Trim ancestors to max_depth
    if max_depth is not None and len(ancestors) > max_depth:
        ancestors = ancestors[-max_depth:]

    # Build ancestor metadata for return value
    ancestor_meta = [
        {"name": a.get("name", ""), "kind": a.get("kind", ""),
         "line": a.get("line", 0), "end_line": a.get("end_line", 0)}
        for a in ancestors
    ]

    if mode == "full":
        # Full mode: show everything from outermost ancestor start to end
        outermost = ancestors[0]
        full_start = outermost.get("line", abs_win_start)
        full_end = outermost.get("end_line", abs_win_end)
        # We can only render lines within the chunk
        render_start = max(0, full_start - chunk_start_line)
        render_end = min(len(lines), full_end - chunk_start_line + 1)
        line_width = len(str(full_end))
        rendered = []
        for i in range(render_start, render_end):
            abs_ln = chunk_start_line + i
            rendered.append(_render_line(abs_ln, lines[i], line_width))
        return {
            "snippet": "\n".join(rendered),
            "snippet_start_line": full_start,
            "snippet_end_line": full_end,
            "best_match_line": abs_match,
            "ancestors": ancestor_meta,
        }

    # Lines mode: collapsed ancestry skeleton
    # Determine line width from the largest line number we'll show
    all_line_nums = [a.get("line", 0) for a in ancestors]
    all_line_nums.extend(range(abs_win_start, abs_win_end + 1))
    innermost_end = ancestors[-1].get("end_line", 0) if ancestors else 0
    if innermost_end:
        all_line_nums.append(innermost_end)
    line_width = len(str(max(all_line_nums))) if all_line_nums else 4

    rendered = []
    cursor = 0  # tracks the next absolute line we expect to show

    # Only render ancestors whose definition line is BEFORE the match window
    for anc in ancestors:
        anc_line = anc.get("line", 0)
        if anc_line >= abs_win_start:
            break  # this ancestor is inside the match window — skip rendering

        # Use stored signature as-is (captured from source at index time)
        sig = anc.get("signature", "") or anc.get("name", "")
        if "\n" in sig:
            sig = sig.split("\n")[0].rstrip()

        def_text = sig
        # Derive indent from the signature's leading whitespace
        indent = " " * (len(sig) - len(sig.lstrip())) if sig else ""

        if cursor > 0 and anc_line > cursor:
            hidden = anc_line - cursor
            if hidden > 0:
                rendered.append(_render_collapse(hidden, indent, line_width))

        rendered.append(_render_line(anc_line, def_text, line_width))
        cursor = anc_line + 1

    # Collapse between last rendered ancestor and match window
    if cursor > 0 and abs_win_start > cursor:
        hidden = abs_win_start - cursor
        if hidden > 0:
            depth_indent = "    " * len(ancestors)
            rendered.append(_render_collapse(hidden, depth_indent, line_width))

    # Render the match window
    for i in range(win_start, win_end):
        abs_ln = chunk_start_line + i
        rendered.append(_render_line(abs_ln, lines[i], line_width))
    cursor = abs_win_end + 1

    # Collapse after match window to innermost ancestor's end
    if innermost_end and cursor <= innermost_end:
        hidden = innermost_end - cursor + 1
        if hidden > 0:
            depth_indent = "    " * len(ancestors)
            rendered.append(_render_collapse(hidden, depth_indent, line_width))

    return {
        "snippet": "\n".join(rendered),
        "snippet_start_line": abs_win_start,
        "snippet_end_line": abs_win_end,
        "best_match_line": abs_match,
        "ancestors": ancestor_meta,
    }


def generate_docid(content: str) -> str:
    """Generate a 6-character document ID from content hash.

    Provides a short, stable identifier for chunks. Collision probability
    is ~1 in 16M per pair (acceptable for local codebase indexes).
    """
    if not content:
        return "000000"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:6]


def enrich_with_docid(results: list[dict]) -> list[dict]:
    """Add docid field to search results that have content."""
    enriched = []
    for r in results:
        r_copy = r.copy()
        content = r_copy.get("content", "")
        if content:
            r_copy["docid"] = generate_docid(content)
        enriched.append(r_copy)
    return enriched


def format_results(
    results: list[dict],
    format: str = "json",
    fields: list[str] | None = None,
) -> str:
    """Format search results in the requested output format.

    Args:
        results: List of search result dicts
        format: Output format — "json", "csv", "markdown", "files"
        fields: Optional field filter (only include these keys)

    Returns:
        Formatted string
    """
    if fields:
        results = [{k: r.get(k) for k in fields if k in r} for r in results]

    if format == "json":
        return json.dumps(results, indent=2, default=str)

    if format == "csv":
        if not results:
            return ""
        all_keys: list[str] = []
        seen_keys: set[str] = set()
        for r in results:
            for k in r:
                if k not in seen_keys:
                    seen_keys.add(k)
                    all_keys.append(k)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: str(v) for k, v in r.items()})
        return output.getvalue()

    if format == "markdown":
        if not results:
            return "*No results found.*"
        lines = []
        for i, r in enumerate(results, 1):
            path = r.get("file_path", "unknown")
            start = r.get("start_line", "?")
            end = r.get("end_line", "?")
            score = r.get("rrf_score", r.get("score", "?"))
            snippet = r.get("snippet", r.get("content", "")[:200])
            lines.append(f"### {i}. `{path}:{start}-{end}` (score: {score})")
            lines.append(f"```\n{snippet}\n```\n")
        return "\n".join(lines)

    if format == "files":
        seen: set[str] = set()
        paths: list[str] = []
        for r in results:
            p = r.get("file_path", "")
            if p and p not in seen:
                seen.add(p)
                paths.append(p)
        return "\n".join(paths)

    return json.dumps(results, indent=2, default=str)


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


def weighted_rrf_merge(
    ranked_lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion.

    Extends standard RRF with per-list weights:
    score(d) = sum over sources of weight_i / (k + rank(d, source_i))

    Args:
        ranked_lists: List of ranked result lists, each with 'id' field
        weights: Per-list weights (e.g., [1.5, 1.0, 0.8] for FTS/FAISS/PPR).
                 If None, all lists weighted equally at 1.0.
        k: RRF constant (default 60)

    Returns:
        Merged list sorted by weighted RRF score, with 'rrf_score' and 'rrf_rank'
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match ranked_lists length ({len(ranked_lists)})"
        )

    score_map: dict = {}
    item_map: dict = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        w = weights[list_idx]
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            reciprocal_rank = w / (k + rank)

            if item_id not in score_map:
                score_map[item_id] = 0.0
                item_map[item_id] = item
            else:
                item_map[item_id].update(item)

            score_map[item_id] += reciprocal_rank

    sorted_results = sorted(
        score_map.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    output = []
    for rrf_rank, (item_id, score) in enumerate(sorted_results, start=1):
        result = item_map[item_id].copy()
        result["rrf_score"] = score
        result["rrf_rank"] = rrf_rank
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
    query_embedding: np.ndarray | None,
    db,
    graph: Optional["ProjectGraph"] = None,
    limit: int = 10,
    source_type: list[str] | None = None,
    search_types: list[SearchType] | None = None,
    advanced_fts: bool = False,
    rrf_weights: dict[str, float] | None = None,
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
        search_types: Which search lists to run. None = parse from query prefix.
            [LEX] = keyword only, [VEC] = semantic only, [HYDE] = semantic (no prefix).
        advanced_fts: If True, allow FTS5 operators (phrases, NOT, *, NEAR).

    Returns:
        list of SearchResult dicts with:
            id, file_path, start_line, end_line, content, score, rank_sources,
            source_type, trusted, section_heading, key_path, page_number, parent_section, graph_version
    """
    # Parse structured query prefix if search_types not explicitly provided
    if search_types is None:
        query, search_types = parse_structured_query(query)

    effective_weights = rrf_weights if rrf_weights else DEFAULT_RRF_WEIGHTS
    run_keyword = SearchType.LEX in search_types
    run_semantic = SearchType.VEC in search_types or SearchType.HYDE in search_types

    ranked_lists = []
    list_labels = []
    ppr_results = []

    # 1. Keyword search (SQL-level filtering when source_type is active)
    keyword_results = []
    if run_keyword:
        try:
            keyword_results = db.keyword_search(
                query, limit=limit, source_type=source_type, advanced_fts=advanced_fts
            )
            if keyword_results:
                ranked_lists.append(keyword_results)
                list_labels.append("keyword")
        except Exception:
            keyword_results = []

    # 1b. BM25 strong-signal short-circuit: skip expensive operations when
    # keyword match is very high confidence (top score >= 0.85 with >= 0.15 gap)
    if keyword_results and bm25_strong_signal_check(keyword_results):
        logger.debug("BM25 short-circuit: strong signal, skipping semantic/PPR")
        weights = [effective_weights.get("keyword", 1.5)]
        merged = weighted_rrf_merge(ranked_lists, weights=weights)
        results = []
        for item in merged[:limit]:
            chunk_id = item["id"]
            try:
                meta = db.get_chunk(chunk_id)
                file_path = meta.get("file_path", "")
                if not file_path and meta.get("file_id"):
                    file_rec = db.get_file(file_id=meta["file_id"])
                    if file_rec:
                        file_path = file_rec.get("path", "")
                chunk_source_type = meta.get("source_type", "code")
                enriched = {
                    "id": chunk_id,
                    "file_id": meta.get("file_id"),
                    "file_path": file_path,
                    "start_line": meta.get("start_line", 0),
                    "end_line": meta.get("end_line", 0),
                    "content": meta.get("content", ""),
                    "score": item.get("rrf_score", item.get("score", 0.0)),
                    "rank_sources": ["keyword"],
                    "source_type": chunk_source_type,
                    "trusted": chunk_source_type == "code",
                    "section_heading": meta.get("section_heading", ""),
                    "key_path": meta.get("key_path", ""),
                    "page_number": meta.get("page_number"),
                    "parent_section": meta.get("parent_section", ""),
                    "graph_version": graph.loaded_at if graph else None,
                    "short_circuited": True,
                }
            except Exception:
                enriched = {
                    "id": chunk_id,
                    "file_path": "",
                    "start_line": 0,
                    "end_line": 0,
                    "content": "",
                    "score": item.get("rrf_score", item.get("score", 0.0)),
                    "rank_sources": ["keyword"],
                    "source_type": "code",
                    "trusted": True,
                    "section_heading": "",
                    "key_path": "",
                    "page_number": None,
                    "parent_section": "",
                    "graph_version": graph.loaded_at if graph else None,
                    "short_circuited": True,
                }
            results.append(enriched)
        return results

    # 2. Semantic search (skipped if only LEX requested)
    semantic_results = []
    if run_semantic and query_embedding is not None:
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
                    list_labels.append("semantic")
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
                    list_labels.append("graph")
        except Exception as e:
            logger.warning(f"PPR computation failed, falling back to 2-way RRF: {e}")

    # 4. Merge with weighted RRF
    if not ranked_lists:
        return []

    weights = [effective_weights.get(label, 1.0) for label in list_labels]
    merged = weighted_rrf_merge(ranked_lists, weights=weights)

    # 5. Enrich with chunk metadata
    results = []
    rank_sources = list_labels

    for item in merged[:limit]:
        chunk_id = item["id"]
        try:
            meta = db.get_chunk(chunk_id)
            # Resolve file_path from files table (chunk_meta stores file_id)
            file_path = meta.get("file_path", "")
            if not file_path and meta.get("file_id"):
                file_rec = db.get_file(file_id=meta["file_id"])
                if file_rec:
                    file_path = file_rec.get("path", "")
            chunk_source_type = meta.get("source_type", "code")
            enriched = {
                "id": chunk_id,
                "file_id": meta.get("file_id"),
                "file_path": file_path,
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
    query_embedding: np.ndarray | None,
    db,
    graph: Optional["ProjectGraph"] = None,
    limit: int = 10,
    formats: list[str] | None = None,
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
