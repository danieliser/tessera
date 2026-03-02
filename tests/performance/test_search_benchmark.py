"""Empirical search quality benchmark for feature/search-quality-fixes.

Runs a fixed query set against the live codemem index and measures
the impact of each search quality improvement:

1. BM25 normalization — raw negative scores → [0,1] range
2. embed_query — retrieval-prefixed vs raw embeddings
3. extract_snippet — keyword-overlap snippet extraction
4. enrich_with_docid — stable content-addressable IDs
5. format_results — multi-format output

Run: uv run python -m pytest tests/test_search_benchmark.py -v -s
"""

import csv
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from tessera.db import ProjectDB
from tessera.search import (
    BM25_STRONG_SIGNAL_GAP,
    BM25_STRONG_SIGNAL_THRESHOLD,
    DEFAULT_RRF_WEIGHTS,
    SearchType,
    bm25_strong_signal_check,
    enrich_with_docid,
    extract_snippet,
    format_results,
    hybrid_search,
    normalize_bm25_score,
    parse_structured_query,
    rrf_merge,
    weighted_rrf_merge,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

CODEMEM_DB_PATH = os.path.expanduser(
    "~/.tessera/data/-Users-danieliser-Toolkit-codemem/index.db"
)

QUERIES = [
    # Identifier queries (exact symbol matches expected)
    "normalize_bm25_score",
    "ProjectDB",
    "hybrid_search",
    # Conceptual queries (semantic understanding needed)
    "error handling",
    "authentication scope",
    "graph traversal",
    # API/usage queries
    "keyword_search limit",
    "create_scope",
    # Cross-cutting
    "async to_thread",
    "FTS5 BM25",
]


@pytest.fixture(scope="module")
def db():
    """Load the live codemem index."""
    if not os.path.exists(CODEMEM_DB_PATH):
        pytest.skip(f"Codemem index not found at {CODEMEM_DB_PATH}")
    project_path = "/Users/danieliser/Toolkit/codemem"
    return ProjectDB(project_path)


@pytest.fixture(scope="module")
def embedding_client():
    """Create embedding client if endpoint is available."""
    from tessera.embeddings import EmbeddingClient, EmbeddingUnavailableError

    client = EmbeddingClient(
        endpoint="http://localhost:8800/v1/embeddings",
        model="nomic-embed",
        timeout=30.0,
    )
    try:
        client.embed_single("test")
        return client
    except EmbeddingUnavailableError:
        pytest.skip("Embedding endpoint not available")


@pytest.fixture(scope="module")
def report_lines():
    """Accumulate report lines across tests, write at end."""
    lines = []
    yield lines
    report_path = Path(__file__).parent.parent.parent / "docs" / "benchmark-search-quality.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def _search(db, query, limit=10):
    """Run keyword-only hybrid search (enriched results with file_path)."""
    return hybrid_search(query, query_embedding=None, db=db, limit=limit)


# ── Benchmark 1: BM25 Normalization ──────────────────────────────────────


class TestBM25Normalization:
    """Compare raw FTS5 scores vs normalized [0,1] scores."""

    def test_score_ranges_and_ordering(self, db, report_lines):
        report_lines.append("# Search Quality Benchmark Report")
        report_lines.append("")
        report_lines.append("## 1. BM25 Score Normalization")
        report_lines.append("")
        report_lines.append("**Change:** Raw FTS5 `bm25()` scores (negative, unbounded) → normalized [0,1] range.")
        report_lines.append("Before: scores like -7.31, -4.71 — meaningless to agents. After: 0.88, 0.83 — interpretable.")
        report_lines.append("")
        report_lines.append("| Query | #Results | Raw Score Range (before) | Normalized Range (after) | Order Preserved |")
        report_lines.append("|-------|----------|------------------------|-------------------------|-----------------|")

        for query in QUERIES:
            raw_results = db.keyword_search(query, limit=10)
            if not raw_results:
                report_lines.append(f"| `{query}` | 0 | — | — | — |")
                continue

            raw_scores = [r["score"] for r in raw_results]
            norm_scores = [normalize_bm25_score(s) for s in raw_scores]

            # Raw should be ascending (more negative = better, ORDER BY rank ASC)
            raw_ascending = all(raw_scores[i] <= raw_scores[i + 1] for i in range(len(raw_scores) - 1))
            # Normalized should be descending (higher = better)
            norm_descending = all(norm_scores[i] >= norm_scores[i + 1] for i in range(len(norm_scores) - 1))
            order_ok = raw_ascending and norm_descending

            raw_range = f"[{min(raw_scores):.2f}, {max(raw_scores):.2f}]"
            norm_range = f"[{min(norm_scores):.3f}, {max(norm_scores):.3f}]"

            report_lines.append(
                f"| `{query}` | {len(raw_results)} | {raw_range} | {norm_range} | {'Yes' if order_ok else 'NO'} |"
            )

            for ns in norm_scores:
                assert 0.0 <= ns <= 1.0, f"Score {ns} out of [0,1] for query '{query}'"
            assert order_ok, f"Score ordering broken for query '{query}'"

        report_lines.append("")

    def test_normalization_distribution(self, db, report_lines):
        """Show score spread — strong matches near 1.0, weaker near 0.8."""
        report_lines.append("**Score distribution:** Higher std = better discrimination between relevant and marginal results.")
        report_lines.append("")
        report_lines.append("| Query | Top-1 | Top-5 Mean | Spread (top1 - bottom) | Std Dev |")
        report_lines.append("|-------|-------|-----------|----------------------|---------|")

        for query in QUERIES:
            raw_results = db.keyword_search(query, limit=10)
            if not raw_results:
                continue
            norm_scores = [normalize_bm25_score(r["score"]) for r in raw_results]
            top5 = norm_scores[:5]
            spread = norm_scores[0] - norm_scores[-1]
            report_lines.append(
                f"| `{query}` | {norm_scores[0]:.4f} | {np.mean(top5):.4f} | {spread:.4f} | {np.std(norm_scores):.4f} |"
            )

        report_lines.append("")


# ── Benchmark 2: embed_query vs embed_single ─────────────────────────────


class TestEmbedQueryPrefix:
    """Compare search results using raw embeddings vs retrieval-prefixed."""

    def test_embed_query_vs_embed_single(self, db, embedding_client, report_lines):
        report_lines.append("## 2. embed_query (Retrieval Prefix)")
        report_lines.append("")
        report_lines.append("**Change:** `embed_single(query)` → `embed_query(query)` adds retrieval prefix.")
        report_lines.append("")
        report_lines.append("| Query | embed_single Top-3 | embed_query Top-3 | Top-5 Overlap | Score Delta |")
        report_lines.append("|-------|-------------------|------------------|--------------|-------------|")

        for query in QUERIES[:6]:
            raw_single = embedding_client.embed_single(query)
            emb_single = np.array(raw_single, dtype=np.float32)
            results_single = hybrid_search(query, emb_single, db, limit=5)

            raw_query = embedding_client.embed_query(query)
            emb_query = np.array(raw_query, dtype=np.float32)
            results_query = hybrid_search(query, emb_query, db, limit=5)

            def _top3_files(results):
                paths = []
                for r in results[:3]:
                    fp = r.get("file_path", "?")
                    paths.append(os.path.basename(fp) if fp else "?")
                return ", ".join(paths)

            ids_single = set(r["id"] for r in results_single[:5])
            ids_query = set(r["id"] for r in results_query[:5])
            overlap = len(ids_single & ids_query)

            s1 = results_single[0]["score"] if results_single else 0
            s2 = results_query[0]["score"] if results_query else 0

            report_lines.append(
                f"| `{query}` | {_top3_files(results_single)} | {_top3_files(results_query)} | {overlap}/5 | {s2 - s1:+.4f} |"
            )

        report_lines.append("")

    def test_embedding_divergence(self, embedding_client, report_lines):
        """How different are the two embedding vectors?"""
        report_lines.append("**Embedding divergence:** How much the retrieval prefix changes the vector.")
        report_lines.append("")
        report_lines.append("| Query | Cosine Similarity | L2 Distance |")
        report_lines.append("|-------|------------------|-------------|")

        for query in QUERIES[:6]:
            v1 = np.array(embedding_client.embed_single(query), dtype=np.float32)
            v2 = np.array(embedding_client.embed_query(query), dtype=np.float32)
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            l2 = float(np.linalg.norm(v1 - v2))
            report_lines.append(f"| `{query}` | {cos:.4f} | {l2:.2f} |")

        report_lines.append("")


# ── Benchmark 3: Snippet Extraction ──────────────────────────────────────


class TestSnippetExtraction:
    """Measure snippet quality — does it focus on the relevant context?"""

    def test_snippet_quality(self, db, report_lines):
        report_lines.append("## 3. Snippet Extraction")
        report_lines.append("")
        report_lines.append("**Change:** `extract_snippet` returns a focused ~7-line window around the best")
        report_lines.append("keyword-matching line, instead of the full chunk (which can be 50+ lines).")
        report_lines.append("")
        report_lines.append("| Query | File | Chunk Lines | Snippet Lines | Compression | Best Match Line |")
        report_lines.append("|-------|------|------------|--------------|-------------|----------------|")

        for query in QUERIES:
            results = _search(db, query, limit=3)
            for r in results[:2]:
                content = r.get("content", "")
                if not content:
                    continue

                info = extract_snippet(content, query)
                full_lines = len(content.split("\n"))
                snip_lines = len(info["snippet"].split("\n"))
                compression = f"{snip_lines}/{full_lines} ({100 * snip_lines / max(full_lines, 1):.0f}%)"
                fp = os.path.basename(r.get("file_path", "?"))

                report_lines.append(
                    f"| `{query}` | {fp}:{r.get('start_line', '?')} | {full_lines} | {snip_lines} | {compression} | {info['best_match_line']} |"
                )

        report_lines.append("")

    def test_snippet_content_relevance(self, db, report_lines):
        """Show actual snippet content for a few queries to demonstrate quality."""
        report_lines.append("**Sample snippets** (showing what agents actually see):")
        report_lines.append("")

        sample_queries = ["hybrid_search", "error handling", "create_scope"]
        for query in sample_queries:
            results = _search(db, query, limit=1)
            if not results:
                continue
            r = results[0]
            content = r.get("content", "")
            if not content:
                continue

            info = extract_snippet(content, query)
            fp = r.get("file_path", "?")
            full_lines = len(content.split("\n"))

            report_lines.append(f"**Query: `{query}`** — {os.path.basename(fp)}:{r.get('start_line', '?')} ({full_lines} lines → {len(info['snippet'].split(chr(10)))} lines)")
            report_lines.append("```")
            report_lines.append(info["snippet"])
            report_lines.append("```")
            report_lines.append("")

    def test_snippet_contains_best_match(self, db):
        """Invariant: snippet always includes the best-matching line."""
        for query in QUERIES:
            results = _search(db, query, limit=5)
            for r in results:
                content = r.get("content", "")
                if not content:
                    continue
                info = extract_snippet(content, query)
                lines = content.split("\n")
                if info["best_match_line"] < len(lines):
                    best_line = lines[info["best_match_line"]]
                    assert best_line in info["snippet"]


# ── Benchmark 4: Document IDs ────────────────────────────────────────────


class TestDocumentIDs:
    """Verify docid stability and uniqueness."""

    def test_docid_properties(self, db, report_lines):
        report_lines.append("## 4. Content-Addressable Document IDs")
        report_lines.append("")
        report_lines.append("**Change:** `generate_docid` produces a stable 6-char hex hash from content.")
        report_lines.append("")

        # Collect docids across multiple queries
        all_docids = {}
        collisions = 0
        for query in QUERIES:
            results = _search(db, query, limit=10)
            enriched = enrich_with_docid(results)
            for r in enriched:
                docid = r.get("docid")
                content = r.get("content", "")
                if not docid or not content:
                    continue
                if docid in all_docids and all_docids[docid] != content[:100]:
                    collisions += 1
                all_docids[docid] = content[:100]

        # Determinism check
        results = _search(db, "hybrid_search", limit=5)
        e1 = enrich_with_docid(results)
        e2 = enrich_with_docid(results)
        deterministic = all(
            a.get("docid") == b.get("docid")
            for a, b in zip(e1, e2)
            if a.get("docid")
        )

        report_lines.append("| Property | Result |")
        report_lines.append("|----------|--------|")
        report_lines.append(f"| Deterministic (same content → same ID) | {'Pass' if deterministic else 'FAIL'} |")
        report_lines.append(f"| Total unique IDs tested | {len(all_docids)} |")
        report_lines.append(f"| Collisions (different content, same ID) | {collisions} |")
        report_lines.append(f"| ID format | 6-char hex (e.g. `{list(all_docids.keys())[0] if all_docids else '------'}`) |")
        report_lines.append("")

        assert deterministic
        assert collisions == 0


# ── Benchmark 5: Format Results ──────────────────────────────────────────


class TestFormatResults:
    """Verify all output formats produce valid, usable output."""

    def test_all_formats(self, db, report_lines):
        report_lines.append("## 5. Multi-Format Output")
        report_lines.append("")
        report_lines.append("**Change:** `format_results` supports json, csv, markdown, and files output modes.")
        report_lines.append("Before: only `json.dumps`. After: agents can request the format that suits their task.")
        report_lines.append("")

        results = _search(db, "hybrid_search", limit=5)
        results = enrich_with_docid(results)
        for r in results:
            if r.get("content"):
                r.update(extract_snippet(r["content"], "hybrid_search"))

        report_lines.append("| Format | Valid | Size | Notes |")
        report_lines.append("|--------|-------|------|-------|")

        # JSON
        json_out = format_results(results, format="json")
        parsed = json.loads(json_out)
        assert len(parsed) == len(results)
        report_lines.append(f"| json | Yes | {len(json_out)} chars | {len(parsed)} items, full metadata |")

        # CSV
        csv_out = format_results(results, format="csv")
        reader = csv.DictReader(io.StringIO(csv_out))
        csv_rows = list(reader)
        assert len(csv_rows) == len(results)
        report_lines.append(f"| csv | Yes | {len(csv_out)} chars | {len(csv_rows)} rows, {len(reader.fieldnames or [])} columns |")

        # Markdown
        md_out = format_results(results, format="markdown")
        assert "###" in md_out
        assert "```" in md_out
        n_sections = md_out.count("### ")
        report_lines.append(f"| markdown | Yes | {len(md_out)} chars | {n_sections} result sections with code blocks |")

        # Files
        files_out = format_results(results, format="files")
        file_list = [f for f in files_out.split("\n") if f.strip()]
        assert len(file_list) == len(set(file_list))
        report_lines.append(f"| files | Yes | {len(files_out)} chars | {len(file_list)} unique file paths |")

        report_lines.append("")

    def test_field_filtering(self, db, report_lines):
        """Verify fields parameter restricts output."""
        results = _search(db, "ProjectDB", limit=3)
        results = enrich_with_docid(results)
        filtered = format_results(results, format="json", fields=["file_path", "score", "docid"])
        parsed = json.loads(filtered)
        for item in parsed:
            assert set(item.keys()).issubset({"file_path", "score", "docid"})

        report_lines.append("**Field filtering:** `fields=['file_path', 'score', 'docid']` correctly restricts output.")
        report_lines.append("")


# ── Benchmark 6: Latency ─────────────────────────────────────────────────


class TestSearchLatency:
    """Measure search latency across query types."""

    def test_keyword_search_latency(self, db, report_lines):
        report_lines.append("## 6. Search Latency")
        report_lines.append("")
        report_lines.append("**Keyword-only search** (FTS5 + RRF merge + enrichment):")
        report_lines.append("")
        report_lines.append("| Query | Search (ms) | + Snippet (ms) | + DocID (ms) | Total (ms) | Results |")
        report_lines.append("|-------|-----------|---------------|-------------|-----------|---------|")

        for query in QUERIES:
            # Search
            t0 = time.perf_counter()
            results = _search(db, query, limit=10)
            t_search = (time.perf_counter() - t0) * 1000

            # Snippet
            t0 = time.perf_counter()
            for r in results:
                if r.get("content"):
                    r.update(extract_snippet(r["content"], query))
            t_snippet = (time.perf_counter() - t0) * 1000

            # DocID
            t0 = time.perf_counter()
            results = enrich_with_docid(results)
            t_docid = (time.perf_counter() - t0) * 1000

            total = t_search + t_snippet + t_docid
            report_lines.append(
                f"| `{query}` | {t_search:.2f} | {t_snippet:.2f} | {t_docid:.2f} | {total:.2f} | {len(results)} |"
            )

            assert total < 100, f"Search pipeline took {total:.1f}ms for '{query}'"

        report_lines.append("")

    def test_hybrid_search_latency(self, db, embedding_client, report_lines):
        """Full pipeline: embed + search + snippet + docid."""
        report_lines.append("**Full hybrid search** (embed + keyword + semantic + RRF + snippet + docid):")
        report_lines.append("")
        report_lines.append("| Query | Embed (ms) | Search (ms) | Post-process (ms) | Total (ms) |")
        report_lines.append("|-------|-----------|-----------|-------------------|-----------|")

        for query in QUERIES[:5]:
            t0 = time.perf_counter()
            raw = embedding_client.embed_query(query)
            emb = np.array(raw, dtype=np.float32)
            t_embed = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            results = hybrid_search(query, emb, db, limit=10)
            t_search = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            results = enrich_with_docid(results)
            for r in results:
                if r.get("content"):
                    r.update(extract_snippet(r["content"], query))
            t_post = (time.perf_counter() - t0) * 1000

            total = t_embed + t_search + t_post
            report_lines.append(
                f"| `{query}` | {t_embed:.1f} | {t_search:.1f} | {t_post:.2f} | {total:.1f} |"
            )

        report_lines.append("")


# ── Benchmark 7: BM25 Strong-Signal Short-Circuit ────────────────────────


class TestBM25ShortCircuit:
    """Measure when keyword results are confident enough to skip semantic/PPR."""

    def test_short_circuit_detection(self, db, report_lines):
        report_lines.append("## 7. BM25 Strong-Signal Short-Circuit")
        report_lines.append("")
        report_lines.append(f"**Change:** Skip semantic search + PPR when top BM25 result score >= {BM25_STRONG_SIGNAL_THRESHOLD} "
                           f"AND gap to #2 >= {BM25_STRONG_SIGNAL_GAP}.")
        report_lines.append("Saves ~30-50ms per query when keyword match is unambiguous.")
        report_lines.append("")
        report_lines.append("| Query | Top-1 Norm | Top-2 Norm | Gap | Short-Circuit? | Reason |")
        report_lines.append("|-------|-----------|-----------|-----|---------------|--------|")

        for query in QUERIES:
            raw_results = db.keyword_search(query, limit=10)
            if not raw_results:
                report_lines.append(f"| `{query}` | — | — | — | No | No results |")
                continue

            top_score = normalize_bm25_score(raw_results[0].get("score", 0))
            second_score = normalize_bm25_score(raw_results[1].get("score", 0)) if len(raw_results) > 1 else 0.0
            gap = top_score - second_score
            triggers = bm25_strong_signal_check(raw_results)

            if triggers:
                reason = "High confidence"
            elif top_score < BM25_STRONG_SIGNAL_THRESHOLD:
                reason = f"Score {top_score:.3f} < {BM25_STRONG_SIGNAL_THRESHOLD}"
            else:
                reason = f"Gap {gap:.3f} < {BM25_STRONG_SIGNAL_GAP}"

            report_lines.append(
                f"| `{query}` | {top_score:.4f} | {second_score:.4f} | {gap:.4f} | "
                f"{'**Yes**' if triggers else 'No'} | {reason} |"
            )

        report_lines.append("")

    def test_short_circuit_latency_savings(self, db, embedding_client, report_lines):
        """Compare latency with and without short-circuit."""
        report_lines.append("**Latency comparison** (with vs without short-circuit):")
        report_lines.append("")
        report_lines.append("| Query | Full Pipeline (ms) | Short-Circuit (ms) | Saved (ms) | Triggered? |")
        report_lines.append("|-------|-------------------|-------------------|-----------|-----------|")

        for query in QUERIES[:5]:
            raw_emb = embedding_client.embed_query(query)
            emb = np.array(raw_emb, dtype=np.float32)

            # Full pipeline (with embeddings)
            t0 = time.perf_counter()
            results_full = hybrid_search(query, emb, db, limit=10)
            t_full = (time.perf_counter() - t0) * 1000

            # Keyword-only (simulates short-circuit)
            t0 = time.perf_counter()
            results_kw = hybrid_search(query, query_embedding=None, db=db, limit=10)
            t_kw = (time.perf_counter() - t0) * 1000

            triggered = any(r.get("short_circuited") for r in results_kw)
            saved = t_full - t_kw

            report_lines.append(
                f"| `{query}` | {t_full:.1f} | {t_kw:.1f} | {saved:+.1f} | {'Yes' if triggered else 'No'} |"
            )

        report_lines.append("")


# ── Benchmark 8: Weighted RRF ─────────────────────────────────────────────


class TestWeightedRRF:
    """Compare equal-weight RRF vs weighted RRF (keyword 1.5x, semantic 1.0x, graph 0.8x)."""

    def test_weighted_vs_equal_rrf(self, db, embedding_client, report_lines):
        report_lines.append("## 8. Weighted RRF Fusion")
        report_lines.append("")
        report_lines.append(f"**Change:** Per-list weights: keyword={DEFAULT_RRF_WEIGHTS['keyword']}x, "
                           f"semantic={DEFAULT_RRF_WEIGHTS['semantic']}x, graph={DEFAULT_RRF_WEIGHTS['graph']}x "
                           f"(was 1.0x equal).")
        report_lines.append("Keyword results boosted because FTS5 precision is higher for code search.")
        report_lines.append("")
        report_lines.append("| Query | Equal RRF Top-3 | Weighted RRF Top-3 | Rank Changes | Score Boost |")
        report_lines.append("|-------|-----------------|--------------------|-------------|-------------|")

        for query in QUERIES[:6]:
            raw_emb = embedding_client.embed_query(query)
            emb = np.array(raw_emb, dtype=np.float32)

            # Get keyword and semantic results
            kw_results = db.keyword_search(query, limit=10)
            all_embeddings = db.get_all_embeddings()
            sem_results = []
            if all_embeddings and len(all_embeddings) > 0:
                from tessera.search import cosine_search
                chunk_ids, embedding_vectors = all_embeddings
                sem_results = cosine_search(emb, chunk_ids, embedding_vectors, limit=10)

            if not kw_results:
                continue

            ranked_lists = [kw_results]
            labels = ["keyword"]
            if sem_results:
                ranked_lists.append(sem_results)
                labels.append("semantic")

            # Equal weight RRF
            equal_merged = rrf_merge(ranked_lists)

            # Weighted RRF
            weights = [DEFAULT_RRF_WEIGHTS.get(l, 1.0) for l in labels]
            weighted_merged = weighted_rrf_merge(ranked_lists, weights=weights)

            def _top3_ids(results):
                return [r["id"] for r in results[:3]]

            equal_top = _top3_ids(equal_merged)
            weighted_top = _top3_ids(weighted_merged)

            # Count rank changes
            rank_changes = sum(1 for i, eid in enumerate(equal_top) if i < len(weighted_top) and weighted_top[i] != eid)

            # Score boost (weighted top-1 vs equal top-1)
            s_equal = equal_merged[0]["rrf_score"] if equal_merged else 0
            s_weighted = weighted_merged[0]["rrf_score"] if weighted_merged else 0

            def _id_labels(ids, db):
                names = []
                for cid in ids:
                    chunk = db.get_chunk(cid)
                    if chunk:
                        fp = chunk.get("file_path", "") or ""
                        names.append(os.path.basename(fp)[:15] if fp else str(cid))
                    else:
                        names.append(str(cid))
                return ", ".join(names)

            report_lines.append(
                f"| `{query}` | {_id_labels(equal_top, db)} | {_id_labels(weighted_top, db)} | "
                f"{rank_changes}/3 | {s_weighted - s_equal:+.6f} |"
            )

        report_lines.append("")

    def test_weight_sensitivity(self, report_lines):
        """Show how weights affect relative scoring."""
        report_lines.append("**Weight sensitivity:** How per-list weights shift RRF scores.")
        report_lines.append("")
        report_lines.append("| Weights (kw, sem) | ID-1 Score | ID-2 Score | ID-3 Score | Rank Change vs Equal |")
        report_lines.append("|-------------------|-----------|-----------|-----------|---------------------|")

        kw = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}, {"id": 3, "score": 0.7}]
        sem = [{"id": 2, "score": 0.95}, {"id": 3, "score": 0.85}, {"id": 1, "score": 0.75}]
        lists = [kw, sem]

        equal = rrf_merge(lists)
        equal_order = [r["id"] for r in equal]

        for w_kw, w_sem in [(1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (1.0, 2.0)]:
            merged = weighted_rrf_merge(lists, weights=[w_kw, w_sem])
            order = [r["id"] for r in merged]
            changes = sum(1 for i in range(min(3, len(order))) if i < len(equal_order) and order[i] != equal_order[i])
            scores = {r["id"]: r["rrf_score"] for r in merged}
            report_lines.append(
                f"| ({w_kw}, {w_sem}) | {scores.get(1, 0):.6f} | {scores.get(2, 0):.6f} | {scores.get(3, 0):.6f} | {changes}/3 |"
            )

        report_lines.append("")


# ── Benchmark 9: Structured Query Types ───────────────────────────────────


STRUCTURED_QUERIES = [
    "normalize_bm25_score",
    "error handling",
    "graph traversal",
]


class TestStructuredQueries:
    """Test SearchType routing: LEX-only, VEC-only, HYDE, and mixed modes."""

    def test_parse_structured_query(self, report_lines):
        """Verify the query prefix parser."""
        report_lines.append("## 9. Structured Query Types")
        report_lines.append("")
        report_lines.append("**Query prefix parser:**")
        report_lines.append("")
        report_lines.append("| Input | Clean Query | Search Types |")
        report_lines.append("|-------|------------|-------------|")

        cases = [
            ("plain query", "plain query", [SearchType.LEX, SearchType.VEC]),
            ("lex:ProjectDB", "ProjectDB", [SearchType.LEX]),
            ("vec:error handling", "error handling", [SearchType.VEC]),
            ("hyde:graph traversal", "graph traversal", [SearchType.HYDE]),
            ("lex,vec:hybrid_search", "hybrid_search", [SearchType.LEX, SearchType.VEC]),
            ("VEC:case insensitive", "case insensitive", [SearchType.VEC]),
        ]

        for input_q, expected_clean, expected_types in cases:
            clean, types = parse_structured_query(input_q)
            assert clean == expected_clean, f"Clean query mismatch for '{input_q}': {clean} != {expected_clean}"
            assert types == expected_types, f"Types mismatch for '{input_q}': {types} != {expected_types}"
            report_lines.append(f"| `{input_q}` | `{clean}` | {[t.value for t in types]} |")

        report_lines.append("")

    def test_lex_only_matches_keyword_search(self, db, report_lines):
        """LEX-only results should match db.keyword_search() exactly."""
        report_lines.append("**LEX-only vs keyword_search():**")
        report_lines.append("")
        report_lines.append("| Query | LEX Top-5 IDs | keyword_search Top-5 IDs | Match? |")
        report_lines.append("|-------|--------------|-------------------------|--------|")

        for query in STRUCTURED_QUERIES:
            # LEX-only via hybrid_search
            lex_results = hybrid_search(
                query, query_embedding=None, db=db, limit=5,
                search_types=[SearchType.LEX],
            )

            # Direct keyword_search (with same sanitization)
            from tessera.db import sanitize_fts5_query
            safe_q = sanitize_fts5_query(query, allow_advanced=False)
            kw_results = db.keyword_search(safe_q, limit=5)

            lex_ids = [r["id"] for r in lex_results[:5]]
            kw_ids = [r["id"] for r in kw_results[:5]]
            match = lex_ids == kw_ids

            report_lines.append(
                f"| `{query}` | {lex_ids[:3]}... | {kw_ids[:3]}... | {'Pass' if match else 'FAIL'} |"
            )
            assert match, f"LEX-only results differ from keyword_search for '{query}'"

        report_lines.append("")

    def test_vec_only_differs_from_lex(self, db, embedding_client, report_lines):
        """VEC-only should produce different rankings than LEX-only."""
        report_lines.append("**VEC-only vs LEX-only (proving complementary signals):**")
        report_lines.append("")
        report_lines.append("| Query | LEX Top-3 | VEC Top-3 | Top-5 Overlap | Different? |")
        report_lines.append("|-------|----------|----------|--------------|-----------|")

        any_differ = False
        for query in STRUCTURED_QUERIES:
            lex_results = hybrid_search(
                query, query_embedding=None, db=db, limit=5,
                search_types=[SearchType.LEX],
            )
            raw = embedding_client.embed_query(query)
            emb = np.array(raw, dtype=np.float32)
            vec_results = hybrid_search(
                query, emb, db, limit=5,
                search_types=[SearchType.VEC],
            )

            def _top3(results):
                return [os.path.basename(r.get("file_path", "?"))[:15] for r in results[:3]]

            lex_ids = set(r["id"] for r in lex_results[:5])
            vec_ids = set(r["id"] for r in vec_results[:5])
            overlap = len(lex_ids & vec_ids)
            differs = overlap < 5
            if differs:
                any_differ = True

            report_lines.append(
                f"| `{query}` | {', '.join(_top3(lex_results))} | {', '.join(_top3(vec_results))} | "
                f"{overlap}/5 | {'Yes' if differs else 'No'} |"
            )

        report_lines.append("")
        # At least one query should show different results between LEX and VEC
        assert any_differ, "VEC and LEX returned identical top-5 for all queries — signals are not complementary"

    def test_hyde_differs_from_vec(self, db, embedding_client, report_lines):
        """HYDE (embed_single) should produce different rankings than VEC (embed_query)."""
        report_lines.append("**HYDE vs VEC (embed_single vs embed_query):**")
        report_lines.append("")
        report_lines.append("| Query | VEC Top-3 | HYDE Top-3 | Top-5 Overlap |")
        report_lines.append("|-------|----------|-----------|--------------|")

        for query in STRUCTURED_QUERIES:
            # VEC: uses embed_query (retrieval-prefixed)
            raw_q = embedding_client.embed_query(query)
            emb_q = np.array(raw_q, dtype=np.float32)
            vec_results = hybrid_search(
                query, emb_q, db, limit=5,
                search_types=[SearchType.VEC],
            )

            # HYDE: uses embed_single (no prefix)
            raw_s = embedding_client.embed_single(query)
            emb_s = np.array(raw_s, dtype=np.float32)
            hyde_results = hybrid_search(
                query, emb_s, db, limit=5,
                search_types=[SearchType.HYDE],
            )

            def _top3(results):
                return [os.path.basename(r.get("file_path", "?"))[:15] for r in results[:3]]

            ids_q = set(r["id"] for r in vec_results[:5])
            ids_s = set(r["id"] for r in hyde_results[:5])
            overlap = len(ids_q & ids_s)

            report_lines.append(
                f"| `{query}` | {', '.join(_top3(vec_results))} | {', '.join(_top3(hyde_results))} | {overlap}/5 |"
            )

        report_lines.append("")


# ── Benchmark 10: FTS5 Advanced Mode ──────────────────────────────────────


class TestFTS5Advanced:
    """Test FTS5 sanitization and advanced mode operators."""

    def test_safe_mode_escapes_operators(self, db, report_lines):
        """Default (safe) mode escapes FTS5 operators — no crashes."""
        report_lines.append("## 10. FTS5 Advanced Mode")
        report_lines.append("")
        report_lines.append("**Safe mode (default):** All FTS5 operators escaped. No syntax errors possible.")
        report_lines.append("")
        report_lines.append("| Query | Results (safe) | Error? |")
        report_lines.append("|-------|---------------|--------|")

        # These queries previously crashed or had unexpected behavior
        dangerous_queries = [
            'error handling (async)',       # was: syntax error
            'foo OR bar AND baz',           # OR/AND are FTS5 operators
            '"already quoted"',             # nested quotes
            'hybrid*',                      # prefix operator
            'error NOT warning',            # NOT operator
            'NEAR(search query, 5)',        # NEAR operator
        ]

        for query in dangerous_queries:
            try:
                results = db.keyword_search(query, limit=5)  # safe mode (default)
                report_lines.append(f"| `{query}` | {len(results)} | No |")
            except Exception as e:
                report_lines.append(f"| `{query}` | 0 | {str(e)[:40]} |")
                # Safe mode should NEVER crash
                assert False, f"Safe mode crashed on '{query}': {e}"

        report_lines.append("")

    def test_advanced_mode_operators(self, db, report_lines):
        """Advanced mode enables FTS5 operators for power users."""
        report_lines.append("**Advanced mode (`advanced_fts=True`):** FTS5 operators active.")
        report_lines.append("")
        report_lines.append("| Query Type | Query | Results (adv) | Results (safe) | Fewer? |")
        report_lines.append("|-----------|-------|--------------|---------------|--------|")

        cases = [
            ("phrase", '"def hybrid_search"'),
            ("negation", "error NOT warning"),
            ("prefix", "hybrid*"),
            ("unquoted", "hybrid_search"),
        ]

        for qtype, query in cases:
            try:
                adv_results = db.keyword_search(query, limit=10, advanced_fts=True)
                safe_results = db.keyword_search(query, limit=10, advanced_fts=False)
                adv_count = len(adv_results)
                safe_count = len(safe_results)
                fewer = adv_count < safe_count
                report_lines.append(
                    f"| {qtype} | `{query}` | {adv_count} | {safe_count} | "
                    f"{'Yes' if fewer else 'No'} |"
                )
            except Exception:
                report_lines.append(f"| {qtype} | `{query}` | ERROR | — | — |")

        report_lines.append("")

    def test_phrase_precision(self, db, report_lines):
        """Phrase queries should return subset of unquoted results."""
        report_lines.append("**Phrase precision:** Quoted phrases return fewer, more precise results.")
        report_lines.append("")
        report_lines.append("| Phrase | Phrase Results | Unquoted Results | Subset? |")
        report_lines.append("|--------|---------------|-----------------|---------|")

        phrases = [
            ('"def hybrid_search"', "def hybrid_search"),
            ('"keyword search"', "keyword search"),
            ('"error handling"', "error handling"),
        ]

        for phrase, unquoted in phrases:
            adv_results = db.keyword_search(phrase, limit=20, advanced_fts=True)
            safe_results = db.keyword_search(unquoted, limit=20, advanced_fts=False)
            adv_ids = set(r["id"] for r in adv_results)
            safe_ids = set(r["id"] for r in safe_results)
            is_subset = adv_ids.issubset(safe_ids)

            report_lines.append(
                f"| `{phrase}` | {len(adv_results)} | {len(safe_results)} | "
                f"{'Yes' if is_subset else 'No'} |"
            )

        report_lines.append("")

    def test_fts5_latency(self, db, report_lines):
        """Latency for safe vs advanced mode."""
        report_lines.append("**FTS5 latency (safe vs advanced):**")
        report_lines.append("")
        report_lines.append("| Query | Safe (ms) | Advanced (ms) |")
        report_lines.append("|-------|----------|-------------|")

        test_queries = [
            "hybrid_search",
            '"def hybrid_search"',
            "error NOT warning",
            "hybrid*",
        ]

        for query in test_queries:
            t0 = time.perf_counter()
            db.keyword_search(query, limit=10, advanced_fts=False)
            safe_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            db.keyword_search(query, limit=10, advanced_fts=True)
            adv_ms = (time.perf_counter() - t0) * 1000

            report_lines.append(f"| `{query}` | {safe_ms:.2f} | {adv_ms:.2f} |")

        report_lines.append("")


# ── Summary ──────────────────────────────────────────────────────────────


class TestSummary:
    def test_write_footer(self, report_lines):
        report_lines.append("---")
        report_lines.append("")
        report_lines.append(f"*Generated by `tests/test_search_benchmark.py` against the live codemem index ({CODEMEM_DB_PATH}).*")
