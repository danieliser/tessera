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
    normalize_bm25_score,
    extract_snippet,
    enrich_with_docid,
    generate_docid,
    format_results,
    hybrid_search,
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
    report_path = Path(__file__).parent.parent / "docs" / "benchmark-search-quality.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def _resolve_file_paths(db, results):
    """Resolve file_id → file_path for results missing file paths.

    hybrid_search enriches from chunk_meta which has file_id but not file_path.
    This joins against the files table to fill in the actual paths.
    """
    for r in results:
        if not r.get("file_path") and r.get("file_id"):
            row = db.conn.execute(
                "SELECT path FROM files WHERE id = ?", (r["file_id"],)
            ).fetchone()
            if row:
                r["file_path"] = row[0]
    return results


def _search(db, query, limit=10):
    """Run keyword-only hybrid search with file path resolution."""
    results = hybrid_search(query, query_embedding=None, db=db, limit=limit)
    # hybrid_search enrichment doesn't join files table — fix file_path
    for r in results:
        if not r.get("file_path"):
            chunk = db.get_chunk(r["id"])
            if chunk and chunk.get("file_id"):
                row = db.conn.execute(
                    "SELECT path FROM files WHERE id = ?", (chunk["file_id"],)
                ).fetchone()
                if row:
                    r["file_path"] = row[0]
    return results


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


# ── Summary ──────────────────────────────────────────────────────────────


class TestSummary:
    def test_write_footer(self, report_lines):
        report_lines.append("---")
        report_lines.append("")
        report_lines.append(f"*Generated by `tests/test_search_benchmark.py` against the live codemem index ({CODEMEM_DB_PATH}).*")
