"""Tests for search.py with PPR graph integration."""

import pytest
import numpy as np
import scipy.sparse
import json
import time
from unittest.mock import Mock

from tessera.search import hybrid_search, doc_search
from tessera.graph import ProjectGraph, ppr_to_ranked_list


def _make_ppr_graph(n: int = 120, loaded_at: float = None) -> ProjectGraph:
    """Create a connected graph that passes the adaptive sparse threshold.

    Builds a chain of n nodes (ensuring single connected component) plus
    extra random edges for density > 0.75. n must be >= 100.
    """
    if loaded_at is None:
        loaded_at = time.perf_counter()
    # Chain edges for full connectivity
    chain_rows = list(range(n - 1))
    chain_cols = list(range(1, n))
    # Extra edges to push density above 0.75
    np.random.seed(42)
    extra = max(0, int(n * 0.85) - (n - 1))
    extra_r = np.random.randint(0, n, extra).tolist()
    extra_c = np.random.randint(0, n, extra).tolist()
    rows = chain_rows + extra_r
    cols = chain_cols + extra_c
    adjacency = scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n), dtype=np.float32,
    )
    symbol_ids = list(range(100, 100 + n))
    id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
    names = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}
    graph = ProjectGraph(
        project_id=1, adjacency_matrix=adjacency,
        symbol_id_to_name=names, loaded_at=loaded_at, id_to_idx=id_to_idx,
    )
    assert not graph.is_sparse_fallback(), (
        f"Helper graph should not be sparse: {graph.n_symbols} symbols, "
        f"{graph.edge_count} edges, LCC={graph.largest_cc_size}"
    )
    return graph


def _make_mock_chunks_and_db(symbol_ids_for_chunks, keyword_results=None, embeddings=True):
    """Build MockDB with chunks mapped to given symbol IDs."""
    chunks = {}
    for i, sids in enumerate(symbol_ids_for_chunks, start=1):
        chunks[i] = {
            "file_path": "test.py",
            "start_line": (i - 1) * 5 + 1,
            "end_line": i * 5,
            "content": f"def func_{i}(): pass",
            "source_type": "code",
            "symbol_ids": json.dumps(sids),
        }
    if keyword_results is None:
        keyword_results = [{"id": i, "score": 1.0 / i} for i in chunks]
    db = MockDB(keyword_results=keyword_results, chunks=chunks)
    if embeddings:
        chunk_ids = list(chunks.keys())
        emb = np.random.randn(len(chunk_ids), 768).astype(np.float32)
        db.embeddings_data = (chunk_ids, emb)
    return db


class MockDB:
    """Mock database for testing hybrid_search with PPR."""

    def __init__(self, keyword_results=None, semantic_results=None, chunks=None):
        self.keyword_results = keyword_results or []
        self.semantic_results = semantic_results or []
        self.chunks = chunks or {}
        self.embeddings_data = None

    def keyword_search(self, query, limit=10, source_type=None):
        results = self.keyword_results[:limit]
        if source_type:
            results = [
                r for r in results
                if self.get_chunk(r["id"]).get("source_type", "code") in source_type
            ][:limit]
        return results

    def get_all_embeddings(self):
        if self.embeddings_data:
            return self.embeddings_data
        return None

    def get_chunk(self, chunk_id):
        return self.chunks.get(chunk_id, {})

    def get_symbol_to_chunks_mapping(self):
        mapping = {}
        for chunk_id, chunk in self.chunks.items():
            if "symbol_ids" in chunk:
                try:
                    sids = json.loads(chunk["symbol_ids"])
                    for sid in sids:
                        if sid not in mapping:
                            mapping[sid] = []
                        mapping[sid].append(chunk_id)
                except (json.JSONDecodeError, TypeError):
                    pass
        return mapping


class TestHybridSearchWithGraph:
    """Test hybrid_search with PPR graph parameter."""

    def test_hybrid_search_with_graph(self):
        """Call hybrid_search with a graph, verify PPR is included."""
        graph = _make_ppr_graph()
        # Use symbol IDs that exist in the graph (100, 101, 102)
        db = _make_mock_chunks_and_db([[100], [101], [102]])

        query_embedding = np.random.randn(768).astype(np.float32)
        results = hybrid_search("test query", query_embedding, db, graph=graph, limit=10)

        assert len(results) > 0
        assert any("graph" in r.get("rank_sources", []) for r in results)
        assert any(r.get("graph_version") is not None for r in results)

    def test_hybrid_search_without_graph(self):
        """Call hybrid_search without graph, verify backward compatibility."""
        chunks = {
            1: {
                "file_path": "test.py", "start_line": 1, "end_line": 5,
                "content": "def func_a(): pass", "source_type": "code",
            },
            2: {
                "file_path": "test.py", "start_line": 6, "end_line": 10,
                "content": "def func_b(): pass", "source_type": "code",
            },
        }
        keyword_results = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]
        query_embedding = np.random.randn(768).astype(np.float32)
        db = MockDB(keyword_results=keyword_results, chunks=chunks)
        db.embeddings_data = ([1, 2], np.random.randn(2, 768).astype(np.float32))

        results = hybrid_search("test query", query_embedding, db, graph=None, limit=10)

        assert len(results) > 0
        for r in results:
            assert "graph" not in r.get("rank_sources", [])
        assert all(r.get("graph_version") is None for r in results)

    def test_hybrid_search_sparse_graph_fallback(self):
        """When graph.is_sparse_fallback() is True, PPR is skipped."""
        # Small graph (< 100 symbols) triggers sparse fallback
        n = 5
        adjacency = scipy.sparse.csr_matrix(
            ([1.0], ([0], [1])), shape=(n, n), dtype=np.float32,
        )
        symbol_ids = [10, 11, 12, 13, 14]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        graph = ProjectGraph(
            project_id=1, adjacency_matrix=adjacency,
            symbol_id_to_name={sid: f"func_{i}" for i, sid in enumerate(symbol_ids)},
            loaded_at=time.perf_counter(), id_to_idx=id_to_idx,
        )
        assert graph.is_sparse_fallback()

        chunks = {1: {"file_path": "test.py", "start_line": 1, "end_line": 5,
                       "content": "def func_a(): pass", "source_type": "code"}}
        db = MockDB(keyword_results=[{"id": 1, "score": 0.9}], chunks=chunks)

        results = hybrid_search("test query", None, db, graph=graph, limit=10)
        assert len(results) > 0
        for r in results:
            assert "graph" not in r.get("rank_sources", [])

    def test_hybrid_search_graph_version_populated(self):
        """Verify graph_version field is set in results even for small graphs."""
        # graph_version is set whenever graph is passed, regardless of sparse fallback
        n = 2
        adjacency = scipy.sparse.csr_matrix(
            ([1.0], ([0], [1])), shape=(n, n), dtype=np.float32,
        )
        loaded_at = time.perf_counter()
        graph = ProjectGraph(
            project_id=1, adjacency_matrix=adjacency,
            symbol_id_to_name={10: "f0", 11: "f1"},
            loaded_at=loaded_at, id_to_idx={10: 0, 11: 1},
        )

        chunks = {1: {"file_path": "test.py", "start_line": 1, "end_line": 5,
                       "content": "def func_a(): pass", "source_type": "code",
                       "symbol_ids": json.dumps([10])}}
        db = MockDB(keyword_results=[{"id": 1, "score": 0.9}], chunks=chunks)

        results = hybrid_search("test query", None, db, graph=graph, limit=10)
        assert len(results) > 0
        for r in results:
            assert r.get("graph_version") == loaded_at

    def test_doc_search_ignores_graph(self):
        """Verify doc_search ignores graph parameter."""
        # doc_search passes graph=None internally, so graph size doesn't matter
        n = 2
        adjacency = scipy.sparse.csr_matrix(
            ([1.0], ([0], [1])), shape=(n, n), dtype=np.float32,
        )
        graph = ProjectGraph(
            project_id=1, adjacency_matrix=adjacency,
            symbol_id_to_name={10: "f0", 11: "f1"},
            loaded_at=time.perf_counter(), id_to_idx={10: 0, 11: 1},
        )

        chunks = {1: {"file_path": "test.md", "start_line": 0, "end_line": 0,
                       "content": "# Test Document", "source_type": "markdown"}}
        db = MockDB(keyword_results=[{"id": 1, "score": 0.9}], chunks=chunks)

        results = doc_search("test query", None, db, graph=graph, limit=10)
        assert len(results) > 0
        for r in results:
            assert "graph" not in r.get("rank_sources", [])


class TestHybridSearchRRFMerging:
    """Test that three-way RRF works correctly."""

    def test_three_way_rrf_merge(self):
        """Verify three-way RRF when PPR produces results."""
        graph = _make_ppr_graph()
        db = _make_mock_chunks_and_db(
            [[100], [101], [102]],
            keyword_results=[{"id": 1, "score": 0.9}, {"id": 2, "score": 0.7}],
        )
        query_embedding = np.random.randn(768).astype(np.float32)

        results = hybrid_search("test query", query_embedding, db, graph=graph, limit=3)

        assert len(results) > 0
        for r in results:
            sources = r.get("rank_sources", [])
            assert len(sources) >= 2


class TestHybridSearchEdgeCases:
    """Test edge cases and error handling."""

    def test_hybrid_search_with_bad_json_symbol_ids(self):
        """Handle malformed symbol_ids JSON gracefully."""
        graph = _make_ppr_graph()
        chunks = {1: {"file_path": "test.py", "start_line": 1, "end_line": 5,
                       "content": "def func_a(): pass", "source_type": "code",
                       "symbol_ids": "not valid json"}}
        db = MockDB(keyword_results=[{"id": 1, "score": 0.9}], chunks=chunks)

        results = hybrid_search("test query", None, db, graph=graph, limit=10)
        assert len(results) > 0
        for r in results:
            assert "rank_sources" in r

    def test_hybrid_search_no_seed_symbols(self):
        """When keyword/semantic results have no symbol_ids, PPR is skipped."""
        graph = _make_ppr_graph()
        # No symbol_ids in chunk
        chunks = {1: {"file_path": "test.py", "start_line": 1, "end_line": 5,
                       "content": "def func_a(): pass", "source_type": "code"}}
        db = MockDB(keyword_results=[{"id": 1, "score": 0.9}], chunks=chunks)

        results = hybrid_search("test query", None, db, graph=graph, limit=10)
        assert len(results) > 0
        for r in results:
            assert "graph" not in r.get("rank_sources", [])
