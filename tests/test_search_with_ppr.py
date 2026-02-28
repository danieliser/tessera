"""Tests for search.py with PPR graph integration."""

import pytest
import numpy as np
import scipy.sparse
import json
import time
from unittest.mock import Mock

from tessera.search import hybrid_search, doc_search
from tessera.graph import ProjectGraph, ppr_to_ranked_list


class MockDB:
    """Mock database for testing hybrid_search with PPR."""

    def __init__(self, keyword_results=None, semantic_results=None, chunks=None):
        """Initialize mock DB.

        Args:
            keyword_results: List of keyword search results
            semantic_results: List of semantic search results
            chunks: Dict mapping chunk_id -> chunk metadata
        """
        self.keyword_results = keyword_results or []
        self.semantic_results = semantic_results or []
        self.chunks = chunks or {}
        self.embeddings_data = None

    def keyword_search(self, query, limit=10, source_type=None):
        """Mock keyword search."""
        results = self.keyword_results[:limit]
        if source_type:
            results = [
                r for r in results
                if self.get_chunk(r["id"]).get("source_type", "code") in source_type
            ][:limit]
        return results

    def get_all_embeddings(self):
        """Mock get_all_embeddings."""
        if self.embeddings_data:
            return self.embeddings_data
        return None

    def get_chunk(self, chunk_id):
        """Mock get_chunk."""
        return self.chunks.get(chunk_id, {})

    def get_symbol_to_chunks_mapping(self):
        """Mock get_symbol_to_chunks_mapping."""
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
        # Create a dense enough mock graph (edges >= n_symbols to avoid sparse fallback)
        n = 3
        # Create a graph with at least n edges to avoid sparse fallback
        rows = [0, 1, 2, 0]
        cols = [1, 2, 0, 2]
        data = [1.0, 1.0, 1.0, 1.0]
        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )
        symbol_ids = [10, 11, 12]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        # Create mock database with results
        chunks = {
            1: {
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def func_a(): pass",
                "source_type": "code",
                "symbol_ids": json.dumps([10]),
            },
            2: {
                "file_path": "test.py",
                "start_line": 6,
                "end_line": 10,
                "content": "def func_b(): pass",
                "source_type": "code",
                "symbol_ids": json.dumps([11]),
            },
            3: {
                "file_path": "test.py",
                "start_line": 11,
                "end_line": 15,
                "content": "def func_c(): pass",
                "source_type": "code",
                "symbol_ids": json.dumps([12]),
            },
        }

        keyword_results = [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.8},
        ]

        query_embedding = np.random.randn(768).astype(np.float32)
        chunk_ids = [1, 2, 3]
        embeddings = np.random.randn(3, 768).astype(np.float32)

        db = MockDB(keyword_results=keyword_results, chunks=chunks)
        db.embeddings_data = (chunk_ids, embeddings)

        results = hybrid_search("test query", query_embedding, db, graph=graph, limit=10)

        assert len(results) > 0
        # Verify rank_sources includes "graph" when graph is provided
        assert any("graph" in r.get("rank_sources", []) for r in results)
        # Verify graph_version is populated
        assert any(r.get("graph_version") is not None for r in results)

    def test_hybrid_search_without_graph(self):
        """Call hybrid_search without graph, verify backward compatibility."""
        chunks = {
            1: {
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def func_a(): pass",
                "source_type": "code",
            },
            2: {
                "file_path": "test.py",
                "start_line": 6,
                "end_line": 10,
                "content": "def func_b(): pass",
                "source_type": "code",
            },
        }

        keyword_results = [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.8},
        ]

        query_embedding = np.random.randn(768).astype(np.float32)
        chunk_ids = [1, 2]
        embeddings = np.random.randn(2, 768).astype(np.float32)

        db = MockDB(keyword_results=keyword_results, chunks=chunks)
        db.embeddings_data = (chunk_ids, embeddings)

        results = hybrid_search("test query", query_embedding, db, graph=None, limit=10)

        assert len(results) > 0
        # Verify rank_sources does not include "graph"
        for r in results:
            assert "graph" not in r.get("rank_sources", [])
        # Verify graph_version is None
        assert all(r.get("graph_version") is None for r in results)

    def test_hybrid_search_sparse_graph_fallback(self):
        """When graph.is_sparse_fallback() is True, PPR is skipped."""
        # Create a sparse graph where edge_count < n_symbols
        n = 5
        rows = [0]  # Only 1 edge
        cols = [1]
        data = [1.0]
        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )
        symbol_ids = [10, 11, 12, 13, 14]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        # Verify it's sparse
        assert graph.is_sparse_fallback()

        chunks = {
            1: {
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def func_a(): pass",
                "source_type": "code",
            },
        }

        keyword_results = [{"id": 1, "score": 0.9}]

        db = MockDB(keyword_results=keyword_results, chunks=chunks)

        results = hybrid_search("test query", None, db, graph=graph, limit=10)

        assert len(results) > 0
        # When sparse fallback, PPR is skipped, so rank_sources should not include "graph"
        for r in results:
            assert "graph" not in r.get("rank_sources", [])

    def test_hybrid_search_graph_version_populated(self):
        """Verify graph_version field is set in results."""
        n = 2
        rows = [0]
        cols = [1]
        data = [1.0]
        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )
        symbol_ids = [10, 11]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        loaded_at = time.perf_counter()
        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=loaded_at,
            id_to_idx=id_to_idx,
        )

        chunks = {
            1: {
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def func_a(): pass",
                "source_type": "code",
                "symbol_ids": json.dumps([10]),
            },
        }

        keyword_results = [{"id": 1, "score": 0.9}]

        db = MockDB(keyword_results=keyword_results, chunks=chunks)

        results = hybrid_search("test query", None, db, graph=graph, limit=10)

        assert len(results) > 0
        # All results should have graph_version set
        for r in results:
            assert r.get("graph_version") == loaded_at

    def test_doc_search_ignores_graph(self):
        """Verify doc_search ignores graph parameter."""
        n = 2
        rows = [0]
        cols = [1]
        data = [1.0]
        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )
        symbol_ids = [10, 11]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        chunks = {
            1: {
                "file_path": "test.md",
                "start_line": 0,
                "end_line": 0,
                "content": "# Test Document",
                "source_type": "markdown",
            },
        }

        keyword_results = [{"id": 1, "score": 0.9}]

        db = MockDB(keyword_results=keyword_results, chunks=chunks)

        # Call doc_search with graph parameter
        results = doc_search("test query", None, db, graph=graph, limit=10)

        assert len(results) > 0
        # doc_search should not use graph (passes graph=None to hybrid_search)
        for r in results:
            assert "graph" not in r.get("rank_sources", [])


class TestHybridSearchRRFMerging:
    """Test that three-way RRF works correctly."""

    def test_three_way_rrf_merge(self):
        """Verify three-way RRF when PPR produces results."""
        # Create a graph with meaningful edges (dense enough to avoid sparse fallback)
        n = 3
        rows = [0, 1, 2, 0]
        cols = [1, 2, 0, 2]
        data = [1.0, 1.0, 1.0, 1.0]
        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )
        symbol_ids = [10, 11, 12]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        chunks = {
            1: {
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def func_a(): pass",
                "source_type": "code",
                "symbol_ids": json.dumps([10]),
            },
            2: {
                "file_path": "test.py",
                "start_line": 6,
                "end_line": 10,
                "content": "def func_b(): pass",
                "source_type": "code",
                "symbol_ids": json.dumps([11]),
            },
            3: {
                "file_path": "test.py",
                "start_line": 11,
                "end_line": 15,
                "content": "def func_c(): pass",
                "source_type": "code",
                "symbol_ids": json.dumps([12]),
            },
        }

        # Keyword results favor chunk 1
        keyword_results = [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.7},
        ]

        query_embedding = np.random.randn(768).astype(np.float32)
        chunk_ids = [1, 2, 3]
        embeddings = np.random.randn(3, 768).astype(np.float32)

        db = MockDB(keyword_results=keyword_results, chunks=chunks)
        db.embeddings_data = (chunk_ids, embeddings)

        results = hybrid_search("test query", query_embedding, db, graph=graph, limit=3)

        assert len(results) > 0
        # All results should have rank_sources that includes at least two sources
        for r in results:
            sources = r.get("rank_sources", [])
            assert len(sources) >= 2


class TestHybridSearchEdgeCases:
    """Test edge cases and error handling."""

    def test_hybrid_search_with_bad_json_symbol_ids(self):
        """Handle malformed symbol_ids JSON gracefully."""
        n = 2
        rows = [0]
        cols = [1]
        data = [1.0]
        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )
        symbol_ids = [10, 11]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        chunks = {
            1: {
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def func_a(): pass",
                "source_type": "code",
                "symbol_ids": "not valid json",  # Bad JSON
            },
        }

        keyword_results = [{"id": 1, "score": 0.9}]

        db = MockDB(keyword_results=keyword_results, chunks=chunks)

        # Should not raise, should handle gracefully
        results = hybrid_search("test query", None, db, graph=graph, limit=10)

        assert len(results) > 0
        # Results should still have rank_sources (even if no PPR contribution)
        for r in results:
            assert "rank_sources" in r

    def test_hybrid_search_no_seed_symbols(self):
        """When keyword/semantic results have no symbol_ids, PPR is skipped."""
        n = 2
        rows = [0]
        cols = [1]
        data = [1.0]
        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )
        symbol_ids = [10, 11]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        chunks = {
            1: {
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def func_a(): pass",
                "source_type": "code",
                # No symbol_ids field
            },
        }

        keyword_results = [{"id": 1, "score": 0.9}]

        db = MockDB(keyword_results=keyword_results, chunks=chunks)

        results = hybrid_search("test query", None, db, graph=graph, limit=10)

        assert len(results) > 0
        # PPR should be skipped due to no seed symbols
        for r in results:
            assert "graph" not in r.get("rank_sources", [])
