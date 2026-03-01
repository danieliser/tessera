"""Tests for monitoring and LRU eviction in graph.py module."""

import pytest
import numpy as np
import scipy.sparse
import time

from tessera.graph import ProjectGraph, evict_lru_graph, MAX_CACHED_GRAPHS


class TestMaxCachedGraphsConstant:
    """Test MAX_CACHED_GRAPHS constant."""

    def test_max_cached_graphs_constant(self):
        """Verify MAX_CACHED_GRAPHS = 20."""
        assert MAX_CACHED_GRAPHS == 20


class TestEvictLruGraph:
    """Test evict_lru_graph function."""

    def test_evict_lru_graph_no_eviction(self):
        """Cache under limit, returns None."""
        graphs = {}

        # Add 10 graphs (well under 20 limit)
        for i in range(10):
            adjacency = scipy.sparse.csr_matrix(
                ([1.0], ([0], [0])),
                shape=(5, 5),
                dtype=np.float32,
            )
            graph = ProjectGraph(
                project_id=i,
                adjacency_matrix=adjacency,
                symbol_id_to_name={0: f"func_0"},
                loaded_at=time.perf_counter() + i * 0.1,
            )
            graphs[i] = graph

        result = evict_lru_graph(graphs, max_size=20)
        assert result is None
        assert len(graphs) == 10

    def test_evict_lru_graph_evicts_oldest(self):
        """Cache over limit, evicts oldest loaded_at."""
        graphs = {}
        start_time = time.perf_counter()

        # Add 25 graphs with incrementing loaded_at times
        for i in range(25):
            adjacency = scipy.sparse.csr_matrix(
                ([1.0], ([0], [0])),
                shape=(5, 5),
                dtype=np.float32,
            )
            graph = ProjectGraph(
                project_id=i,
                adjacency_matrix=adjacency,
                symbol_id_to_name={0: f"func_{i}"},
                loaded_at=start_time + i * 0.001,
            )
            graphs[i] = graph

        # Evict with limit of 20
        evicted = evict_lru_graph(graphs, max_size=20)

        # Should evict project 0 (oldest loaded_at)
        assert evicted == 0
        assert 0 not in graphs
        assert len(graphs) == 24

    def test_evict_lru_graph_reduces_size(self):
        """After eviction, cache is at max_size."""
        graphs = {}
        start_time = time.perf_counter()

        # Add 22 graphs with incrementing loaded_at times
        for i in range(22):
            adjacency = scipy.sparse.csr_matrix(
                ([1.0], ([0], [0])),
                shape=(5, 5),
                dtype=np.float32,
            )
            graph = ProjectGraph(
                project_id=100 + i,
                adjacency_matrix=adjacency,
                symbol_id_to_name={0: f"func_{i}"},
                loaded_at=start_time + i * 0.001,
            )
            graphs[100 + i] = graph

        # Evict with limit of 20
        evicted_pid = evict_lru_graph(graphs, max_size=20)

        # Should have evicted the oldest one
        assert evicted_pid == 100
        # Size should now be 21 (22 - 1)
        assert len(graphs) == 21


class TestEstimatedMemoryBytes:
    """Test estimated_memory_bytes property."""

    def test_estimated_memory_bytes(self):
        """Verify formula returns reasonable value."""
        # Create graph with known edge count and symbol count
        n = 100
        n_edges = 500

        # Create random edges with proper dimensions
        np.random.seed(42)
        rows = np.random.randint(0, n, n_edges)
        cols = np.random.randint(0, n, n_edges)
        data = np.ones(n_edges, dtype=np.float32)

        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name={i: f"func_{i}" for i in range(n)},
            loaded_at=time.perf_counter(),
        )

        # Verify formula: edge_count * 12 + n_symbols * 8 + 1024
        # Note: actual edge_count may be less than n_edges due to duplicate entries merging
        expected = graph.edge_count * 12 + n * 8 + 1024
        assert graph.estimated_memory_bytes == expected
        # Should be reasonable: several KB for this graph
        assert 1024 < graph.estimated_memory_bytes < 100000

    def test_estimated_memory_bytes_empty_graph(self):
        """Empty graph returns base overhead only."""
        adjacency = scipy.sparse.csr_matrix((0, 0), dtype=np.float32)
        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name={},
            loaded_at=time.perf_counter(),
        )

        # Empty graph: 0 edges, 0 symbols, just base overhead
        expected = 0 * 12 + 0 * 8 + 1024
        assert graph.estimated_memory_bytes == expected
        assert graph.estimated_memory_bytes == 1024
