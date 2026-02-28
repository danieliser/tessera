"""Performance benchmarks for Personalized PageRank computation.

Tests PPR on synthetic graphs of varying sizes to validate performance gates.
"""

import pytest
import numpy as np
import scipy.sparse
import time
from unittest.mock import Mock

from tessera.graph import ProjectGraph, load_project_graph
from tessera.db import ProjectDB


class TestPPRPerformanceBenchmarks:
    """Benchmark PPR performance on graphs of different sizes."""

    def test_1k_symbols_5k_edges_under_10ms(self):
        """1K symbols, 5K edges — PPR must complete in <10ms."""
        n = 1000
        n_edges = 5000

        np.random.seed(42)
        rows = np.random.randint(0, n, n_edges)
        cols = np.random.randint(0, n, n_edges)
        data = np.ones(n_edges, dtype=np.float32)

        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )

        symbol_ids = list(range(1000, 1000 + n))
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        seed_ids = symbol_ids[:10]

        start = time.perf_counter()
        result = graph.personalized_pagerank(seed_ids)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\n1K symbols, 5K edges: {elapsed_ms:.2f}ms")
        assert len(result) > 0
        assert elapsed_ms < 10, f"PPR took {elapsed_ms:.2f}ms, must be <10ms"

    def test_10k_symbols_50k_edges_under_50ms(self):
        """10K symbols, 50K edges — PPR should complete in <50ms (green)."""
        n = 10000
        n_edges = 50000

        np.random.seed(42)
        rows = np.random.randint(0, n, n_edges)
        cols = np.random.randint(0, n, n_edges)
        data = np.ones(n_edges, dtype=np.float32)

        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )

        symbol_ids = list(range(10000, 10000 + n))
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        seed_ids = symbol_ids[:10]

        start = time.perf_counter()
        result = graph.personalized_pagerank(seed_ids)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\n10K symbols, 50K edges: {elapsed_ms:.2f}ms")
        assert len(result) > 0
        assert elapsed_ms < 50, f"PPR took {elapsed_ms:.2f}ms, must be <50ms"

    @pytest.mark.performance
    def test_50k_symbols_250k_edges_under_100ms(self):
        """50K symbols, 250K edges — PPR must complete in <100ms (red gate)."""
        n = 50000
        n_edges = 250000

        np.random.seed(42)
        rows = np.random.randint(0, n, n_edges)
        cols = np.random.randint(0, n, n_edges)
        data = np.ones(n_edges, dtype=np.float32)

        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )

        symbol_ids = list(range(50000, 50000 + n))
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        seed_ids = symbol_ids[:10]

        start = time.perf_counter()
        result = graph.personalized_pagerank(seed_ids)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\n50K symbols, 250K edges: {elapsed_ms:.2f}ms")
        assert len(result) > 0
        assert elapsed_ms < 100, f"PPR took {elapsed_ms:.2f}ms, must be <100ms"

    def test_graph_loading_benchmark(self, test_project_dir):
        """Benchmark graph loading from ProjectDB."""
        db = ProjectDB(str(test_project_dir / "bench_project"))

        # Insert test data
        file_id = db.conn.execute(
            "INSERT INTO files (project_id, path, language) VALUES (?, ?, ?)",
            (1, "/test.py", "python"),
        ).lastrowid

        # Insert many symbols
        n_symbols = 1000
        symbol_ids = []
        for i in range(n_symbols):
            sid = db.conn.execute(
                "INSERT INTO symbols (project_id, file_id, name, kind, line, col) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, file_id, f"func_{i}", "function", i, 0),
            ).lastrowid
            symbol_ids.append(sid)

        # Insert many edges
        np.random.seed(42)
        n_edges = 5000
        for _ in range(n_edges):
            from_idx = np.random.randint(0, n_symbols)
            to_idx = np.random.randint(0, n_symbols)
            db.conn.execute(
                "INSERT INTO edges (project_id, from_id, to_id, type, weight) "
                "VALUES (?, ?, ?, ?, ?)",
                (1, symbol_ids[from_idx], symbol_ids[to_idx], "call", 1.0),
            )

        db.conn.commit()

        # Mock get_all_symbols and get_all_edges
        def mock_get_all_symbols(project_id):
            cursor = db.conn.execute(
                "SELECT id, name FROM symbols WHERE project_id = ? ORDER BY id",
                (project_id,),
            )
            return [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]

        def mock_get_all_edges(project_id):
            cursor = db.conn.execute(
                "SELECT from_id, to_id, weight FROM edges WHERE project_id = ?",
                (project_id,),
            )
            return [
                {"from_id": row[0], "to_id": row[1], "weight": row[2]}
                for row in cursor.fetchall()
            ]

        db.get_all_symbols = mock_get_all_symbols
        db.get_all_edges = mock_get_all_edges

        # Benchmark loading
        start = time.perf_counter()
        graph = load_project_graph(db, 1)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\nGraph loading (1K symbols, 5K edges): {elapsed_ms:.2f}ms")
        assert graph.n_symbols == n_symbols
        # Note: CSR matrix may deduplicate edges, so edge_count may be less than n_edges
        # Just check that we have a reasonable number of edges
        assert graph.edge_count > 0, "Should have at least some edges"
        # Loading should be reasonably fast
        assert elapsed_ms < 1000, f"Graph loading took {elapsed_ms:.2f}ms"
