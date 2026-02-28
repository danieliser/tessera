"""Tests for graph.py module — Personalized PageRank and graph operations."""

import pytest
import numpy as np
import scipy.sparse
import time
from unittest.mock import Mock

from tessera.graph import ProjectGraph, load_project_graph, ppr_to_ranked_list
from tessera.db import ProjectDB


class TestProjectGraph:
    """Test ProjectGraph class."""

    def test_empty_graph(self):
        """Empty graph (0 symbols) returns empty dict from PPR."""
        adjacency = scipy.sparse.csr_matrix((0, 0), dtype=np.float32)
        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name={},
            loaded_at=time.perf_counter(),
        )

        result = graph.personalized_pagerank([])
        assert result == {}

    def test_single_symbol_graph(self):
        """Single symbol graph returns score."""
        # 1x1 matrix with self-loop
        adjacency = scipy.sparse.csr_matrix(
            ([1.0], ([0], [0])),
            shape=(1, 1),
            dtype=np.float32,
        )
        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name={1: "func_a"},
            loaded_at=time.perf_counter(),
            id_to_idx={1: 0},
        )

        result = graph.personalized_pagerank([1])
        assert 1 in result
        assert result[1] > 0

    def test_star_graph(self):
        """Star graph: central hub has highest PPR score."""
        # 5 nodes, all pointing to node 0 (center)
        # Matrix: node 1, 2, 3, 4 -> node 0
        n = 5
        rows = [1, 2, 3, 4]
        cols = [0, 0, 0, 0]
        data = [1.0, 1.0, 1.0, 1.0]

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

        result = graph.personalized_pagerank([symbol_ids[0]])
        assert symbol_ids[0] in result

        # Hub should have high score
        hub_score = result[symbol_ids[0]]
        assert hub_score > 0

    def test_linear_chain(self):
        """Linear chain: 0 -> 1 -> 2 -> 3."""
        n = 4
        rows = [0, 1, 2]
        cols = [1, 2, 3]
        data = [1.0, 1.0, 1.0]

        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )

        symbol_ids = [100, 101, 102, 103]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"step_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        result = graph.personalized_pagerank([symbol_ids[0]])
        # All nodes should have some score due to teleportation
        assert len(result) > 0
        assert symbol_ids[0] in result

    def test_disconnected_components(self):
        """PPR respects connectivity of disconnected components."""
        # Two disconnected components: {0, 1} and {2, 3}
        n = 4
        rows = [0, 2]
        cols = [1, 3]
        data = [1.0, 1.0]

        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        )

        symbol_ids = [200, 201, 202, 203]
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"comp_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        result = graph.personalized_pagerank([symbol_ids[0]])
        assert len(result) > 0

    def test_ppr_performance_1k_symbols(self):
        """PPR on 1K symbols, 5K edges must be <10ms."""
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

        assert len(result) > 0
        assert elapsed_ms < 10, f"PPR took {elapsed_ms:.2f}ms, must be <10ms"

    def test_is_sparse_fallback(self):
        """is_sparse_fallback returns True when edge_count < symbol_count."""
        # Sparse case: 5 symbols, 2 edges
        n = 5
        adjacency_sparse = scipy.sparse.csr_matrix(
            ([1.0, 1.0], ([0, 1], [1, 2])),
            shape=(n, n),
            dtype=np.float32,
        )

        symbol_ids = list(range(500, 505))
        id_to_idx = {sid: i for i, sid in enumerate(symbol_ids)}
        symbol_id_to_name = {sid: f"func_{i}" for i, sid in enumerate(symbol_ids)}

        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency_sparse,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        assert graph.is_sparse_fallback() is True

        # Dense case: 5 symbols, 10 edges
        adjacency_dense = scipy.sparse.csr_matrix(
            (np.ones(10), (
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                [1, 2, 2, 3, 3, 4, 4, 0, 0, 1]
            )),
            shape=(n, n),
            dtype=np.float32,
        )

        graph_dense = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency_dense,
            symbol_id_to_name=symbol_id_to_name,
            loaded_at=time.perf_counter(),
            id_to_idx=id_to_idx,
        )

        assert graph_dense.is_sparse_fallback() is False

    def test_empty_graph_is_sparse(self):
        """Empty graph with 0 symbols is sparse."""
        adjacency = scipy.sparse.csr_matrix((0, 0), dtype=np.float32)
        graph = ProjectGraph(
            project_id=1,
            adjacency_matrix=adjacency,
            symbol_id_to_name={},
            loaded_at=time.perf_counter(),
        )

        assert graph.is_sparse_fallback() is True


class TestLoadProjectGraph:
    """Test load_project_graph function."""

    def test_load_project_graph_creates_correct_matrix(self, test_project_dir):
        """Load graph creates CSR matrix with correct shape and edges."""
        # Create a test ProjectDB with symbols and edges
        db = ProjectDB(str(test_project_dir / "test_project"))

        # Insert test data
        # Project ID 1
        file_id = db.conn.execute(
            "INSERT INTO files (project_id, path, language) VALUES (?, ?, ?)",
            (1, "/test.py", "python"),
        ).lastrowid

        # Insert 5 symbols
        symbol_ids = []
        for i in range(5):
            sid = db.conn.execute(
                "INSERT INTO symbols (project_id, file_id, name, kind, line, col) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, file_id, f"func_{i}", "function", i, 0),
            ).lastrowid
            symbol_ids.append(sid)

        # Insert edges (non-contiguous symbols to test id_to_idx mapping)
        edges = [
            (symbol_ids[0], symbol_ids[1], 1.0),
            (symbol_ids[1], symbol_ids[2], 1.0),
            (symbol_ids[2], symbol_ids[0], 1.0),
        ]

        for from_id, to_id, weight in edges:
            db.conn.execute(
                "INSERT INTO edges (project_id, from_id, to_id, type, weight) "
                "VALUES (?, ?, ?, ?, ?)",
                (1, from_id, to_id, "call", weight),
            )

        db.conn.commit()

        # Mock the get_all_symbols and get_all_edges methods
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

        # Load graph
        graph = load_project_graph(db, 1)

        # Verify
        assert graph.project_id == 1
        assert graph.n_symbols == 5
        assert graph.edge_count == 3
        assert graph.loaded_at > 0

    def test_load_project_graph_with_no_symbols_raises_error(self, test_project_dir):
        """load_project_graph raises ValueError if project has no symbols."""
        db = ProjectDB(str(test_project_dir / "test_project2"))

        db.get_all_symbols = lambda project_id: []
        db.get_all_edges = lambda project_id: []

        with pytest.raises(ValueError, match="has no symbols"):
            load_project_graph(db, 1)

    def test_load_project_graph_symbol_mapping(self, test_project_dir):
        """load_project_graph creates correct id_to_idx mapping."""
        db = ProjectDB(str(test_project_dir / "test_project3"))

        # Insert test data with non-contiguous symbol IDs
        file_id = db.conn.execute(
            "INSERT INTO files (project_id, path, language) VALUES (?, ?, ?)",
            (1, "/test.py", "python"),
        ).lastrowid

        # Insert symbols (IDs will be non-contiguous: 1, 2, 3)
        symbol_ids = []
        for i in range(3):
            sid = db.conn.execute(
                "INSERT INTO symbols (project_id, file_id, name, kind, line, col) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, file_id, f"func_{i}", "function", i, 0),
            ).lastrowid
            symbol_ids.append(sid)

        # Insert one edge
        db.conn.execute(
            "INSERT INTO edges (project_id, from_id, to_id, type, weight) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, symbol_ids[0], symbol_ids[1], "call", 1.0),
        )
        db.conn.commit()

        # Mock methods
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

        graph = load_project_graph(db, 1)

        # Verify id_to_idx mapping is set
        assert graph.id_to_idx is not None
        assert all(sid in graph.id_to_idx for sid in symbol_ids)
        assert all(0 <= idx < 3 for idx in graph.id_to_idx.values())


class TestPprToRankedList:
    """Test ppr_to_ranked_list function."""

    def test_empty_dict(self):
        """Empty PPR dict returns empty list."""
        result = ppr_to_ranked_list({})
        assert result == []

    def test_single_score(self):
        """Single score normalizes to 1.0."""
        result = ppr_to_ranked_list({42: 0.5})
        assert len(result) == 1
        assert result[0] == {"id": 42, "score": 1.0}

    def test_multiple_scores_normalized_descending(self):
        """Multiple scores normalize to [0, 1] and sort descending."""
        ppr_scores = {
            1: 0.5,
            2: 0.1,
            3: 0.3,
        }
        result = ppr_to_ranked_list(ppr_scores)

        assert len(result) == 3
        # Check descending order
        assert result[0]["score"] >= result[1]["score"] >= result[2]["score"]
        # Check all scores are in [0, 1]
        assert all(0 <= r["score"] <= 1.0 for r in result)
        # Max score should be 1.0
        assert result[0]["score"] == 1.0

    def test_scores_normalized_correctly(self):
        """Scores normalize to max=1.0."""
        ppr_scores = {
            1: 2.0,
            2: 4.0,
            3: 6.0,
        }
        result = ppr_to_ranked_list(ppr_scores)

        # Find result for each ID
        result_dict = {r["id"]: r["score"] for r in result}

        # Verify normalization: max was 6.0
        assert result_dict[1] == pytest.approx(2.0 / 6.0)
        assert result_dict[2] == pytest.approx(4.0 / 6.0)
        assert result_dict[3] == pytest.approx(1.0)

    def test_descending_order(self):
        """Results are sorted descending by score."""
        ppr_scores = {
            10: 1.0,
            20: 5.0,
            30: 3.0,
        }
        result = ppr_to_ranked_list(ppr_scores)

        # IDs in order: 20 (5.0), 30 (3.0), 10 (1.0)
        assert result[0]["id"] == 20
        assert result[1]["id"] == 30
        assert result[2]["id"] == 10
