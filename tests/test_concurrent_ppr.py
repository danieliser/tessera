"""Concurrent reindex + search test for PPR graph updates.

This test validates:
1. hybrid_search performs correctly under concurrent graph swaps
2. No race conditions or crashes during graph updates
3. Threading.Lock is actually exercised for graph mutations
4. P95 latency delta during reindex is minimal (<100ms)

Test architecture:
- Synthetic 1K+ symbol graph (sparse CSR matrix)
- Mock DB-like object with required interfaces
- 10 concurrent search threads each running 5 queries (50 total)
- 3 reindex cycles with random graph swaps (simulates DB updates)
- Measurements: P95 latency baseline vs during reindex
"""

import pytest
import time
import random
import threading
from typing import Dict, List, Tuple
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import scipy.sparse

from tessera.graph import ProjectGraph, ppr_to_ranked_list
from tessera.search import hybrid_search


# ==============================================================================
# Synthetic Graph Generation
# ==============================================================================

def create_synthetic_graph(n_symbols: int = 1500, edge_density: float = 0.02) -> ProjectGraph:
    """Create a synthetic 1K+ symbol graph.

    Args:
        n_symbols: Number of nodes in graph (default 1500)
        edge_density: Fraction of possible edges to create (~0.02 = 2% sparse)

    Returns:
        ProjectGraph instance with random topology
    """
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Create sparse adjacency matrix
    num_edges = int(n_symbols * n_symbols * edge_density)
    rows = rng.integers(0, n_symbols, num_edges)
    cols = rng.integers(0, n_symbols, num_edges)
    data = np.ones(num_edges, dtype=np.float32)

    adjacency = scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_symbols, n_symbols),
        dtype=np.float32,
    )

    # Create mappings
    symbol_id_to_name = {i: f"symbol_{i}" for i in range(n_symbols)}
    id_to_idx = {i: i for i in range(n_symbols)}

    graph = ProjectGraph(
        project_id=1,
        adjacency_matrix=adjacency,
        symbol_id_to_name=symbol_id_to_name,
        loaded_at=time.perf_counter(),
        id_to_idx=id_to_idx,
    )

    return graph


def create_swapped_graph(base_graph: ProjectGraph, perturbation: float = 0.1) -> ProjectGraph:
    """Create a perturbed version of the base graph for reindex simulation.

    Modifies ~10% of edges randomly to simulate a fresh reindex with small changes.

    Args:
        base_graph: Original ProjectGraph
        perturbation: Fraction of edges to modify (default 0.1 = 10%)

    Returns:
        New ProjectGraph with perturbed adjacency matrix
    """
    rng = np.random.default_rng()

    # Copy adjacency and apply random perturbations
    adj_copy = base_graph.graph.copy()
    nnz = adj_copy.nnz

    # Randomly remove/add ~10% of edges
    n_perturb = max(1, int(nnz * perturbation))
    for _ in range(n_perturb):
        # Random row/col
        i, j = rng.integers(0, base_graph.n_symbols, 2)
        if adj_copy[i, j] > 0:
            adj_copy[i, j] = 0
        else:
            adj_copy[i, j] = 1.0

    adj_copy = adj_copy.tocsr()

    new_graph = ProjectGraph(
        project_id=1,
        adjacency_matrix=adj_copy,
        symbol_id_to_name=base_graph.symbol_id_to_name.copy(),
        loaded_at=time.perf_counter(),
        id_to_idx=base_graph.id_to_idx.copy(),
    )

    return new_graph


# ==============================================================================
# Mock Database
# ==============================================================================

class MockProjectDB:
    """Mock DB-like object that implements required interfaces for hybrid_search."""

    def __init__(self, n_chunks: int = 500):
        """Initialize mock DB.

        Args:
            n_chunks: Number of mock chunks (default 500)
        """
        self.n_chunks = n_chunks
        self.chunks = {}
        self.embeddings_cache = None
        self.lock = threading.Lock()

        # Initialize mock chunk data
        for chunk_id in range(n_chunks):
            self.chunks[chunk_id] = {
                "id": chunk_id,
                "file_path": f"src/module_{chunk_id % 50}.py",
                "start_line": chunk_id * 10,
                "end_line": chunk_id * 10 + 9,
                "content": f"def func_{chunk_id}(): pass",
                "source_type": "code",
                "symbol_ids": f"[{chunk_id % 100}, {(chunk_id + 1) % 100}]",
            }

        # Create embeddings: random 768-dim vectors
        rng = np.random.default_rng(42)
        self.embeddings_array = rng.normal(0, 1, (n_chunks, 768)).astype(np.float32)

        # Build symbol to chunks mapping
        self._symbol_to_chunks = {}
        for chunk_id in range(n_chunks):
            sym_ids = [chunk_id % 100, (chunk_id + 1) % 100]
            for sym_id in sym_ids:
                if sym_id not in self._symbol_to_chunks:
                    self._symbol_to_chunks[sym_id] = []
                self._symbol_to_chunks[sym_id].append(chunk_id)

    def keyword_search(
        self, query: str, limit: int = 10, source_type: List[str] = None
    ) -> List[Dict]:
        """Mock keyword search returning top-k results.

        Returns list of dicts with 'id' and 'score' fields.
        """
        # Simulate search by returning deterministic mock results based on query hash
        query_hash = hash(query) % self.n_chunks
        results = []
        for i in range(min(limit, 5)):  # Return up to 5 results
            chunk_id = (query_hash + i) % self.n_chunks
            results.append({
                "id": chunk_id,
                "score": 1.0 - (i * 0.15),  # Decay score
            })
        return results

    def get_all_embeddings(self) -> Tuple[List[int], np.ndarray]:
        """Return all chunk embeddings.

        Returns:
            (chunk_ids list, embedding matrix as 2D numpy array)
        """
        chunk_ids = list(range(self.n_chunks))
        return chunk_ids, self.embeddings_array.copy()

    def get_chunk(self, chunk_id: int) -> Dict:
        """Get chunk metadata by ID.

        Returns dict with file_path, start_line, end_line, content, source_type keys.
        """
        return self.chunks.get(chunk_id, {})

    def get_symbol_to_chunks_mapping(self) -> Dict[int, List[int]]:
        """Get symbol ID to chunk IDs mapping.

        Returns:
            Dict mapping {symbol_id: [chunk_id, ...]}
        """
        return self._symbol_to_chunks.copy()


# ==============================================================================
# Test Class
# ==============================================================================

@pytest.mark.performance
class TestConcurrentPPR:
    """Concurrent reindex + search test suite."""

    def test_concurrent_search_with_reindex(self):
        """Test hybrid_search with concurrent graph reindex cycles.

        Validates:
        - 10 concurrent search threads
        - 5 queries per thread (50 total)
        - 3 overlapping reindex cycles with graph swaps
        - No crashes, no exceptions
        - P95 latency delta < 100ms
        - Lock is exercised during graph swaps
        """
        # Setup
        base_graph = create_synthetic_graph(n_symbols=1500)
        mock_db = MockProjectDB(n_chunks=500)
        lock_acquires = []
        reindex_count = [0]

        # Shared state: current graph + lock (mimic server.py pattern)
        graph_state = {"graph": base_graph}

        # Custom lock wrapper to track acquisitions
        class TrackedLock:
            def __init__(self):
                self._lock = threading.Lock()
                self.acquire_count = 0

            def __enter__(self):
                self.acquire_count += 1
                lock_acquires.append(time.perf_counter())
                self._lock.__enter__()
                return self

            def __exit__(self, *args):
                return self._lock.__exit__(*args)

        graph_lock = TrackedLock()

        # Timing
        search_latencies_baseline = []
        search_latencies_during_reindex = []

        # Queries to execute
        test_queries = [
            "function definition",
            "async handler",
            "database query",
            "error handling",
            "type annotations",
        ]

        def run_search_worker(worker_id: int, queries: List[str]) -> List[float]:
            """Worker thread: run 5 searches and measure latency.

            Args:
                worker_id: Thread ID for logging
                queries: List of search queries

            Returns:
                List of latency measurements (seconds)
            """
            latencies = []
            for query in queries:
                # Get current graph (might change during search)
                with graph_lock:
                    current_graph = graph_state["graph"]

                # Measure search latency
                start = time.perf_counter()
                try:
                    # Create query embedding (random)
                    query_embedding = np.random.randn(768).astype(np.float32)

                    results = hybrid_search(
                        query=query,
                        query_embedding=query_embedding,
                        db=mock_db,
                        graph=current_graph,
                        limit=10,
                        source_type=None,
                    )

                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)

                    # Track if in reindex phase
                    if reindex_count[0] > 0:
                        search_latencies_during_reindex.append(elapsed)
                    else:
                        search_latencies_baseline.append(elapsed)

                    assert isinstance(results, list)
                    assert all(isinstance(r, dict) for r in results)

                except Exception as e:
                    pytest.fail(f"Worker {worker_id} search failed: {e}")

            return latencies

        def reindex_cycle(cycle_id: int):
            """Reindex cycle: simulate graph update with lock.

            Args:
                cycle_id: Cycle ID for logging
            """
            time.sleep(random.uniform(0.01, 0.05))  # Random offset
            reindex_count[0] += 1

            try:
                # Create new perturbed graph
                new_graph = create_swapped_graph(graph_state["graph"], perturbation=0.1)

                # Swap with lock (critical section, should be recorded)
                with graph_lock:
                    graph_state["graph"] = new_graph

            finally:
                reindex_count[0] -= 1

        # Launch: 10 search workers + 3 reindex cycles
        with ThreadPoolExecutor(max_workers=13) as executor:
            futures = []

            # Submit 10 search workers
            for worker_id in range(10):
                query_subset = test_queries * (worker_id % 2 + 1)  # 5 or 10 queries each
                future = executor.submit(run_search_worker, worker_id, query_subset)
                futures.append(("search", worker_id, future))

            # Submit reindex cycles with slight delay
            time.sleep(0.01)
            for cycle_id in range(3):
                time.sleep(random.uniform(0.005, 0.02))
                future = executor.submit(reindex_cycle, cycle_id)
                futures.append(("reindex", cycle_id, future))

            # Collect results
            exceptions = []
            for task_type, task_id, future in futures:
                try:
                    result = future.result(timeout=30)
                    if task_type == "search":
                        assert isinstance(result, list), f"Worker {task_id} returned non-list"
                except Exception as e:
                    exceptions.append((task_type, task_id, e))

        # Validate: no exceptions
        assert len(exceptions) == 0, f"Concurrent execution failed: {exceptions}"

        # Validate: lock was actually used
        assert len(lock_acquires) >= 6, (
            f"Lock not exercised enough (acquires={len(lock_acquires)}, "
            f"expected >=6 for 3 reindex swaps + searches)"
        )

        # Validate: latencies collected
        assert len(search_latencies_baseline) > 0, "No baseline latencies recorded"
        assert len(search_latencies_during_reindex) > 0, "No reindex latencies recorded"

        # Calculate P95 latency
        baseline_p95 = np.percentile(search_latencies_baseline, 95) * 1000  # ms
        reindex_p95 = np.percentile(search_latencies_during_reindex, 95) * 1000  # ms
        delta_p95 = reindex_p95 - baseline_p95

        print(f"\n=== Concurrent PPR Test Results ===")
        print(f"Lock acquisitions: {len(lock_acquires)}")
        print(f"Baseline searches: {len(search_latencies_baseline)}")
        print(f"Reindex searches: {len(search_latencies_during_reindex)}")
        print(f"Baseline P95 latency: {baseline_p95:.2f}ms")
        print(f"Reindex P95 latency: {reindex_p95:.2f}ms")
        print(f"P95 delta: {delta_p95:.2f}ms")

        # Assert: P95 delta < 100ms (lock contention acceptable)
        assert delta_p95 < 100.0, (
            f"P95 latency delta {delta_p95:.2f}ms exceeds 100ms threshold"
        )

    def test_graph_is_sparse_fallback_not_triggered(self):
        """Ensure synthetic graph is NOT marked as sparse fallback.

        This validates that PPR computation actually runs (not skipped).
        """
        graph = create_synthetic_graph(n_symbols=1500)

        # Graph should have reasonable edge count
        assert graph.n_symbols > 1000
        assert graph.edge_count > graph.n_symbols  # Should NOT be sparse fallback

        # Verify is_sparse_fallback returns False
        assert not graph.is_sparse_fallback(), (
            f"Graph incorrectly marked sparse: {graph.n_symbols} symbols, "
            f"{graph.edge_count} edges"
        )

    def test_mock_db_interfaces(self):
        """Validate mock DB implements all required interfaces.

        Required by hybrid_search:
        - keyword_search(query, limit, source_type)
        - get_all_embeddings() -> (chunk_ids, embeddings_2d)
        - get_chunk(chunk_id) -> dict with file_path, start_line, end_line, content, source_type
        - get_symbol_to_chunks_mapping() -> {symbol_id: [chunk_ids]}
        """
        db = MockProjectDB(n_chunks=500)

        # Test keyword_search
        results = db.keyword_search("test query", limit=10)
        assert isinstance(results, list)
        assert all("id" in r and "score" in r for r in results)

        # Test get_all_embeddings
        chunk_ids, embeddings = db.get_all_embeddings()
        assert isinstance(chunk_ids, list)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (500, 768)
        assert embeddings.dtype == np.float32

        # Test get_chunk
        chunk = db.get_chunk(0)
        assert "file_path" in chunk
        assert "start_line" in chunk
        assert "end_line" in chunk
        assert "content" in chunk
        assert "source_type" in chunk

        # Test get_symbol_to_chunks_mapping
        mapping = db.get_symbol_to_chunks_mapping()
        assert isinstance(mapping, dict)
        assert all(isinstance(k, int) and isinstance(v, list) for k, v in mapping.items())

    def test_hybrid_search_with_graph(self):
        """Test hybrid_search integration with graph and mock DB.

        Ensures PPR path is exercised and produces valid results.
        """
        graph = create_synthetic_graph(n_symbols=1500)
        db = MockProjectDB(n_chunks=500)

        query_embedding = np.random.randn(768).astype(np.float32)

        results = hybrid_search(
            query="function definition",
            query_embedding=query_embedding,
            db=db,
            graph=graph,
            limit=10,
            source_type=None,
        )

        # Validate results
        assert isinstance(results, list)
        assert len(results) > 0

        # Check required fields
        for result in results:
            assert "id" in result
            assert "file_path" in result
            assert "start_line" in result
            assert "end_line" in result
            assert "content" in result
            assert "score" in result
            assert "rank_sources" in result
            assert "source_type" in result

        # Verify graph was used
        assert any("graph" in r.get("rank_sources", []) for r in results), (
            "PPR graph not used in ranking (expected 'graph' in rank_sources)"
        )

    @pytest.mark.parametrize("n_workers", [5, 10, 20])
    def test_concurrent_searches_varying_workers(self, n_workers: int):
        """Test concurrent searches with varying worker counts.

        Validates thread safety across different concurrency levels.

        Args:
            n_workers: Number of concurrent search threads
        """
        graph = create_synthetic_graph(n_symbols=1500)
        db = MockProjectDB(n_chunks=500)

        results_list = []
        exceptions = []

        def search_worker(worker_id: int):
            try:
                for _ in range(3):
                    query_embedding = np.random.randn(768).astype(np.float32)
                    results = hybrid_search(
                        query=f"query_{worker_id}",
                        query_embedding=query_embedding,
                        db=db,
                        graph=graph,
                        limit=10,
                    )
                    results_list.append(results)
            except Exception as e:
                exceptions.append((worker_id, e))

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(search_worker, i)
                for i in range(n_workers)
            ]
            for future in as_completed(futures):
                future.result(timeout=30)

        # No exceptions
        assert len(exceptions) == 0, f"Exceptions during concurrent searches: {exceptions}"

        # All searches returned results
        assert len(results_list) == n_workers * 3
        assert all(isinstance(r, list) for r in results_list)
