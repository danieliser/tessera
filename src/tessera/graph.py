"""Graph module for Tessera — Personalized PageRank and impact analysis.

Provides:
  - ProjectGraph: CSR sparse matrix wrapper with PPR computation
  - load_project_graph: Load graph from ProjectDB with symbol ID mapping
  - ppr_to_ranked_list: Convert PPR scores to ranked list format
"""

import logging
import time
from typing import Dict, List, Optional
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

logger = logging.getLogger(__name__)

# Simple LRU cap; configurable LRU in Phase 6
MAX_CACHED_GRAPHS = 20


def evict_lru_graph(graphs: dict[int, "ProjectGraph"], max_size: int = MAX_CACHED_GRAPHS) -> Optional[int]:
    """Evict least-recently-loaded graph if cache exceeds max_size.

    Returns evicted project_id, or None if no eviction needed.
    """
    if len(graphs) <= max_size:
        return None

    # Find LRU by loaded_at timestamp (oldest = least recently used)
    oldest_pid = min(graphs, key=lambda pid: graphs[pid].loaded_at)
    evicted = graphs.pop(oldest_pid)
    logger.info(
        "Evicted graph for project %d (loaded_at=%.1f, %d symbols, %d edges). "
        "Cache size: %d/%d",
        oldest_pid, evicted.loaded_at, evicted.n_symbols, evicted.edge_count,
        len(graphs), max_size
    )
    return oldest_pid


class ProjectGraph:
    """Sparse graph representation with Personalized PageRank computation.

    Stores a CSR sparse adjacency matrix and metadata for a project's
    call/reference graph. Supports efficient PPR computation via power iteration.
    """

    def __init__(
        self,
        project_id: int,
        adjacency_matrix: scipy.sparse.csr_matrix,
        symbol_id_to_name: Dict[int, str],
        loaded_at: float,
        id_to_idx: Dict[int, int] = None,
    ):
        """Initialize ProjectGraph.

        Args:
            project_id: ID of the project in ProjectDB
            adjacency_matrix: CSR sparse matrix where [i,j] = weight from i to j.
                Shape is (n_symbols, n_symbols).
            symbol_id_to_name: Mapping from original symbol IDs to symbol names.
                Keys are original symbol IDs (not matrix indices).
            loaded_at: Timestamp when graph was loaded (perf_counter seconds).
            id_to_idx: Mapping from original symbol ID to matrix index.
                If not provided, assumes symbol IDs are 0-based contiguous.
        """
        self.project_id = project_id
        self.graph = adjacency_matrix
        self.symbol_id_to_name = symbol_id_to_name
        self.loaded_at = loaded_at
        self.n_symbols = adjacency_matrix.shape[0]
        self.edge_count = adjacency_matrix.nnz
        self.id_to_idx = id_to_idx or {sid: idx for idx, sid in enumerate(sorted(symbol_id_to_name.keys()))}
        self.idx_to_id = {idx: sid for sid, idx in self.id_to_idx.items()}

        # Compute largest connected component size for sparse threshold
        self.largest_cc_size = self._compute_largest_cc_size()

    def _compute_largest_cc_size(self) -> int:
        """Compute the size of the largest connected component.

        Uses scipy's connected_components on the undirected version of the graph.
        Cost is O(n + m), negligible for code-scale graphs.
        """
        if self.n_symbols == 0:
            return 0
        # Treat as undirected for connectivity analysis
        n_components, labels = scipy.sparse.csgraph.connected_components(
            self.graph, directed=False, return_labels=True
        )
        if n_components == 0:
            return 0
        # Count nodes in each component, return the largest
        component_sizes = np.bincount(labels)
        return int(component_sizes.max())

    @property
    def estimated_memory_bytes(self) -> int:
        """Estimate memory usage of this graph.

        CSR format: ~12 bytes per non-zero (4 bytes col index + 4 bytes row ptr amortized + 4 bytes data)
        Plus overhead for arrays and Python objects.
        """
        return self.edge_count * 12 + self.n_symbols * 8 + 1024  # base overhead

    def is_sparse_fallback(self) -> bool:
        """Check if graph is too sparse for meaningful PPR.

        Uses adaptive threshold based on research (percolation theory,
        PageRank convergence analysis, real-world code graphs):
          - Density (edges/symbols) >= 0.75
          - Largest connected component >= 50% of symbols
          - Minimum 100 symbols for meaningful signal

        The 50% LCC threshold reflects real PHP/JS codebases where
        cross-file resolution typically connects 60-70% of symbols.

        Returns:
            True if graph doesn't meet PPR engagement thresholds.
        """
        if self.n_symbols < 100:
            return True
        density = self.edge_count / self.n_symbols
        if density < 0.75:
            return True
        cc_ratio = self.largest_cc_size / self.n_symbols
        if cc_ratio < 0.50:
            return True
        return False

    def personalized_pagerank(
        self,
        seed_symbol_ids: List[int],
        alpha: float = 0.15,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> Dict[int, float]:
        """Compute Personalized PageRank using power iteration.

        Computes PPR starting from seed symbols using power iteration:
          p_{k+1} = (1-alpha) * A_norm^T @ p_k + alpha * p_seed

        Where A_norm is column-stochastic normalized adjacency matrix.

        Args:
            seed_symbol_ids: List of original symbol IDs to personalize toward.
            alpha: Teleport probability (default 0.15).
            max_iter: Maximum power iteration iterations (default 50).
            tol: Convergence tolerance in L2 norm (default 1e-6).

        Returns:
            Dict mapping original symbol_id -> ppr_score for scores > 1e-10.
            Scores are unnormalized (sum to ~1.0 but may vary slightly).
        """
        if self.n_symbols == 0:
            return {}

        # Personalization vector: uniform over seed_symbol_ids
        p_seed = np.zeros(self.n_symbols, dtype=np.float32)
        unique_seeds = set(seed_symbol_ids)
        if not unique_seeds:
            return {}

        seed_count = len(unique_seeds)

        for sid in unique_seeds:
            # Map original symbol ID to matrix index
            if sid in self.id_to_idx:
                idx = self.id_to_idx[sid]
                p_seed[idx] = 1.0 / seed_count

        p = p_seed.copy()

        # Column-stochastic normalization: out_degrees = sum over rows
        # For CSR matrix, sum(axis=0) gives column sums, sum(axis=1) gives row sums
        # We want out-degree normalization: divide column j by sum of column j
        graph_norm = self.graph.copy().astype(np.float32)

        # Get out-degrees (row sums for forward graph)
        out_degrees = np.array(graph_norm.sum(axis=1)).ravel()
        out_degrees[out_degrees == 0] = 1.0  # Avoid division by zero

        # Normalize: multiply each row i by 1/out_degree[i]
        # This makes the matrix row-stochastic (rows sum to 1)
        graph_norm = scipy.sparse.diags(1.0 / out_degrees) @ graph_norm

        # Power iteration
        for iteration in range(max_iter):
            p_old = p.copy()
            # p_{k+1} = (1-alpha) * A^T @ p_k + alpha * p_seed
            p = (1 - alpha) * graph_norm.T @ p + alpha * p_seed

            # Check convergence in L2 norm
            if np.linalg.norm(p - p_old, ord=2) < tol:
                logger.debug(f"PPR converged after {iteration + 1} iterations")
                break

        # Convert to dict, filtering out near-zero scores
        # Map matrix indices back to original symbol IDs
        result = {}
        for idx in range(self.n_symbols):
            if p[idx] > 1e-10:
                # Map index back to original symbol ID
                if idx in self.idx_to_id:
                    sid = self.idx_to_id[idx]
                    result[sid] = float(p[idx])

        return result


def load_project_graph(db: "ProjectDB", project_id: int) -> ProjectGraph:
    """Load a project graph from ProjectDB.

    Fetches all symbols and edges, builds a CSR sparse matrix with proper
    ID-to-index mapping, and returns a ProjectGraph instance.

    Args:
        db: ProjectDB instance
        project_id: Project ID to load graph for

    Returns:
        ProjectGraph instance

    Raises:
        ValueError: If project has no symbols
    """
    load_start = time.perf_counter()

    # Get all symbols for this project
    symbols = db.get_all_symbols(project_id)
    if not symbols:
        raise ValueError(f"Project {project_id} has no symbols")

    # Create mapping from symbol ID to matrix index
    # IMPORTANT: Symbol IDs from database are NOT contiguous 0-based indices
    id_to_idx = {s["id"]: idx for idx, s in enumerate(symbols)}
    n_symbols = len(symbols)

    # Get all edges and build sparse matrix
    edges = db.get_all_edges(project_id)

    rows, cols, data = [], [], []
    for edge in edges:
        from_id = edge["from_id"]
        to_id = edge["to_id"]
        weight = edge.get("weight", 1.0) or 1.0

        # Map symbol IDs to matrix indices
        if from_id in id_to_idx and to_id in id_to_idx:
            rows.append(id_to_idx[from_id])
            cols.append(id_to_idx[to_id])
            data.append(float(weight))

    # Build CSR matrix
    if rows:
        adjacency = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_symbols, n_symbols),
            dtype=np.float32,
        )
    else:
        adjacency = scipy.sparse.csr_matrix(
            (n_symbols, n_symbols),
            dtype=np.float32,
        )

    # Build symbol ID to name mapping for ProjectGraph
    symbol_id_to_name = {s["id"]: s["name"] for s in symbols}

    load_time = time.perf_counter() - load_start
    if load_time > 0.5:  # 500ms
        logger.warning(
            f"Graph load took {load_time*1000:.1f}ms for project {project_id} "
            f"({n_symbols} symbols, {len(edges)} edges)"
        )

    # Return ProjectGraph with id_to_idx mapping
    graph = ProjectGraph(
        project_id=project_id,
        adjacency_matrix=adjacency,
        symbol_id_to_name=symbol_id_to_name,
        loaded_at=load_start,
        id_to_idx=id_to_idx,
    )

    return graph


def ppr_to_ranked_list(ppr_scores: Dict[int, float]) -> List[Dict]:
    """Convert PPR scores to ranked list format.

    Normalizes scores to [0, 1] and sorts descending by score.

    Args:
        ppr_scores: Dict mapping symbol_id -> score

    Returns:
        List of dicts [{'id': symbol_id, 'score': normalized_score}]
        sorted descending by score. Scores are normalized to [0, 1].
    """
    if not ppr_scores:
        return []

    # Normalize scores to [0, 1]
    max_score = max(ppr_scores.values())
    if max_score <= 0:
        max_score = 1.0

    normalized = [
        {"id": sid, "score": score / max_score}
        for sid, score in ppr_scores.items()
    ]

    # Sort descending by score
    normalized.sort(key=lambda x: x["score"], reverse=True)

    return normalized
