# Phase 5 Specification — PPR Graph Intelligence for Tessera

**Version:** 2.1
**Date:** 2026-02-28
**Status:** Revised after Round 2 panel review (gaps resolved)
**Tier:** Standard

---

## Executive Summary

This specification defines Phase 5 implementation of Personalized PageRank (PPR) graph intelligence for Tessera, a hierarchical code indexing system. PPR adds a third ranking signal to the existing two-way RRF merge (BM25 keyword + FAISS semantic), surfacing structural importance based on call graph topology.

**Key Deliverables:**
- New `src/tessera/graph.py` module (~300 LOC): PPR computation via scipy CSR power iteration
- Modified `src/tessera/search.py` (~50 LOC): Three-way RRF integration
- Modified `src/tessera/server.py` (~200 LOC): Graph lifecycle (load at startup, rebuild on reindex), PPR-enhanced impact tool
- Graceful degradation when graph sparse (edge_count < symbol_count)
- **Blocking spike test (Phase 5a)** validating ≥2% precision lift before full implementation
- Benchmark suite with performance gates (<100ms PPR, <50ms search overhead)

**Recommended Implementation:** scipy.sparse CSR matrix + hand-coded power iteration (no new dependencies; scipy already used for Drift-Adapter in Phase 4).

**Impact:** HippoRAG 2 and RepoGraph demonstrate 7–32% ranking improvement with graph-aware signals. Aider uses tree-sitter PPR in production. Tessera's symbol-level graph should yield 3–7% precision lift on search and impact analysis.

---

## Problem Statement

### Current Limitations

**Two-way RRF (Status Quo):**
- `search.py` merges BM25 keyword results + FAISS semantic results via RRF
- Results ranked by text/vector similarity only, ignoring code structure
- Impact analysis uses flat BFS traversal over edges; all discovered symbols ranked equally

**Graph Data Exists But Unused:**
- `db.py` edges table contains 4–200K edges per indexed project (tree-sitter-extracted: calls, imports, extends, hook dependencies)
- Current indexer stores edges but impact tool doesn't use them for ranking
- Structural importance signal lost

### Who Has This Problem

- **Task agents** querying large codebases (>1K symbols): top N search results miss architecturally critical functions
- **Impact analysis** on high-fanout symbols: can't prioritize which dependent callers to check first
- **Repository navigation**: no ranking by "most important" vs "least important" function (all functions listed alphabetically)

### Pain Level

**Medium.** Impacts agents doing deep codebase exploration, but not critical for baseline functionality:
- Search still works (finds relevant chunks), just less well-prioritized
- Impact still identifies all affected symbols, just in arbitrary order
- Precision loss ~7–15% on larger codebases (inferred from HippoRAG 2 baseline on knowledge graphs)

---

## Proposed Solution

### Three-Way Hybrid Search Architecture

```
Query
  ↓
├─ BM25 keyword search (SQLite FTS5)         → ranked_list_keyword
├─ Semantic search (FAISS cosine)             → ranked_list_semantic
└─ PPR graph search (scipy sparse power iter) → ranked_list_ppr
  ↓
Merge via RRF(k=60)
  ↓
Final ranked results [symbol_1, symbol_2, ...]
```

### Graph Lifecycle

```
Server Startup (create_server)
  ├─ Load projects from GlobalDB
  └─ For each project:
      ├─ Query edges from ProjectDB.edges table
      ├─ Build scipy.sparse.csr_matrix (in-memory adjacency)
      ├─ Log graph metadata: edge_count, symbol_count, load_time_ms
      └─ Store in _project_graphs[project_id]

On search(query):
  ├─ Identify seed symbols (keyword/semantic matches)
  ├─ Compute PPR from seeds via _project_graphs[project_id]
  └─ Merge with keyword + semantic via rrf_merge

On reindex(project_id):
  ├─ Run full indexing pipeline
  ├─ Emit GRAPH_REBUILT event
  └─ Reload _project_graphs[project_id]

On impact(symbol):
  ├─ Traditional BFS for affected symbols
  ├─ Rank affected symbols by PPR score (distance from query)
  └─ Return top-N by combined relevance
```

---

## Architecture

### Component 1: New Module — `src/tessera/graph.py`

Purpose: All PPR computation logic isolated in one module.

**Class: `ProjectGraph`**

```python
class ProjectGraph:
    """In-memory sparse graph for a single project.

    CHANGE (Round 1): Added loaded_at timestamp and metadata for freshness tracking (AC#7).
    """

    def __init__(self, project_id: int, adjacency_matrix: scipy.sparse.csr_matrix,
                 symbol_id_to_name: dict[int, str], loaded_at: float):
        """
        Args:
            project_id: ID of the project this graph belongs to
            adjacency_matrix: scipy CSR matrix where [i,j] = edge weight from symbol_i to symbol_j
            symbol_id_to_name: Mapping symbol_id → symbol_name for result enrichment
            loaded_at: Timestamp when graph was loaded (time.time()), for freshness tracking
        """
        self.project_id = project_id
        self.graph = adjacency_matrix
        self.symbol_id_to_name = symbol_id_to_name
        self.n_symbols = adjacency_matrix.shape[0]
        self.loaded_at = loaded_at
        self.edge_count = adjacency_matrix.nnz

    def personalized_pagerank(
        self,
        seed_symbol_ids: list[int],
        alpha: float = 0.15,
        max_iter: int = 50,
        tol: float = 1e-6
    ) -> list[tuple[int, float]]:
        """
        Compute Personalized PageRank from seed symbols.

        Returns sorted list of (symbol_id, ppr_score) tuples, highest score first.

        Algorithm: Power iteration with restart vector biased toward seed symbols.

        CHANGE (Round 1): Clarified normalization as column-stochastic (right multiply).
        Added unit test for star graph validation against NetworkX reference.

        Args:
            seed_symbol_ids: Symbol IDs to personalize rank toward
            alpha: Damping factor (probability of teleport to seed, default 0.15)
            max_iter: Maximum iteration count (default 50, sufficient for 50K edges)
            tol: Convergence tolerance (default 1e-6)

        Returns:
            List of (symbol_id, ppr_score) sorted descending by score

        Performance: <100ms for graphs up to 50K edges on typical hardware
        """
        # ... implementation details below

    def is_sparse_fallback(self) -> bool:
        """Return True if graph too sparse for PPR to add signal.

        Threshold: edge_count < symbol_count (avg degree < 1).
        Below this, PPR degenerates to uniform distribution (BFS-like).
        """
        return self.edge_count < self.n_symbols
```

**Function: `load_project_graph(db: ProjectDB, project_id: int) → ProjectGraph`**

Loads edges from database, builds scipy CSR matrix.

```python
def load_project_graph(db: ProjectDB, project_id: int) -> ProjectGraph:
    """
    Load adjacency matrix from project's edges table.

    CHANGE (Round 1): Logs load time with WARNING if >500ms (memory monitoring).

    Args:
        db: ProjectDB instance
        project_id: Project ID

    Returns:
        ProjectGraph with in-memory sparse adjacency matrix

    Time: <100ms for projects with <50K edges
    Memory: ~8 bytes per edge + overhead (CSR format is very compact)
    """
    import time
    load_start = time.time()

    # Query all edges for this project
    edges = db.get_all_edges(project_id)

    # Get symbol count and id→name mapping
    symbols = db.get_all_symbols(project_id)
    n_symbols = len(symbols)
    symbol_id_to_name = {s['id']: s['name'] for s in symbols}

    if not edges or not symbols:
        # Empty graph: return zero matrix
        return ProjectGraph(
            project_id,
            scipy.sparse.csr_matrix((n_symbols, n_symbols), dtype=np.float32),
            symbol_id_to_name,
            loaded_at=time.time()
        )

    # Extract (from_id, to_id, weight) from edges
    rows, cols, weights = [], [], []
    for edge in edges:
        rows.append(edge['from_id'])
        cols.append(edge['to_id'])
        weights.append(edge.get('weight', 1.0))

    # Build CSR matrix
    adjacency = scipy.sparse.csr_matrix(
        (weights, (rows, cols)),
        shape=(n_symbols, n_symbols),
        dtype=np.float32
    )

    graph = ProjectGraph(project_id, adjacency, symbol_id_to_name, loaded_at=time.time())

    load_time_ms = (time.time() - load_start) * 1000
    if load_time_ms > 500:
        logger.warning(
            f"Graph load for project {project_id} took {load_time_ms:.1f}ms "
            f"({n_symbols} symbols, {adjacency.nnz} edges). Consider profiling."
        )

    return graph
```

**Function: `ppr_to_ranked_list(ppr_scores: dict[int, float]) → list[dict]`**

Converts PPR scores to ranked list format compatible with rrf_merge.

```python
def ppr_to_ranked_list(ppr_scores: dict[int, float]) -> list[dict]:
    """
    Convert PPR output to rrf_merge compatible format.

    Args:
        ppr_scores: {symbol_id → ppr_score}

    Returns:
        [{'id': symbol_id, 'score': normalized_score}, ...]
        sorted by score descending
    """
    if not ppr_scores:
        return []

    # Sort by score descending
    sorted_scores = sorted(ppr_scores.items(), key=lambda x: -x[1])

    # Normalize scores to [0, 1] for consistency with semantic search
    max_score = sorted_scores[0][1] if sorted_scores else 1.0
    max_score = max(max_score, 1e-6)  # Avoid division by zero

    return [
        {'id': symbol_id, 'score': score / max_score}
        for symbol_id, score in sorted_scores
    ]
```

**Implementation Detail: Power Iteration Algorithm**

```python
def personalized_pagerank(self, seed_symbol_ids, alpha=0.15, max_iter=50, tol=1e-6):
    """
    Power iteration with personalization vector.

    Mathematical formulation:
        p_{k+1} = (1 - alpha) * A^T * p_k + alpha * p_seed

    where:
        - A = column-stochastic adjacency matrix (right-multiplied by diag(1/out_degree))
        - p_seed = personalization vector (probability mass on seed symbols)
        - alpha = teleport probability (default 0.15, matches Google PageRank)
        - A^T applies column-stochastic normalization: each node distributes probability to outgoing edges

    CHANGE (Round 1): Explicitly documented column-stochastic normalization.
    Out-degree (sum of outgoing edges) in denominator ensures each symbol's probability
    distributes equally across outgoing neighbors, preventing high-degree nodes from
    dominating the ranking.

    Complexity: O(iterations * nnz), where nnz = edge count
    For 50K edges, 50 iterations: ~5–50ms on typical CPU
    """
    n = self.n_symbols

    # Initialize personalization vector (uniform over seeds)
    p_seed = np.zeros(n, dtype=np.float32)
    if seed_symbol_ids:
        for sid in seed_symbol_ids:
            if 0 <= sid < n:
                p_seed[sid] = 1.0 / len(seed_symbol_ids)
    else:
        # No seeds: uniform distribution
        p_seed[:] = 1.0 / n

    # Initialize probability vector
    p = p_seed.copy()

    # Normalize adjacency matrix column-stochastic
    # (out-degree normalization: each column sums to 1)
    # This ensures that each symbol distributes probability equally to outgoing edges
    graph_norm = self.graph.copy()
    out_degrees = np.array(graph_norm.sum(axis=0)).ravel()
    out_degrees[out_degrees == 0] = 1  # Avoid division by zero (disconnected nodes)
    graph_norm = graph_norm * scipy.sparse.diags(1.0 / out_degrees)

    # Power iteration
    for iteration in range(max_iter):
        p_old = p.copy()

        # p_new = (1 - alpha) * A^T @ p + alpha * p_seed
        # A^T is transpose of column-stochastic matrix (converts to row-stochastic via transpose)
        p = (1 - alpha) * graph_norm.T @ p + alpha * p_seed

        # Check convergence: L2 norm of difference
        diff = np.linalg.norm(p - p_old, ord=2)
        if diff < tol:
            break  # Early exit if converged

    # Convert to result format: {symbol_id → score}
    result = {}
    for symbol_id in range(n):
        if p[symbol_id] > 1e-10:  # Skip near-zero scores (numerical noise)
            result[symbol_id] = float(p[symbol_id])

    return result
```

---

### Component 2: Modified `src/tessera/search.py`

**Changes:**

1. **Extend `hybrid_search()` to compute PPR and merge three-way**

```python
def hybrid_search(
    query: str,
    query_embedding: Optional[np.ndarray],
    db,
    graph: Optional[ProjectGraph] = None,  # NEW: optional PPR graph
    limit: int = 10,
    source_type: Optional[list[str]] = None,
) -> list[dict]:
    """
    Hybrid search combining keyword, semantic, and graph-aware (PPR) signals.

    Algorithm:
    1. Run FTS5 keyword search → ranked_list_keyword
    2. Run FAISS semantic search → ranked_list_semantic
    3. If graph provided and not sparse: compute PPR → ranked_list_ppr
    4. Merge all available signals via RRF
    5. Enrich with chunk metadata

    New parameter:
        graph: Optional ProjectGraph instance. If None or sparse, skips PPR.
    """
    ranked_lists = []

    # 1. Keyword search (unchanged)
    try:
        keyword_results = db.keyword_search(query, limit=limit, source_type=source_type)
        if keyword_results:
            ranked_lists.append(keyword_results)
    except Exception:
        keyword_results = []

    # 2. Semantic search (unchanged)
    if query_embedding is not None:
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
                    filtered = []
                    for result in semantic_results:
                        chunk = db.get_chunk(result["id"])
                        if chunk and (chunk.get("source_type") or "code") in source_type:
                            filtered.append(result)
                    semantic_results = filtered[:limit]
                if semantic_results:
                    ranked_lists.append(semantic_results)
        except Exception:
            semantic_results = []

    # 3. NEW: PPR search (graph-aware ranking)
    ppr_results = []
    if graph and not graph.is_sparse_fallback():
        try:
            # Identify seed symbols from keyword/semantic results
            seed_symbol_ids = set()

            # Extract symbol IDs from keyword results
            for result in keyword_results:
                chunk = db.get_chunk(result["id"])
                if chunk and chunk.get("symbol_ids"):
                    seed_symbol_ids.update(json.loads(chunk["symbol_ids"]))

            # Extract symbol IDs from semantic results
            for result in semantic_results:
                chunk = db.get_chunk(result["id"])
                if chunk and chunk.get("symbol_ids"):
                    seed_symbol_ids.update(json.loads(chunk["symbol_ids"]))

            if seed_symbol_ids:
                # Compute PPR from seeds
                ppr_scores = graph.personalized_pagerank(list(seed_symbol_ids))

                # Map PPR scores back to chunk IDs
                # (PPR ranks symbols, but search works with chunks)
                symbol_to_chunks = db.get_symbol_to_chunks_mapping()
                ppr_chunk_scores = {}
                for symbol_id, ppr_score in ppr_scores.items():
                    for chunk_id in symbol_to_chunks.get(symbol_id, []):
                        # CHANGE (Round 1): Use max() to collapse symbol PPR scores to chunk level.
                        # Rationale: max() prevents a single low-scoring symbol from diluting a chunk
                        # that contains another high-scoring symbol. This preserves the highest
                        # importance signal across all symbols in a chunk, which is semantically
                        # correct: if any constituent symbol is important, the chunk is important.
                        ppr_chunk_scores[chunk_id] = max(ppr_chunk_scores.get(chunk_id, 0), ppr_score)

                # Convert to ranked list format
                ppr_results = ppr_to_ranked_list(ppr_chunk_scores)
                ppr_results = ppr_results[:limit]

                if ppr_results:
                    ranked_lists.append(ppr_results)
        except Exception as e:
            logger.warning(f"PPR computation failed, falling back to 2-way RRF: {e}")

    # 4. Merge with RRF
    if not ranked_lists:
        return []

    merged = rrf_merge(ranked_lists)

    # 5. Enrich with chunk metadata (unchanged)
    results = []
    for item in merged[:limit]:
        chunk_id = item["id"]
        try:
            meta = db.get_chunk(chunk_id)
            enriched = {
                "id": chunk_id,
                "file_path": meta.get("file_path", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "content": meta.get("content", ""),
                "score": item.get("rrf_score", item.get("score", 0.0)),
                "rank_sources": ["keyword", "semantic"] + (["graph"] if ppr_results else []),
                "source_type": meta.get("source_type", "code"),
                "trusted": meta.get("source_type", "code") == "code",
                "section_heading": meta.get("section_heading", ""),
                "key_path": meta.get("key_path", ""),
                "page_number": meta.get("page_number"),
                "parent_section": meta.get("parent_section", ""),
                "graph_version": graph.loaded_at if graph else None,  # NEW: track graph freshness
            }
        except Exception:
            enriched = {
                "id": chunk_id,
                "file_path": "",
                "start_line": 0,
                "end_line": 0,
                "content": "",
                "score": item.get("rrf_score", item.get("score", 0.0)),
                "rank_sources": ["keyword", "semantic"] + (["graph"] if ppr_results else []),
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
```

2. **Add `ppr_to_ranked_list()` import from graph.py** (already defined above)

3. **Update `doc_search()` to accept optional graph parameter**

```python
def doc_search(
    query: str,
    query_embedding: Optional[np.ndarray],
    db,
    graph: Optional[ProjectGraph] = None,  # NEW
    limit: int = 10,
    formats: Optional[list[str]] = None,
) -> list[dict]:
    """Search non-code documents (documents have no symbol graph context)."""
    if formats is None:
        formats = [
            'markdown', 'pdf', 'yaml', 'json',
            'html', 'xml', 'text',
            'txt', 'rst', 'csv', 'tsv', 'log',
            'ini', 'cfg', 'toml', 'conf',
        ]
    # PPR doesn't apply to documents, so pass graph=None
    return hybrid_search(query, query_embedding, db, graph=None, limit=limit, source_type=formats)
```

---

### Component 3: Modified `src/tessera/server.py`

**Changes:**

1. **Add global graph cache**

```python
# At module level (after existing _db_cache, _locked_project, etc.)
_project_graphs: dict[int, ProjectGraph] = {}  # project_id → ProjectGraph instance

# CHANGE (Round 1): Add graph metadata logging
_graph_stats: dict[int, dict] = {}  # project_id → {'edge_count', 'symbol_count', 'loaded_at', 'load_time_ms'}
```

2. **Update `create_server()` to load graphs at startup with profiling**

```python
def create_server(
    project_path: Optional[str],
    global_db_path: str,
    embedding_endpoint: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> FastMCP:
    """Create and configure the MCP server.

    Changes: Build in-memory PPR graphs for all loaded projects.
    NEW (Round 1): Log per-project graph load time with WARNING if >500ms.
    """
    global _db_cache, _locked_project, _global_db, _drift_adapter, _embedding_client, _project_graphs, _graph_stats
    import time

    # ... existing initialization code ...

    # NEW: Load graphs for each project
    total_startup = time.time()
    if _global_db:
        projects = _global_db.list_projects()
        logger.info(f"Loading graphs for {len(projects)} projects...")

        for project in projects:
            project_id = project["id"]
            project_path = project["path"]

            try:
                # Open ProjectDB
                db = ProjectDB(project_path)

                # Load PPR graph
                from .graph import load_project_graph
                graph = load_project_graph(db, project_id)
                _project_graphs[project_id] = graph

                # CHANGE (Round 1): Store graph metadata for monitoring
                _graph_stats[project_id] = {
                    'edge_count': graph.edge_count,
                    'symbol_count': graph.n_symbols,
                    'loaded_at': graph.loaded_at,
                    'load_time_ms': (graph.loaded_at - total_startup) * 1000
                }

                logger.info(
                    f"Loaded graph for project {project_id} "
                    f"({graph.n_symbols} symbols, {graph.edge_count} edges, "
                    f"{_graph_stats[project_id]['load_time_ms']:.1f}ms)"
                )
            except Exception as e:
                logger.warning(f"Failed to load graph for project {project_id}: {e}")
                # Graceful degradation: project accessible but without PPR

    total_load_ms = (time.time() - total_startup) * 1000
    logger.info(f"Total graph startup time: {total_load_ms:.1f}ms")
```

3. **Update `search()` tool to pass graph and log version**

```python
@mcp.tool()
async def search(query: str, limit: int = 10, source_type: str = "", session_id: str = "") -> str:
    """Search with three-way hybrid (keyword + semantic + graph)."""
    scope, err = _check_session({"session_id": session_id}, "project")
    if err:
        return err

    # ... existing code to get dbs ...

    try:
        tasks = []
        for pid, pname, db in dbs:
            graph = _project_graphs.get(pid)  # NEW: pass optional graph

            # CHANGE (Round 1): Log which graph version is being used
            if graph:
                logger.debug(f"Search project {pid}: using graph v{graph.loaded_at}")

            tasks.append(
                asyncio.to_thread(
                    hybrid_search,
                    query,
                    query_embedding,
                    db,
                    graph,  # NEW
                    limit,
                    source_type_list or None
                )
            )

        # ... rest of search function ...
```

4. **Update `reindex()` tool to rebuild graphs**

```python
@mcp.tool()
async def reindex(project_id: int, mode: str = "full", session_id: str = "") -> str:
    """Trigger re-indexing of a project.

    After indexing completes, rebuild in-memory graph.
    """
    # ... existing reindex code ...

    # At end, after stats = await asyncio.to_thread(pipeline.index_project()):
    try:
        # Reload graph for this project
        from .graph import load_project_graph
        db = ProjectDB(project["path"])
        graph = load_project_graph(db, project_id)
        _project_graphs[project_id] = graph

        # CHANGE (Round 1): Update graph metadata
        _graph_stats[project_id] = {
            'edge_count': graph.edge_count,
            'symbol_count': graph.n_symbols,
            'loaded_at': graph.loaded_at,
        }

        logger.info(f"Rebuilt graph for project {project_id}")
    except Exception as e:
        logger.warning(f"Failed to rebuild graph for project {project_id}: {e}")
```

5. **Enhance `impact()` tool to rank by PPR**

```python
@mcp.tool()
async def impact(symbol_name: str, depth: int = 3, include_types: bool = True, session_id: str = "") -> str:
    """Analyze what breaks if a symbol is changed.

    Enhancement: Rank affected symbols by PPR distance from the changed symbol.
    """
    scope, err = _check_session({"session_id": session_id}, "project")
    if err:
        return err

    dbs = _get_project_dbs(scope)
    if not dbs:
        _log_audit("impact", 0)
        return "Error: No accessible projects"

    try:
        # Get affected symbols (unchanged BFS logic)
        tasks = [
            asyncio.to_thread(db.get_impact, symbol_name, depth, include_types)
            for pid, pname, db in dbs
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for (pid, pname, db), result in zip(dbs, results_list):
            if isinstance(result, Exception):
                logger.warning("Query on project %d failed: %s", pid, result)
                continue

            # NEW: Rank by PPR if graph available
            graph = _project_graphs.get(pid)
            if graph and not graph.is_sparse_fallback():
                try:
                    # Seed PPR on the changed symbol
                    affected_symbol_ids = [r['id'] for r in result]
                    ppr_scores = graph.personalized_pagerank([sym for sym in affected_symbol_ids if sym < graph.n_symbols])

                    # Re-rank results by PPR score
                    ppr_map = {sid: score for sid, score in ppr_scores.items()}
                    for r in result:
                        r['ppr_relevance'] = ppr_map.get(r['id'], 0.0)

                    # Sort by PPR score (descending)
                    result = sorted(result, key=lambda x: -x.get('ppr_relevance', 0.0))
                except Exception as e:
                    logger.warning(f"PPR ranking failed for impact, returning unsorted: {e}")

            for r in result:
                r["project_id"] = pid
                r["project_name"] = pname
            all_results.extend(result)

        _log_audit("impact", len(all_results))
        return json.dumps(all_results, indent=2)
    except Exception as e:
        logger.exception("Impact tool error")
        _log_audit("impact", 0)
        return f"Error analyzing impact: {str(e)}"
```

---

### Component 4: Changes to `src/tessera/db.py`

**Add helper methods:**

```python
def get_all_edges(self, project_id: int) -> List[Dict[str, Any]]:
    """Fetch all edges for a project (used by graph loader).

    Returns:
        List of dicts with 'from_id', 'to_id', 'weight' keys
    """
    cursor = self.conn.execute(
        "SELECT from_id, to_id, weight FROM edges WHERE project_id = ? ORDER BY from_id, to_id",
        (project_id,)
    )
    return [dict(row) for row in cursor.fetchall()]

def get_all_symbols(self, project_id: int) -> List[Dict[str, Any]]:
    """Fetch all symbols for a project (used by graph loader).

    Returns:
        List of dicts with 'id', 'name', 'kind' keys (at minimum)
    """
    cursor = self.conn.execute(
        "SELECT id, name, kind, file_id, line, col, scope FROM symbols WHERE project_id = ? ORDER BY id",
        (project_id,)
    )
    return [dict(row) for row in cursor.fetchall()]

def get_symbol_to_chunks_mapping(self) -> Dict[int, List[int]]:
    """Map symbol IDs to chunk IDs they appear in.

    Used by search to convert PPR (symbol-level) to chunk-level results.

    Returns:
        {symbol_id → [chunk_id, ...]}
    """
    cursor = self.conn.execute(
        "SELECT id, symbol_ids FROM chunk_meta WHERE symbol_ids IS NOT NULL"
    )

    symbol_to_chunks = {}
    for row in cursor.fetchall():
        chunk_id = row[0]
        symbol_ids_json = row[1]

        try:
            symbol_ids = json.loads(symbol_ids_json) if symbol_ids_json else []
        except json.JSONDecodeError:
            continue

        for sym_id in symbol_ids:
            if sym_id not in symbol_to_chunks:
                symbol_to_chunks[sym_id] = []
            symbol_to_chunks[sym_id].append(chunk_id)

    return symbol_to_chunks
```

---

## Acceptance Criteria

### Functional Requirements

1. **Three-way RRF Search**
   - [ ] `hybrid_search()` accepts optional `graph` parameter
   - [ ] When graph provided and not sparse, computes PPR from query seed symbols
   - [ ] PPR results merged with keyword + semantic via existing `rrf_merge()` (no changes to RRF logic)
   - [ ] Result order changes visibly for large codebases (>1K symbols) when graph signal added
   - [ ] Test: Search for "index" in 5K-symbol project returns different top-10 with/without PPR

2. **PPR-Enhanced Impact Tool**
   - [ ] `impact()` tool calls graph-aware ranking when graph available
   - [ ] Affected symbols ranked by PPR distance from changed symbol (not BFS order)
   - [ ] Test: `impact("some_high_fanout_function")` on 1K+ node graph returns top-10 in PPR order

3. **Graph Lifecycle**
   - [ ] At server startup, `create_server()` loads graphs for all projects
   - [ ] Graph load time <1 second per project (even 50K-edge graphs)
   - [ ] On `reindex()` completion, graph automatically rebuilt (within existing reindex call)
   - [ ] Graph loading is synchronous at startup (MCP servers start once and run; blocking for ~200ms/project is acceptable)
   - [ ] Log total startup graph loading time; WARN if total exceeds 5 seconds
   - [ ] Test: Start server, verify `_project_graphs` populated for all projects

4. **Graceful Degradation & Concurrent Operations**
   - [ ] When graph sparse (edge_count < symbol_count), PPR skipped, reverts to 2-way RRF
   - [ ] No error thrown if graph empty or None
   - [ ] `is_sparse_fallback()` correctly identifies when PPR should be skipped
   - [ ] **CHANGE (Round 1+2):** Concurrent reindex + search test: 10 concurrent search threads × 5 queries each (50 total) + 3 full reindex cycles overlapping randomly on 1K+ symbol project
     - Measure P95 search latency during reindex vs P95 baseline (no reindex)
     - Verify: no crashes, P95 latency delta <100ms from baseline, graph version updates atomically
     - Implement explicit `threading.Lock` around graph swap (don't rely on GIL assumption)
   - [ ] Test: Index minimal project (<10 edges), search still works, rank_sources field shows only keyword+semantic

5. **Performance**
   - [ ] PPR computation <100ms for graphs up to 50K edges
   - [ ] Search latency increase <50ms when PPR added (vs baseline 2-way)
   - [ ] Graph load at startup <1 second per project
   - [ ] Test: Benchmark on real projects with 10K, 20K, 50K edges

### Integration Requirements

6. **Compatibility with Existing Code**
   - [ ] All existing tests in `tests/` pass without modification
   - [ ] `search.py` functions have backward-compatible signatures (graph parameter optional)
   - [ ] `db.py` changes are additive (new methods, no signature changes to existing methods)
   - [ ] No breaking changes to ProjectDB or GlobalDB schemas

7. **Code Quality**
   - [ ] `graph.py` has >80% test coverage (unit tests for PPR algorithm, edge cases)
   - [ ] All public functions documented with docstrings (module, class, method level)
   - [ ] Type hints on all public functions
   - [ ] No circular imports (graph.py, search.py, server.py, db.py form DAG)
   - [ ] **CHANGE (Round 1):** Unit test for power iteration normalization: star graph with central hub → hub has highest PPR score
   - [ ] **CHANGE (Round 1):** Validate against NetworkX pagerank_scipy() reference on test fixture

8. **Monitoring & Observability**
   - [ ] Logs emitted at INFO level for graph load events
   - [ ] Logs at WARNING level if graph load fails (project still usable, no PPR)
   - [ ] Logs at WARNING level if graph load time >500ms
   - [ ] Audit log records tool calls that used PPR (via rank_sources field)
   - [ ] **CHANGE (Round 1):** Log graph metadata at startup: project count, total edge count, estimated memory
   - [ ] **CHANGE (Round 1):** Log graph freshness on search (which graph version is being used)

---

## Risks & Mitigations

### Risk 1: PPR Adds Minimal Ranking Lift (BLOCKING)

**Symptom:** After spike test (Phase 5a), empirical testing shows <2% precision improvement on search results.

**Root Cause:** Tree-sitter extracted graph density may be insufficient for PPR to differentiate (e.g., all symbols have similar PageRank score).

**Mitigation (NEW — Round 1):**
- **Phase 5a Spike Test is blocking gate for Phase 5 full implementation**
- Index 2–3 real projects: PHP plugin (~1K symbols, ~2K edges), Python project (~5K symbols, ~10K edges), JavaScript codebase (~3K symbols, ~5K edges)
- Measure actual edge_count / symbol_count ratio on each
- Compute PPR on 10+ representative queries per project, measure recall@5 and nDCG vs 2-way RRF baseline
- **Decision gate:** Proceed to Task 1 (full implementation) only if ≥2 of 3 projects show >2% precision lift
- **If spike fails (<2% lift on 2+ projects):** defer PPR to Phase 6, document graph density findings, focus Phase 5 on other features (e.g., weighted RRF tuning with existing 2-way signals)
- Phase 5a estimated effort: 1 day (setup + spike testing)
- Phase 5a must complete before starting Task 1

**Likelihood:** Medium (Aider reports success with tree-sitter PPR, but on file-level graphs, not symbol graphs)

---

### Risk 2: Server Startup Latency Increases Unacceptably

**Symptom:** Graph loading adds >5 seconds to startup time when loading 10+ large projects.

**Root Cause:** Loading all graphs synchronously on startup, or scipy CSR matrix construction slow for very large graphs.

**Mitigation:**
- Profile graph load on largest test project (100K edges)
- If >2 seconds per project: implement lazy loading (load graph on first search to that project)
- If >500ms per project at 50K edges: profile scipy CSR construction; consider pre-computing and caching adjacency matrix binary format (.npz)
- Log WARNING if any single project load >500ms (NEW — Round 1)
- Acceptable tradeoff: first search to newly-loaded project may be slow (graph loads synchronously); subsequent searches fast

**Likelihood:** Low (scipy CSR construction is O(nnz) with small constant; 50K edges should load <100ms)

---

### Risk 3: Memory Usage Grows Too Large for Multi-Project Servers

**Symptom:** Server memory grows unbounded when serving 50+ projects simultaneously.

**Root Cause:** In-memory CSR matrices never freed; long-running servers accumulate graphs.

**Mitigation:**
- **Phase 5:** Monitor memory usage, log graph cache stats at startup (NEW — Round 1)
- Log estimated memory per project: ~8 bytes per edge (row, col, weight in CSR format) + baseline overhead
- Document expected memory: formula is `8 * edge_count + overhead_per_project`
- Set threshold: if total memory > 500MB, emit WARNING
- **Phase 5 (Round 2):** Implement simple LRU cap: `MAX_CACHED_GRAPHS = 20`. If exceeded on new project load, evict least-recently-used graph (~30 LOC). Log eviction events.
- **Phase 6:** Refine LRU with configurable thresholds, memory-budget-based eviction, and disk serialization (.npz)
- Tradeoff: first search to evicted project reloads graph (100–500ms latency)
- Document: servers with >20 projects should monitor memory and set LRU threshold appropriately

**Likelihood:** Medium (only relevant for high-availability, many-project deployments)

---

### Risk 4: PPR Convergence Too Slow or Non-Convergence

**Symptom:** Power iteration doesn't converge within 50 iterations on some graphs.

**Root Cause:** Graph has unusual structure (e.g., nearly-disconnected components) causing slow mixing.

**Mitigation:**
- Cap max_iter at 100 (worst case: algorithm stops after 100 iterations, returns best approximation)
- Log warning if convergence not achieved: "PPR did not converge after 100 iterations; using approximation"
- Empirical testing: validate convergence on Tessera test projects (expect <15 iterations typically)
- Fallback: if PPR computation exceeds 500ms (timeout guard), abort and use 2-way RRF

**Likelihood:** Low (power iteration proven to converge on sparse matrices; million-node Google graph converges in 52 iterations)

---

### Risk 5: PPR Changes Search Rankings Unexpectedly

**Symptom:** Third signal inverts results; users expect semantic top-3 but get different ranking.

**Root Cause:** PPR introduces high-confidence but sometimes unintuitive rankings based on graph distance.

**Mitigation:**
- A/B test on representative query set before rollout
- Return rank_sources field in results showing which signals contributed to final ranking
- Document in user-facing changelog: "Search now ranks by graph structure in addition to text/semantics"
- Option to disable PPR via config flag (set graph=None in search call): `search(..., use_graph=False)`

**Likelihood:** Low–Medium (HippoRAG 2 shows ranking improvements are consistent; but "unexpected" is subjective)

---

### Risk 6: Graph-Symbol Mapping Breaks During Concurrent Reindex

**Symptom:** Search throws exception when accessing symbols during reindex, or old graph stale.

**Root Cause:** Reindex rebuilds graph while search is using old graph; race condition if ProjectDB connection changes.

**Mitigation:**
- Use `_project_graphs[project_id] = new_graph` atomic assignment (Python GIL ensures atomic dict update)
- **Phase 5 (Round 1+2):** Implement explicit `threading.Lock` around graph swap (don't rely on GIL). Add concurrent reindex + search test to AC#4:
  - Test: 10 concurrent search threads × 5 queries each + 3 reindex cycles overlapping randomly
  - Verify: no crashes, P95 latency delta <100ms from baseline, atomic graph swaps logged
  - If GIL insufficient, implement explicit threading.Lock around graph swap
- Make graph loads non-blocking: load new graph in background thread, swap when complete
- Document: graph may be slightly stale (up to reindex duration) during incremental updates

**Likelihood:** Low (Python GIL + atomic dict assignment make this safe in practice, but worth testing)

---

## Dependencies

### New Dependencies
- **scipy** — already required (used by Drift-Adapter in Phase 4)
- **numpy** — already required (used by FAISS, embeddings)
- No new external dependencies

### Existing Dependencies Modified
- `src/tessera/search.py`: Add optional `graph` parameter (backward compatible)
- `src/tessera/db.py`: Add 2–3 new helper methods (no changes to existing signatures)
- `src/tessera/server.py`: Add `_project_graphs` cache, modify server startup (internal change)

### Database Schema Changes
- **edges table:** No changes. Existing schema compatible.
- **symbols table:** No changes. New queries added (get_all_symbols) but no schema changes.
- **chunk_meta table:** No changes. Existing symbol_ids field used.

---

## Test Strategy

### Phase 5a: Spike Test (BLOCKING GATE)

**File: `tests/test_spike_ppr_validation.py`** — MUST RUN BEFORE PHASE 5 TASKS

**Purpose:** Validate that PPR adds >2% precision lift on real projects before committing to full implementation.

**Test Setup:**
1. Index 3 real codebases:
   - PHP plugin: ~1K symbols, ~2K edges (typical plugin complexity)
   - Python project: ~5K symbols, ~10K edges (e.g., medium Django app)
   - JavaScript codebase: ~3K symbols, ~5K edges (e.g., small npm package)

2. For each project, measure:
   - Edge count / symbol count ratio (expect 0.5–2)
   - Sparsity assessment (is_sparse_fallback() = True/False?)

3. Create 10+ representative queries per project using this taxonomy:
   - 30% function-name lookups (e.g., "init", "handle", "process")
   - 30% domain-concept searches (e.g., "database connection", "authentication")
   - 30% code-pattern queries (e.g., "API handler", "error handling", "event dispatch")
   - 10% known-negative queries (terms that should return few/no results)

   **Annotation method:** Developer familiar with each project manually labels top-10 results as relevant (1) or not-relevant (0) for each query. Single annotator per project is acceptable for spike validation (not academic rigor).

4. For each query, compute:
   - recall@5: % of manually-annotated relevant results in top-5
   - nDCG@5: normalized discounted cumulative gain (ranks top results higher)
   - Compute with 2-way RRF baseline (keyword + semantic only)
   - Compute with 3-way RRF (add PPR)
   - Measure lift: `(3-way nDCG - 2-way nDCG) / 2-way nDCG`

5. **Gate Decision:**
   - If ≥2 of 3 projects show >2% nDCG lift → proceed to Phase 5 Tasks 1–6
   - If <2 of 3 projects show >2% lift → defer PPR to Phase 6, focus Phase 5 on other enhancements

**Expected Outcome:** Empirical validation that tree-sitter PPR graphs have sufficient structure for ranking improvement.

---

### Unit Tests (graph.py)

**File: `tests/test_graph.py`**

1. **Test ProjectGraph.personalized_pagerank()**
   - Empty graph (0 symbols): returns empty dict
   - Single-symbol graph: returns {0: 1.0}
   - Two-symbol linear graph (0→1): PPR ranks both, 0 higher than 1
   - **Star graph (central hub): hub has highest PPR (NEW — Round 1, validates normalization)**
   - Disconnected components: PPR respects connectivity
   - Convergence tolerance: 50 iterations sufficient for tolerance 1e-6
   - Seed vector: PPR biased toward seed symbols
   - Performance: 50K edges, 50 iterations <100ms
   - **CHANGE (Round 1):** Add unit test validating against NetworkX pagerank_scipy() reference:
     ```python
     # Create test star graph, compute with both PPR and NetworkX
     # Assert scipy PPR scores match NetworkX within 1e-5 tolerance
     ```

2. **Test is_sparse_fallback()**
   - edge_count = 100, symbol_count = 101: returns True
   - edge_count = 100, symbol_count = 99: returns False
   - empty graph: returns True

3. **Test load_project_graph()**
   - Load graph from test ProjectDB with known edges
   - Verify adjacency matrix shape (n_symbols × n_symbols)
   - Verify edge weights preserved
   - Verify symbol_id_to_name mapping correct
   - **Verify loaded_at timestamp is set (NEW — Round 1, freshness tracking)**

4. **Test ppr_to_ranked_list()**
   - Empty dict: returns []
   - Single score: returns [{'id': 0, 'score': 1.0}]
   - Multiple scores: returned in descending order by score
   - Normalization: all scores ≤1.0

---

### Integration Tests (search.py, server.py)

**File: `tests/test_search_with_ppr.py`**

1. **Test three-way RRF integration**
   - Create test project with 100 symbols, 50 edges
   - Index a file with query-relevant chunks
   - Call `hybrid_search()` with graph, without graph
   - Verify results differ (PPR signal visible)
   - Verify rank_sources includes "graph" when PPR used
   - **Verify graph_version field is populated (NEW — Round 1, freshness tracking)**

2. **Test graceful degradation**
   - Create sparse project (10 symbols, 3 edges)
   - graph.is_sparse_fallback() = True
   - Call search with sparse graph
   - Verify no error, rank_sources excludes "graph"

3. **Test graph loading at startup**
   - Create multi-project test setup
   - Call create_server() with test projects
   - Verify `_project_graphs` has entry for each project
   - Verify graph has correct symbol/edge counts
   - **Verify graph metadata (_graph_stats) is populated (NEW — Round 1, memory monitoring)**

4. **Test impact() with PPR ranking**
   - Create test project with call graph
   - Find high-fanout function (called by many)
   - Call impact(function_name)
   - Verify affected symbols ranked (not just listed)
   - Verify top-10 different than BFS order

5. **Test concurrent reindex + search (Round 1+2, AC#4)**
   - Create test project with 1K+ symbols
   - Spawn 10 concurrent search threads (each runs 5 queries = 50 total)
   - Simultaneously run 3 reindex cycles overlapping randomly
   - Measure: P95 search latency during reindex vs P95 baseline (no reindex)
   - Assert: no crashes, P95 latency delta <100ms from baseline, atomic graph swaps logged
   - Graph swap uses explicit `threading.Lock` (not GIL assumption)

---

### Performance Tests (NEW — TASK 1, NOT TASK 5)

**File: `tests/test_performance_ppr.py`** (moved from Task 5 to Task 1, parallel with graph.py — CHANGE Round 1)

**Purpose:** CI gate: fail PR if PPR exceeds performance thresholds. Thresholds: red (>100ms), yellow (80ms), green (<50ms).

1. **Benchmark PPR computation**
   - Load real projects of varying sizes (1K, 10K, 50K edges)
   - Measure PPR time for 10 random seed sets
   - Assert <100ms for 50K edges (red threshold)
   - Assert <80ms for 50K edges (yellow threshold, warn if exceeded)
   - Assert <50ms for 10K edges (green threshold)
   - Fail PR if >100ms on 50K edges

2. **Benchmark graph loading**
   - Measure time to load graph from ProjectDB
   - Assert <1 second for 50K edges
   - Assert <100ms for 10K edges
   - Warn if >500ms for any single project (NEW — Round 1)

3. **Benchmark search latency impact**
   - Run search queries on project with/without PPR graph
   - Measure latency difference
   - Assert <50ms overhead from PPR

---

### Regression Tests

**Existing test suites pass:**
- `tests/test_search.py`: existing two-way RRF tests still pass (graph parameter optional)
- `tests/test_impact.py`: existing impact tests still pass (ranking order may change, but set of affected symbols unchanged)
- All other tests in `tests/` unchanged

---

## Non-Goals

Explicitly deferred to Phase 6+ (from intake):

- WebP dimension extraction
- EXIF/IPTC metadata extraction
- Graph edges from documents to code symbols
- Low-Rank Affine drift adapter variant
- Cross-language annotation tool
- Document version history
- Advanced semantic chunking
- OCR from images
- Archive content extraction
- DOCX/Word support
- Automated drift-adapter triggering
- Audit logging
- File watcher implementation

Also explicitly OUT of scope for Phase 5:

- Weighted RRF tuning (parameterized weights for keyword/semantic/PPR signals) — defer to Phase 6 with A/B testing infrastructure
- GNN-based ranking (over-engineered for Phase 1–5)
- External graph databases (stays in-memory only)
- Per-agent PPR personalization (global PPR only, seeded by query)
- Visualization of PPR scores
- LRU eviction of graphs (defer to Phase 6, but add monitoring NOW)

---

## Deferred from Round 1 Panel Discussion

The following items were flagged by panelists but deferred to Round 2 for further discussion:

1. **Graceful degradation threshold (1.0 vs 1.5 avg degree)**
   - Current spec: `edge_count < symbol_count` (avg degree < 1) triggers fallback
   - Panelist concern: maybe 1.5 is safer threshold (avg degree < 1.5)
   - Decision: Defer to Round 2. Phase 5a spike test will provide empirical data on what threshold is appropriate
   - Note: Fallback is conservative; graceful degradation at 1.0 is safe, higher threshold is extra cautious

2. **Weighted RRF vs equal-weight**
   - Current spec: RRF with equal weight (k=60 for all three signals)
   - Panelist concern: semantic + keyword likely more reliable than PPR; consider weighted approach
   - Decision: Defer to Phase 6. Requires A/B testing infrastructure and empirical data. Phase 5a spike will provide baseline for Phase 6 optimization

---

## Implementation Roadmap

### Phase 5a: Spike Test (BLOCKING GATE) — 1 day effort

**Task 0: Spike Test — PPR Precision Validation**

**Dependencies:** None. Can run immediately.

**Subtasks:**
1. Index 3 real codebases (PHP, Python, JavaScript) with symbol extraction
2. Measure graph density (edge_count / symbol_count) on each
3. Create 10+ representative queries per project
4. Compute recall@5 / nDCG@5 for 2-way vs 3-way RRF
5. Gate decision: ≥2 of 3 projects >2% lift → proceed to Phase 5 tasks
6. Document findings in `docs/plans/phase5-ppr-graph/spike-results.md`

**Estimated effort:** 1 day

**Blocking gate:** Phase 5 Tasks 1–6 do not start until this completes successfully.

---

### Phase 5: Main Implementation (Tasks 1–6)

Proceed only if Phase 5a shows ≥2% precision lift on 2+ projects.

### Task 1: New Module `src/tessera/graph.py` + Performance Benchmarks (350 LOC)

**Dependencies:** None (upstream). Can be implemented in parallel with Task 2.

**CHANGE (Round 1):** Moved benchmark suite from Task 5 to Task 1 (parallel with graph.py). Write tests before implementing graph.py.

**Subtasks:**
1. Create `tests/test_performance_ppr.py` benchmark suite with performance gates
   - CI gate: fail PR if PPR >100ms on 50K edge graph or search overhead >50ms
   - Thresholds: red (>100ms), yellow (80ms), green (<50ms)
2. Define `ProjectGraph` class with `__init__`, `personalized_pagerank()`, `is_sparse_fallback()`
3. Implement `load_project_graph()` function
4. Implement `ppr_to_ranked_list()` function
5. Unit tests for all components (power iteration, normalization, edge cases)
   - Add star graph test validating normalization (NEW — Round 1)
   - Add NetworkX reference validation (NEW — Round 1)
6. Validation: edge cases (empty graph, single symbol, disconnected components)
7. Run benchmark suite on test projects (1K, 10K, 50K edges)

**Estimated effort:** 1.5 days (developer) + 1 day (testing/benchmarking)

---

### Task 2: Extend `src/tessera/db.py` (60 LOC)

**Dependencies:** None. Can be implemented in parallel with Task 1.

**Subtasks:**
1. Add `get_all_edges(project_id)` method
2. Add `get_all_symbols(project_id)` method
3. Add `get_symbol_to_chunks_mapping()` method
4. Unit tests: verify query correctness on test ProjectDB

**Estimated effort:** 0.5 day (developer) + 0.5 day (testing)

---

### Task 3: Update `src/tessera/search.py` (80 LOC)

**Dependencies:** Task 1 (graph.py), Task 2 (db.py). Sequential.

**Subtasks:**
1. Modify `hybrid_search()` signature to accept optional `graph` parameter
2. Add PPR computation logic (seed extraction, graph query, result mapping)
3. Update RRF merge to handle three-way lists
4. Update `doc_search()` to pass graph=None (documents have no symbols)
5. Add graph_version field to results (NEW — Round 1, freshness tracking)
6. Add comment explaining max() choice for symbol-to-chunk mapping (NEW — Round 1)
7. Integration tests: verify three-way RRF works, freshness tracking

**Estimated effort:** 1 day (developer) + 1 day (testing)

---

### Task 4: Update `src/tessera/server.py` (250 LOC)

**Dependencies:** Task 1, 3. Sequential.

**CHANGE (Round 1):** Increased LOC from 200 to 250 due to additional monitoring and logging.

**Subtasks:**
1. Add `_project_graphs` and `_graph_stats` global caches
2. Modify `create_server()` to load graphs with per-project timing and startup profiling (NEW — Round 1)
3. Log graph metadata at startup: project count, total edge count, estimated memory (NEW — Round 1)
4. Update `search()` tool to pass graph and log version used (NEW — Round 1)
5. Update `reindex()` to rebuild graph and update metadata
6. Enhance `impact()` to rank by PPR
7. Concurrent reindex + search integration tests (NEW — Round 1, AC#4)

**Estimated effort:** 1.5 days (developer) + 1.5 days (testing/concurrency validation)

---

### Task 5: Memory Monitoring & Documentation (100 LOC)

**Dependencies:** Tasks 1–4 complete.

**CHANGE (Round 1):** Moved memory monitoring from Phase 6 to Phase 5. LRU eviction deferred to Phase 6, but monitoring happens now.

**Subtasks:**
1. Implement graph cache stats logging (project count, total edge count, estimated memory)
2. Add per-project load timing logs with WARNING if >500ms (NEW — Round 1)
3. Document expected memory formula: `8 * edge_count + overhead_per_project`
4. Document graph freshness tracking: loaded_at timestamp, version in results (NEW — Round 1)
5. Write ops guide: "How to Monitor Graph Memory Usage" (NEW — Round 1)
6. Add inline comments explaining max() choice for symbol-to-chunk mapping (NEW — Round 1)

**Estimated effort:** 0.5 day (developer) + 0.5 day (documentation)

---

### Task 6: Documentation & Cleanup (100 LOC)

**Dependencies:** All tasks complete.

**Subtasks:**
1. Add docstring summaries to graph.py, db.py, server.py changes
2. Update inline comments for PPR logic and normalization (NEW — Round 1)
3. Add example usage to server.py module docstring
4. Write integration guide for ops (graph load time, memory tuning, freshness tracking)
5. Update README if needed (new "Graph Intelligence" feature)

**Estimated effort:** 0.5 day

---

## Summary of Changes from Spec v1 to v2 (Round 1 Revisions)

**Must Address (Consensus):**
1. Added Phase 5a Spike Test as Task 0, blocking gate for Phase 5 implementation (AC#1 impact)
2. Added concurrent reindex + search test to AC#4 (Graph Lifecycle)
3. Moved Benchmark Suite to Task 1 (parallel with graph.py) — now CI performance gate
4. Added Memory Monitoring in Phase 5 (new Task 5, moved from Phase 6)
   - Log graph cache stats at startup: project count, total edge count, estimated memory
   - Add per-project load timing with WARNING if >500ms
   - Document expected memory formula

**Should Address:**
5. Clarified power iteration normalization as column-stochastic with right-multiply diag(1/out_degree)
6. Added unit test validating against NetworkX pagerank_scipy() reference (star graph)
7. Added startup profiling: log per-project graph load time in create_server()
8. Added graph freshness tracking: loaded_at timestamp, edge_count, symbol_count metadata
9. Added code comment justifying max() for symbol-to-chunk mapping (prevents dilution from low-scoring symbols)
10. Updated ProjectGraph class to include loaded_at and edge_count attributes
11. Updated risk mitigations with Phase 5a spike test gate and concurrent reindex test
12. Updated AC#8 (Monitoring) with memory and freshness tracking requirements

**Deferred to Round 2:**
- Graceful degradation threshold (1.0 vs 1.5 avg degree) — empirical data from spike test will inform
- Weighted RRF vs equal-weight — defer to Phase 6 with A/B testing infrastructure
