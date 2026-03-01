# Phase 5 Specification — PPR Graph Intelligence for Tessera

**Version:** 1.0
**Date:** 2026-02-28
**Status:** Ready for Implementation
**Tier:** Standard

---

## Executive Summary

This specification defines Phase 5 implementation of Personalized PageRank (PPR) graph intelligence for Tessera, a hierarchical code indexing system. PPR adds a third ranking signal to the existing two-way RRF merge (BM25 keyword + FAISS semantic), surface structural importance based on call graph topology.

**Key Deliverables:**
- New `src/tessera/graph.py` module (~300 LOC): PPR computation via scipy CSR power iteration
- Modified `src/tessera/search.py` (~50 LOC): Three-way RRF integration
- Modified `src/tessera/server.py` (~200 LOC): Graph lifecycle (load at startup, rebuild on reindex), PPR-enhanced impact tool
- Graceful degradation when graph sparse (edge_count < symbol_count)
- Benchmark suite validating <100ms performance gate on 50K edge graphs

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
    """In-memory sparse graph for a single project."""

    def __init__(self, project_id: int, adjacency_matrix: scipy.sparse.csr_matrix,
                 symbol_id_to_name: dict[int, str]):
        """
        Args:
            project_id: ID of the project this graph belongs to
            adjacency_matrix: scipy CSR matrix where [i,j] = edge weight from symbol_i to symbol_j
            symbol_id_to_name: Mapping symbol_id → symbol_name for result enrichment
        """
        self.project_id = project_id
        self.graph = adjacency_matrix
        self.symbol_id_to_name = symbol_id_to_name
        self.n_symbols = adjacency_matrix.shape[0]

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
        nnz = self.graph.nnz
        return nnz < self.n_symbols
```

**Function: `load_project_graph(db: ProjectDB, project_id: int) → ProjectGraph`**

Loads edges from database, builds scipy CSR matrix.

```python
def load_project_graph(db: ProjectDB, project_id: int) -> ProjectGraph:
    """
    Load adjacency matrix from project's edges table.

    Args:
        db: ProjectDB instance
        project_id: Project ID

    Returns:
        ProjectGraph with in-memory sparse adjacency matrix

    Time: <100ms for projects with <50K edges
    Memory: ~8 bytes per edge + overhead (CSR format is very compact)
    """
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
            symbol_id_to_name
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

    return ProjectGraph(project_id, adjacency, symbol_id_to_name)
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
        - A = normalized adjacency matrix (column-stochastic)
        - p_seed = personalization vector (probability mass on seed symbols)
        - alpha = teleport probability (default 0.15, matches Google PageRank)

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
    graph_norm = self.graph.copy()
    out_degrees = np.array(graph_norm.sum(axis=0)).ravel()
    out_degrees[out_degrees == 0] = 1  # Avoid division by zero
    graph_norm = graph_norm * scipy.sparse.diags(1.0 / out_degrees)

    # Power iteration
    for iteration in range(max_iter):
        p_old = p.copy()

        # p_new = (1 - alpha) * A^T @ p + alpha * p_seed
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
```

2. **Update `create_server()` to load graphs at startup**

```python
def create_server(
    project_path: Optional[str],
    global_db_path: str,
    embedding_endpoint: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> FastMCP:
    """Create and configure the MCP server.

    Changes: Build in-memory PPR graphs for all loaded projects.
    """
    global _db_cache, _locked_project, _global_db, _drift_adapter, _embedding_client, _project_graphs

    # ... existing initialization code ...

    # NEW: Load graphs for each project
    if _global_db:
        projects = _global_db.list_projects()
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

                logger.info(f"Loaded graph for project {project_id} ({graph.n_symbols} symbols, {graph.graph.nnz} edges)")
            except Exception as e:
                logger.warning(f"Failed to load graph for project {project_id}: {e}")
                # Graceful degradation: project accessible but without PPR
```

3. **Update `search()` tool to pass graph**

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
   - [ ] Server accessible and queryable during graph load (non-blocking)
   - [ ] Test: Start server, verify `_project_graphs` populated for all projects

4. **Graceful Degradation**
   - [ ] When graph sparse (edge_count < symbol_count), PPR skipped, reverts to 2-way RRF
   - [ ] No error thrown if graph empty or None
   - [ ] `is_sparse_fallback()` correctly identifies when PPR should be skipped
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

8. **Monitoring & Observability**
   - [ ] Logs emitted at INFO level for graph load events
   - [ ] Logs at WARNING level if graph load fails (project still usable, no PPR)
   - [ ] Audit log records tool calls that used PPR (via rank_sources field)

---

## Risks & Mitigations

### Risk 1: PPR Adds Minimal Ranking Lift

**Symptom:** After implementation, empirical testing shows <2% precision improvement on search results.

**Root Cause:** Tree-sitter extracted graph density may be insufficient for PPR to differentiate (e.g., all symbols have similar PageRank score).

**Mitigation:**
- Perform Phase 1 spike test on real 1K+ symbol project before final rollout
- Measure: recall@5, recall@10, nDCG on representative search queries
- If <3% lift observed, document as "graph density insufficient for this project" and defer weighted RRF tuning to Phase 6
- Decision gate: proceed to production if >3% lift on 2+ test projects

**Likelihood:** Medium (Aider reports success with tree-sitter PPR, but on file-level graphs, not symbol graphs)

---

### Risk 2: Server Startup Latency Increases Unacceptably

**Symptom:** Graph loading adds >5 seconds to startup time when loading 10+ large projects.

**Root Cause:** Loading all graphs synchronously on startup, or scipy CSR matrix construction slow for very large graphs.

**Mitigation:**
- Profile graph load on largest test project (100K edges)
- If >2 seconds per project: implement lazy loading (load graph on first search to that project)
- If >500ms per project at 50K edges: profile scipy CSR construction; consider pre-computing and caching adjacency matrix binary format (.npz)
- Acceptable tradeoff: first search to newly-loaded project may be slow (graph loads synchronously); subsequent searches fast

**Likelihood:** Low (scipy CSR construction is O(nnz) with small constant; 50K edges should load <100ms)

---

### Risk 3: Memory Usage Grows Too Large for Multi-Project Servers

**Symptom:** Server memory grows unbounded when serving 50+ projects simultaneously.

**Root Cause:** In-memory CSR matrices never freed; long-running servers accumulate graphs.

**Mitigation:**
- Implement LRU eviction: keep 10 most-recently-used project graphs in memory, evict others
- Profile memory per graph: expected ~8 bytes per edge (row, col, weight in CSR format)
- Set threshold: if total memory > 500MB, evict least-recently-used graph
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
- Make graph loads non-blocking: load new graph in background thread, swap when complete
- Document: graph may be slightly stale (up to reindex duration) during incremental updates
- Test: concurrent search + reindex on test project, verify no crashes

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

### Unit Tests (graph.py)

**File: `tests/test_graph.py`**

1. **Test ProjectGraph.personalized_pagerank()**
   - Empty graph (0 symbols): returns empty dict
   - Single-symbol graph: returns {0: 1.0}
   - Two-symbol linear graph (0→1): PPR ranks both, 0 higher than 1
   - Star graph (central hub): hub has highest PPR
   - Disconnected components: PPR respects connectivity
   - Convergence tolerance: 50 iterations sufficient for tolerance 1e-6
   - Seed vector: PPR biased toward seed symbols
   - Performance: 50K edges, 50 iterations <100ms

2. **Test is_sparse_fallback()**
   - edge_count = 100, symbol_count = 101: returns True
   - edge_count = 100, symbol_count = 99: returns False
   - empty graph: returns True

3. **Test load_project_graph()**
   - Load graph from test ProjectDB with known edges
   - Verify adjacency matrix shape (n_symbols × n_symbols)
   - Verify edge weights preserved
   - Verify symbol_id_to_name mapping correct

4. **Test ppr_to_ranked_list()**
   - Empty dict: returns []
   - Single score: returns [{'id': 0, 'score': 1.0}]
   - Multiple scores: returned in descending order by score
   - Normalization: all scores ≤1.0

### Integration Tests (search.py, server.py)

**File: `tests/test_search_with_ppr.py`**

1. **Test three-way RRF integration**
   - Create test project with 100 symbols, 50 edges
   - Index a file with query-relevant chunks
   - Call `hybrid_search()` with graph, without graph
   - Verify results differ (PPR signal visible)
   - Verify rank_sources includes "graph" when PPR used

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

4. **Test impact() with PPR ranking**
   - Create test project with call graph
   - Find high-fanout function (called by many)
   - Call impact(function_name)
   - Verify affected symbols ranked (not just listed)
   - Verify top-10 different than BFS order

**File: `tests/test_performance_ppr.py`** (benchmarks)

1. **Benchmark PPR computation**
   - Load real projects of varying sizes (1K, 10K, 50K edges)
   - Measure PPR time for 10 random seed sets
   - Assert <100ms for 50K edges
   - Assert <50ms for 10K edges

2. **Benchmark graph loading**
   - Measure time to load graph from ProjectDB
   - Assert <1 second for 50K edges
   - Assert <100ms for 10K edges

3. **Benchmark search latency impact**
   - Run search queries on project with/without PPR graph
   - Measure latency difference
   - Assert <50ms overhead from PPR

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

---

## Implementation Roadmap

### Task 1: New Module `src/tessera/graph.py` (300 LOC)

**Dependencies:** None (upstream). Can be implemented in parallel.

**Subtasks:**
1. Define `ProjectGraph` class with `__init__`, `personalized_pagerank()`, `is_sparse_fallback()`
2. Implement `load_project_graph()` function
3. Implement `ppr_to_ranked_list()` function
4. Unit tests for all three components
5. Validation: edge cases (empty graph, single symbol, disconnected components)

**Estimated effort:** 1 day (developer) + 0.5 day (testing)

---

### Task 2: Extend `src/tessera/db.py` (60 LOC)

**Dependencies:** None. Can be implemented in parallel.

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
5. Integration tests: verify three-way RRF works

**Estimated effort:** 1 day (developer) + 1 day (testing)

---

### Task 4: Update `src/tessera/server.py` (200 LOC)

**Dependencies:** Task 1, 3. Sequential.

**Subtasks:**
1. Add `_project_graphs` global cache
2. Modify `create_server()` to load graphs at startup
3. Update `search()` tool to pass graph to hybrid_search
4. Update `reindex()` to rebuild graph after indexing
5. Enhance `impact()` to rank by PPR
6. Integration tests: verify graph lifecycle

**Estimated effort:** 1.5 days (developer) + 1 day (testing)

---

### Task 5: Benchmarking & Validation (50 LOC)

**Dependencies:** All tasks complete. Sequential.

**Subtasks:**
1. Create `tests/test_performance_ppr.py` benchmark suite
2. Run on test projects: 1K, 10K, 50K edges
3. Validate <100ms gate for 50K edges
4. Validate <50ms latency overhead for search
5. Document results in `docs/plans/phase5-ppr-graph/benchmark-results.md`

**Estimated effort:** 1 day (developer) + 1 day (measurement/analysis)

---

### Task 6: Documentation & Cleanup (100 LOC)

**Dependencies:** All tasks complete.

**Subtasks:**
1. Add docstring summaries to graph.py, db.py changes
2. Update inline comments for PPR logic
3. Add example usage to server.py module docstring
4. Write integration guide for ops (graph load time, memory tuning)
5. Update README if needed (new "Graph Intelligence" feature)

**Estimated effort:** 0.5 day

---

## Total Effort Estimate

| Task | LOC | Developer Days | Testing Days | Total |
|------|-----|---|---|---|
| 1. graph.py | 300 | 1 | 0.5 | 1.5 |
| 2. db.py | 60 | 0.5 | 0.5 | 1 |
| 3. search.py | 80 | 1 | 1 | 2 |
| 4. server.py | 200 | 1.5 | 1 | 2.5 |
| 5. benchmarks | 50 | 1 | 1 | 2 |
| 6. docs | 100 | 0.5 | — | 0.5 |
| **Total** | **790** | **6** | **4** | **9** |

**Budget:** ~1,500 LOC total (Phase 5 scope from architecture spec).
**Actual:** ~790 LOC (significantly under budget; leaves room for error handling, optional features, unforeseen complexity).

---

## Success Metrics

1. **Functional Completion**
   - All acceptance criteria met
   - All tests passing (both new and existing)
   - No regressions in search, impact, or other tools

2. **Performance**
   - PPR <100ms for 50K edge graphs
   - Search latency increase <50ms when PPR enabled
   - Server startup with PPR graphs <10 seconds (all projects combined)

3. **Code Quality**
   - graph.py >80% test coverage
   - All public functions type-hinted and documented
   - No circular imports, no linting errors

4. **Ranking Improvement**
   - On representative 1K+ symbol project: >3% precision lift (recall@5, nDCG)
   - Impact analysis ranks affected symbols by PPR distance (visibly different from BFS)
   - Search results show rank_sources indicating which signals contributed

5. **Graceful Degradation**
   - Sparse projects (<1 edge per symbol) transparently fall back to 2-way RRF
   - Empty graphs don't crash search or impact tools
   - Server continues operating if graph load fails for one project

---

## Future Work (Phase 6+)

1. **Graph Backend Comparison** — Benchmark NetworkX, scikit-network, fast-pagerank as drop-in replacements for scipy. Document performance/maintenance tradeoffs.

2. **Weighted RRF** — Parameterize weights for keyword/semantic/PPR signals. A/B test on real agents to find optimal weighting.

3. **Lazy Graph Loading** — Load graphs on first access (not startup) to reduce server initialization time for many-project deployments.

4. **LRU Cache for Graphs** — Implement eviction policy for in-memory graphs when memory exceeds threshold (e.g., 500MB on multi-project servers).

5. **Graph Visualization** — Add optional `visualize_ppr()` function to inspect PPR scores and verify correctness on small graphs during debugging.

6. **Per-Agent Personalization** — Allow agents to provide custom personalization vectors (e.g., "rank functions in this module higher").

7. **Document-Symbol Graph** — Extend edges table to include edges from document sections to code symbols, enabling PPR across hybrid search (code + docs).

---

## Appendix: Algorithm Details

### Personalized PageRank Formula

```
p_{k+1} = (1 - alpha) * A^T * p_k + alpha * p_seed

where:
  - A = column-stochastic adjacency matrix (normalized out-degrees)
  - p = probability vector (size = symbol count)
  - alpha = damping factor (probability of teleport, default 0.15)
  - p_seed = personalization vector (biased toward seed symbols)
```

**Intuition:** At each iteration, a random walker at node i either:
1. Follows an outgoing edge (probability 1 - alpha)
2. Teleports to a seed symbol (probability alpha)

This converges to a stationary distribution where node importance reflects both incoming edges (degree centrality) and proximity to seed symbols (personalization).

### Convergence Analysis

- **Typical convergence:** 10–20 iterations for tolerance 1e-6
- **Worst case (dense graph):** 50–100 iterations
- **Proof:** Power iteration on stochastic matrices converges exponentially (eigenvalue gap bounded away from 1)
- **For Tessera:** 50 iterations sufficient for all expected graph sizes

### Sparse Matrix Representation

**CSR Format (Compressed Sparse Row):**
- Storage: O(nnz) where nnz = number of edges
- Matrix-vector multiply: O(nnz) time
- Memory: ~8 bytes per edge (Python numpy dtypes)
- For 50K edges: ~400 KB (negligible)

**Example:**
```
Graph:  0 → 1
        0 → 2
        1 → 2

CSR Matrix:
  row = [0, 0, 1]  (from_id)
  col = [1, 2, 2]  (to_id)
  data = [1, 1, 1] (weights)
  shape = (3, 3)
```

---

## References

1. [HippoRAG 2: From RAG to Memory (arXiv 2502.14802)](https://arxiv.org/abs/2502.14802) — 7% improvement with PPR on knowledge graphs
2. [RepoGraph: Enhancing AI Software Engineering (ICLR 2025)](https://openreview.net/forum?id=dw9VUsSHGB) — 32.8% improvement with graph-aware code ranking
3. [Building a better repository map with tree sitter (Aider, Oct 2023)](https://aider.chat/2023/10/22/repomap.html) — Production use of tree-sitter + PageRank
4. [Reciprocal Rank Fusion (OpenSearch)](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/) — RRF handles sparse signals gracefully
5. Architecture Spec: `docs/plans/architecture/spec-v2.md` lines 331–373 (PPR design)

---

**Prepared by:** Specification Writer
**Reviewed by:** (awaiting panel feedback)
**Status:** Ready for Phase 1 implementation round
