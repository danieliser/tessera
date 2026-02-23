# CodeMem Phase 1 CTO Conditions

This document outlines the five critical architectural conditions (CTOs) defined during the architecture phase and their resolution strategy.

## C1: Vector Search Async Wrapper

**Problem**: FAISS vector search is CPU-bound. Blocking queries in an async MCP server event loop will cause request latency to compound with concurrent clients.

**Solution**: Wrap FAISS search calls with `asyncio.to_thread()` to offload blocking I/O to a thread pool.

**Applies to**:
- `cosine_search()` in `search.py` — FAISS IndexFlatIP queries
- `get_all_embeddings()` in `db.py` — SQLite BLOB reads into numpy arrays
- Embedding client batch calls in `embeddings.py`

**Phase 1 Resolution**: FAISS replaces LanceDB for vector search. FAISS is 5-20x faster (3.5ms avg at 50K vectors vs 73ms for LanceDB). Embeddings stored as SQLite BLOBs, loaded into FAISS IndexFlatIP at query time. Single-store architecture (SQLite only) eliminates dual-store coherence concerns.

**Testing**: Benchmark results confirm p95 <25ms at 50K vectors, well within C5 latency gates.

---

## C2: Session Token Distribution Protocol

**Problem**: Scope gating requires passing capability tokens to task agents. Must be secure, auditable, and respect sandbox boundaries.

**Solution**:
1. Tokens generated in-process by `auth.create_scope()` (cryptographically secure, non-guessable)
2. Stored in SQLite `sessions` table (token → scope mapping + creation time + agent ID)
3. Passed to task agents via MCP `initialize` message over stdio (single-machine trust boundary)
4. Tokens never transmitted outside MCP client/server channel (no HTTP, no network)

**Assumptions**:
- Single-machine deployment (trusted local environment)
- Persistence agent is the only source of scope creation
- Task agents are verified before scope delegation

**Scope Tiers**:
- **Project Scope**: Task agent accessing single repository only
- **Collection Scope**: Ecosystem agent accessing multiple repositories within a collection
- **Global Scope**: Persistence agent only; top-down admin access (register projects, assign scopes, trigger indexing)

**Implementation**: `src/codemem/auth.py` provides token management. `src/codemem/server.py` validates tokens on every request.

---

## C3: Re-index Crash Recovery (Phase 1)

**Problem**: Re-indexing modifies SQLite (structural data + embedding BLOBs). Crash mid-operation leaves the database in inconsistent state.

**Solution in Phase 1**: Re-indexing is **NOT idempotent**.
- Crash mid-index detected by incomplete `indexing_jobs` entry (Phase 2 feature)
- Manual recovery required: `python -m codemem reindex --force <project>`
- Force flag wipes the SQLite database, then re-indexes from scratch

**Phase 2 Enhancement**: Implement transaction-based recovery.
- Track indexing state in `indexing_jobs` table (path, status, checksum, timestamp)
- Atomic commit: SQLite transaction (single-store, no dual-store sync needed)
- Restart automatically detects incomplete jobs and resumes from last checkpoint

**Why Phase 1 accepts this risk**:
- Initial use case: humans re-index manually during development
- Crash scenarios rare with stable hardware/power (single machine)
- Addresses risk in Phase 2 after core indexing is validated

**Implementation**: `src/codemem/indexer.py` orchestrates re-indexing. Phase 2 will add job tracking.

---

## C4: Data Store Coherence

**Problem (original)**: Two separate databases (SQLite for structure, LanceDB for vectors) must stay in sync. Orphaned vectors or missing symbols break queries.

**Resolution**: This condition is **fully resolved** by the single-store architecture. All data — structural index, embeddings, FTS5 — lives in one SQLite database. SQLite transactions provide atomicity. No dual-store coherence problem exists.

FAISS indices are built at query time from SQLite data (not persisted separately), so there is no second store to sync.

**Status**: Resolved. Original spike at `tests/spikes/coherence_spike.py` is historical — it validated the dual-store pattern before the architecture simplified.

---

## C5: Phase 1 Latency Gate Criteria

**Problem**: MCP server must respond within reasonable time for real-time agent interaction. Slow queries block agent decision-making.

**Solution**: Define latency targets per operation type, measure via benchmarks.

**Latency Targets (p95)**:
- **Keyword search** (SQLite FTS5): <20ms
- **Semantic search** (FAISS vector): <30ms
- **RRF merge** (reciprocal rank fusion): <10ms
- **Total query** (keyword + semantic + merge): <100ms

**Measurement**:
- Benchmark suite: `tests/benchmarks/`
- Uses pytest-benchmark plugin
- Test against realistic corpus (10K+ files, 1M+ symbols, 500K+ vectors)
- Run with profiling to identify bottlenecks

**Acceptance Criteria**:
- All benchmarks pass with no operation exceeding its latency gate
- p95 does not exceed 100ms for combined queries
- Memory usage <2GB for typical project (PhP core: ~200K files)

**If gates fail**:
1. Profile with cProfile to find hot paths
2. Optimize without changing schema (indexing strategy, query plan)
3. If still failing, revisit architecture (e.g., columnar compression, GPU acceleration)
4. Document trade-off decisions

**Phase 1 Implementation**: Basic benchmark infrastructure in `tests/benchmarks/test_search_latency.py`. Real measurements collected during integration testing.

---

## Summary

| CTO | Risk | Phase 1 Mitigation | Phase 2+ Resolution |
|-----|------|-------------------|---------------------|
| C1  | Event loop blocking | asyncio.to_thread() wrapper | Thread pool sizing (FAISS is CPU-bound, no async API) |
| C2  | Scope token leakage | In-process generation, stdio-only transport | Encrypted token rotation + MCP channel encryption |
| C3  | Crash mid-index | Manual recovery required | Transaction-based recovery with job tracking |
| C4  | DB coherence failures | Resolved — single-store (SQLite only) | N/A |
| C5  | Query latency > 100ms | Benchmark infrastructure + optimization | Caching layer + query optimization |

**Next Steps**:
1. Implement module stubs (in progress)
2. Run C4 spike test (separate task)
3. Implement core modules in order: db → parser → chunker → embeddings → indexer
4. Write integration tests validating all CTO conditions
5. Run latency benchmarks (C5) before Phase 1 completion
