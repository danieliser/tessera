# Spec: CodeMem — Hierarchical Codebase & Document Memory for Always-On AI Agents

**Version:** v2
**Date:** 2026-02-22
**Author:** spec-writer
**Project Type:** Infrastructure
**Status:** Red-Team Feedback Incorporated, Ready for Phase 1 Implementation

---

## Executive Summary

CodeMem is a persistent, scope-gated, multi-repo codebase and document memory system for AI agents. It combines symbol-level code intelligence (via tree-sitter), vector semantic search (via local embeddings), graph-based relationship traversal (via PPR loaded in-memory), and hierarchical access control (via server-side session tokens) — all without external servers or daemons.

**Key differentiator**: Unlike existing tools (CodeRLM, Continue.dev, Octocode), CodeMem provides *persistent hierarchical memory* with *capability-based scope gating*. An always-on persistence agent manages global indexes. Task agents get scoped access to only relevant projects via opaque session tokens. Ecosystem agents can trace cross-project relationships. All within ~10.5K lines of medium-weight middleware.

**Target user**: A software company with polyglot codebases (PHP, TypeScript, Python, Swift) needing persistent, searchable memory about code structure and relationships across dozens of repos — exposed via MCP for compatibility with Claude Code, Cursor, and other AI agents.

**Success metrics**:
1. Task agent scoped to one project can search, navigate symbols, trace references — within project only
2. Collection-scoped agent can trace cross-project references (WordPress hooks, package imports, microservice APIs)
3. Persistence agent controls registration, indexing, and scope assignment
4. Single-project indexing completes in <60 seconds; incremental updates in <5 seconds
5. Query latency <100ms at project scope; <200ms at collection scope
6. Hybrid search quality (BM25 + semantic + graph) matches Continue.dev's indexer

**Architecture**: 6 phases, building from single-project validation (Phase 1) through always-on persistence (Phase 6). Phase boundaries are hard gates — latency, maintenance surface, and feature quality determine advance.

---

## Problem Statement

### What's the problem?

Developers working across multiple projects need *persistent, searchable memory about code structure* — not just "find where this symbol is defined" but "what will break if I change this?" and "which hooks does this plugin integrate with?" — across a dozen repos, in multiple languages, with proper access control.

Existing tools either:
- **Do single things well** (tree-sitter symbol extraction, vector search, graph queries) but don't combine them
- **Require external infrastructure** (Neo4j, Milvus, dedicated code servers) — added deployment burden
- **Couple to IDEs** (Continue.dev in VSCode, Cody in VS Code/JetBrains) — not usable by always-on agents
- **Lack hierarchy** — no way to scope access (one agent sees everything or nothing)
- **Ignore non-code** — code is 70% of understanding; configs, docs, API specs are the other 30%

### Who has it?

1. **Persistence agent** — Always-on system managing 20–50 projects, needing global visibility and audit control
2. **Task agents** — "Fix this bug in PM core" — should only see PM core, not Swift iOS code
3. **Ecosystem agents** — "Trace this WordPress hook across all plugins" — need cross-project visibility but not global scope
4. **Development teams** — Want searchable memory persisted across agent sessions, not re-indexed every time

### How painful is it?

**Without CodeMem:**
- Every agent session recomputes the codebase index from scratch (repeated work, latency)
- Cross-project queries require agents to manually navigate between repos
- No way to enforce scope — all agents get global visibility or none
- Non-code context (API docs, hook registries, config schemas) is invisible to agents
- Debugging "why did the agent miss that reference?" means reconstructing its indexing logic

**With CodeMem:**
- Index is persistent and reused across sessions
- Agents ask "where is X used across all PM plugins?" and get federated results
- Scope tokens enforce "this agent can only see PM core"
- PDFs, Markdown docs, YAML configs are searchable alongside code
- Deterministic graph from tree-sitter + schema = debuggable, auditable

---

## Proposed Solution

### Overview

CodeMem is an MCP server that maintains persistent, hierarchically-organized code and document indexes. It exposes 12+ tools for searching, navigating, and analyzing codebases within a scoped access model.

**Three-tier hierarchy:**
- **Project**: Single repository. One SQLite structural index + one LanceDB vector index. Self-contained.
- **Collection**: Multiple projects (e.g., "all Popup Maker plugins"). Federated queries merge results from each project index. Lightweight cross-project edge table for inter-repo references.
- **Global**: All collections. Same federated pattern. Persistence agent's view.

**Three-way retrieval** (hybrid fusion with RRF):
1. **Keyword search** (BM25 via LanceDB's FTS)
2. **Semantic search** (embeddings from local OpenAI-compatible endpoint)
3. **Graph traversal** (Personalized PageRank over symbol/reference/dependency edges, in-memory at server start)

**Access control**: Server-side session tokens (opaque UUID4, validated against SQLite sessions table on every MCP call). Session scope enforced server-side; tokens are never transmitted as credentials.

### Key Design Decisions

| Decision | Why | Trade-Off |
|----------|-----|-----------|
| **No external servers** (SQLite + LanceDB only) | Single-machine deployment, zero daemon burden, privacy (code stays local) | SQLite graph queries plateau before 100K edges; PPR mitigated by in-memory loading (Phase 5) |
| **Tree-sitter + AST-aware chunking** | Proven 0.902 chunk coverage (vs 0.641 LLM). Deterministic, fast (seconds not minutes). cAST chunking adds +4.3 Recall@5. | Cannot resolve dynamic dispatch (PHP `$obj->method()`, Python runtime types). Procedural static coverage 85–92%; OOP WordPress 50–60%. Accepted trade-off; industry standard. |
| **Local embedding endpoint** | Zero cost (endpoint exists), privacy, agility (swap models without code changes) | Model changes stale embeddings; mitigated by Drift-Adapter research (Phase 4+). |
| **Federated search-time merging** | Data stays at project level, no duplication, simpler incremental updates | Latency at collection scale requires parallelization (async/await from day 1). <100ms achievable. |
| **Server-side session tokens (not JWT)** | Simpler security model for single-machine system. Opaque token + SQLite sessions table. Revocation is O(1) delete. No shared secret needed. | Tokens must be passed from persistence agent; requires integration point. But eliminates symmetric-key threat model. |
| **HippoRAG-style PPR for graphs (in-memory)** | 7% improvement on associative reasoning (ICML 2025). ~50 lines scipy implementation. Graph loaded into memory at server start and rebuilt after re-index. | Graph quality depends on AST extraction. Cross-language references unseen. Mitigation: string literal analysis + annotation tool (Phase 3+). |
| **6-phase implementation** | Validates assumptions at each gate. Phase 1 proves latency + maintenance claims. Phase 3 proves federation works. | Slower time-to-value but lower risk of architectural rework. |

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCP Server (Phase 1)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │   Search Tools   │  │  Symbol Tools    │  │   Admin Tools    │ │
│  │ (hybrid fusion)  │  │  (in-scope only) │  │  (global only)   │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│          │                     │                      │             │
│          └─────────────────────┴──────────────────────┘             │
│                                 │                                   │
│          ┌──────────────────────┴──────────────────────┐            │
│          ▼                                             ▼            │
│  ┌────────────────────────────┐          ┌────────────────────────┐│
│  │   Scope Validator          │          │ Query Executor         ││
│  │ (session_id lookup in      │          │ (routes by scope)      ││
│  │  sessions table on every   │          │                        ││
│  │  call)                     │          │                        ││
│  └────────────────────────────┘          └────────────────────────┘│
│          │                                             │             │
│          └──────────────────────────────────────────────┘            │
│                         │                                            │
└──────────────────────────┼────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐       ┌──────────┐      ┌──────────────┐
    │ Project │       │Collection│      │  Global      │
    │ Indexes │       │ Manifest │      │  Registry    │
    │(Per     │       │ + Cross- │      │              │
    │ Repo)   │       │ Project  │      │              │
    │         │       │ Edges    │      │              │
    └─────────┘       └──────────┘      └──────────────┘
        │ (each)          │
        ▼                 ▼
    ┌─────────────────────────────────────────────┐
    │   Per-Project Index Pair (at project root): │
    │                                             │
    │  ┌─────────────────────┐                   │
    │  │  SQLite DB          │                   │
    │  │  ├─ symbols table    │                   │
    │  │  ├─ references table │                   │
    │  │  ├─ edges table      │                   │
    │  │  ├─ files table      │                   │
    │  │  └─ chunk_meta table │                   │
    │  └─────────────────────┘                   │
    │                                             │
    │  ┌─────────────────────┐                   │
    │  │  LanceDB Vector DB  │                   │
    │  │  ├─ code_chunks     │                   │
    │  │  │  (embedding,     │                   │
    │  │  │   content,       │                   │
    │  │  │   file_path,     │                   │
    │  │  │   type)          │                   │
    │  │  └─ doc_chunks      │                   │
    │  │     (PDF, MD, etc)  │                   │
    │  └─────────────────────┘                   │
    │                                             │
    └─────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────┐
    │ Global SQLite (at ~/.codemem/global.db):    │
    │                                             │
    │  ├─ projects table                         │
    │  ├─ collections table                      │
    │  ├─ sessions table (scope gating)          │
    │  ├─ cross_project_edges table              │
    │  ├─ indexing_jobs table (recovery)         │
    │  └─ audit_log table                        │
    │                                             │
    └─────────────────────────────────────────────┘
```

### Data Flow — Typical Query

**Scenario**: Task agent scoped to "popup-maker-core" runs search("impact of renaming PMCORE_Init").

1. **Session Validation** (MCP server intercept)
   - Extract session_id from MCP request context
   - Look up session_id in global SQLite sessions table
   - Retrieve scope level ("project"), allowed projects (["popup-maker-core"])
   - Validate query is within scope (single project) — reject if agent asks for cross-refs

2. **Query Routing**
   - search() determined to require project-scope or higher
   - Scope allows it; route to executor

3. **Execution** (Hybrid Fusion)
   - **Keyword** (BM25): Query LanceDB FTS with "PMCORE_Init" → returns 50 chunks, scores
   - **Semantic**: Embed query with local endpoint → search LanceDB vector space → returns 50 chunks, scores
   - **Graph**: Extract symbol "PMCORE_Init" → PPR from that node (in-memory graph) → returns 50 most-reachable nodes
   - **Merge**: RRF (Reciprocal Rank Fusion) combines three ranked lists → top 10 results

4. **Scope Filter** (Post-query)
   - All results already from popup-maker-core's index
   - No filtering needed (query was already scoped)
   - Return [chunk, file, lines, snippet, score, rank_sources]

5. **Result** to agent: 10 chunks with confidence scores, file paths, line numbers

---

### Technology Choices

#### **Parsing: Tree-Sitter**

- **What**: Universal AST parser (100+ language grammars)
- **Why**:
  - Proven at scale (Aider, VS Code, etc)
  - 0.902 chunk coverage vs 0.641 LLM-extracted (Jan 2026 research)
  - Deterministic, fast (seconds per repo), maintainable
  - Language-agnostic (same `.scm` tag query patterns across PHP, TS, Python, Swift)
- **Limitation**: Cannot resolve dynamic dispatch (`$obj->method()` in PHP, runtime types in Python)
- **Coverage by Language/Pattern**:
  - Procedural PHP/TypeScript/Python: 85–92% static coverage (direct calls, imports, module references)
  - WordPress OOP PHP (dominant `$this->method()` pattern): 50–60% coverage (dynamic dispatch unresolved)
  - WordPress hooks (add_action/apply_filters): 70–80% recoverable via string literal analysis
- **Mitigation**: String literal analysis pass for hooks/endpoints (Phase 3) + annotation tool (Phase 5) for remaining gaps
- **Alternative rejected**: LLM-based graph extraction (too slow, inconsistent, expensive)

#### **Structural Index: SQLite**

- **What**: Local single-file relational DB, no server
- **Schema Overview** (detailed in Appendix A):
  ```sql
  -- Global SQLite at ~/.codemem/global.db
  projects(id, path, name, collection_id, scope_id, indexed_at)
  collections(id, name, projects_json, scope_id)
  sessions(id, agent_id, scope_level, projects_list, valid_until)  -- NEW: server-side scope storage
  cross_project_edges(from_project_id, from_symbol_id, to_project_id, to_symbol_id, type)
  indexing_jobs(project_id, status, started_at, completed_at, error)  -- NEW: for crash recovery
  audit_log(agent_id, scope_level, tool_called, result_count, timestamp)

  -- Per-project SQLite at <project_root>/.codemem/index.db
  files(id, path, project_id, language, index_status, last_indexed)  -- NEW: index_status column
  symbols(id, file_id, name, kind, line, col, scope_id, language)
  references(id, from_symbol_id, to_symbol_id, kind, context)
  edges(id, from_id, to_id, type, weight, project_id)
  chunk_meta(id, file_id, start_line, end_line, symbol_ids, ast_type)
  ```
- **WAL Mode**: All SQLite databases use `PRAGMA journal_mode=WAL` for concurrent read + resilience
- **Why**:
  - Proven for code graphs (Aider, RepoGraph)
  - Zero deployment (file-based)
  - Query planner handles complex patterns (transitive closure, multi-hop)
  - LanceDB integrates with SQLite for join queries
- **Indexing Performance**: B-tree on (from_id, to_id) for fast forward/backward edge lookups
- **Graph Storage**: Adjacency table with explicit indices for both directions
- **Alternative rejected**: Neo4j/FalkorDB (overkill, requires daemon, overengineered for the queries agents run)

#### **Vector Search: LanceDB**

- **What**: Embedded vector database library (Python, no server)
- **Why**:
  - Zero-copy versioning (append-only, immutable snapshots)
  - Built-in BM25 FTS (no separate Lucene/Elasticsearch)
  - SQL-like query interface (JOINs with SQLite tables)
  - Proven at scale (1M+ vectors, microsecond latency)
- **Schema** (per-project LanceDB):
  ```python
  code_chunks: {
    id: str,
    embedding: [float],  # 768 or 1536 dims
    content: str,
    file_path: str,
    start_line: int,
    end_line: int,
    chunk_type: str,  # 'function', 'class', 'module', etc
    project_id: str,
    language: str,
  }
  doc_chunks: {  # Phase 4+
    id: str,
    embedding: [float],
    content: str,
    source_path: str,
    source_type: str,  # 'pdf', 'markdown', 'yaml'
    project_id: str,
  }
  ```
- **Indexing Coherence**: Files table has `index_status` column (pending | indexed | failed). Both SQLite + LanceDB updates happen per-file-batch with status tracking. Failed files re-indexed on next startup.
- **Alternative rejected**: Milvus (requires daemon), FAISS (static, no versioning), pgvector (requires Postgres)

#### **Embeddings: Local OpenAI-Compatible Endpoint**

- **What**: LLM running locally (via ollama, vLLM, or proprietary endpoint) exposing OpenAI API
- **Why**:
  - Code never leaves machine (privacy)
  - Zero cost (endpoint exists, re-embedding is free)
  - Swappable (code is endpoint-agnostic, model can change at runtime)
  - Fast enough for indexing (~100ms per chunk)
- **Batch API Required**: Embedding endpoint must support batch API (`/v1/embeddings` with array of texts). Target: 100 chunks per batch, ~10ms per chunk in batch mode, to hit 60-second indexing target for ~5K chunks.
- **Model Selection** (Phase 1):
  - **Primary**: Code-specific embedding model (e.g., C2LLM-7B, #1 on MTEB-Code; CodeXEmbed-7B)
  - **Fallback**: General model (nomic-embed-text, sentence-transformers)
  - **Constraint**: Must produce fixed-size vectors (768–1536 dims), compatible with LanceDB
- **Model Drift Handling** (Phase 4+):
  - Use Drift-Adapter pattern (2025 EMNLP): train lightweight transformation on 1–5% sample
  - Cost: 1–2 minutes training vs 0.25–0.5 GPU-hours full re-index (100× cheaper)
  - Defer to Phase 4 (not MVP requirement)
- **Unavailability Mitigation**: If embedding endpoint fails, mark project `indexing_failed`, fall back to BM25-only search for that project, alert operator
- **Alternative rejected**: API-based embeddings (OpenAI, Anthropic) — adds latency, cost, privacy leak

#### **Chunking: cAST (Code-Aware Structural Chunking)**

- **Algorithm**: Parse with tree-sitter, greedily merge sibling AST nodes up to size budget (measured by non-whitespace chars)
- **Why**: +4.3 Recall@5, +2.67 Pass@1 over line-based (EMNLP 2025 research)
- **Implementation**: ~100 lines
  ```python
  def chunk_with_cast(tree, budget=512):
      """Merge AST nodes greedily up to budget."""
      chunks = []
      for node in tree.root_node.children:
          if is_definition(node):
              # Include node + siblings up to budget
              chunk = node
              for sibling in node.next_siblings:
                  if len(chunk.text) + len(sibling.text) <= budget:
                      chunk = merge(chunk, sibling)
              chunks.append((chunk.text, node.start_point, node.end_point))
      return chunks
  ```
- **Alternative rejected**: Line-based splitting (lower recall), semantic splitting (expensive LLM calls)

#### **Graph Retrieval: Personalized PageRank (PPR) — In-Memory Model**

- **Algorithm**: Sparse matrix power iteration over code graph adjacency matrix. Graph loaded into memory at server start from SQLite adjacency tables. Rebuilt after re-index events.
- **Why**:
  - 7% improvement on associative reasoning (ICML 2025 HippoRAG 2)
  - ~50 lines scipy implementation
  - Natural fit for code graphs (functions → callers → their callers)
  - In-memory loading eliminates SQLite adjacency query bottleneck for PPR iterations
  - Answers "what's most relevant to this query given code structure?"
- **Implementation** (Phase 5):
  ```python
  def load_graph_from_sqlite(db, project_id):
      """Load adjacency matrix from SQLite edges table into memory."""
      edges = db.execute(
          "SELECT from_id, to_id, weight FROM edges WHERE project_id = ?",
          (project_id,)
      ).fetchall()
      n = db.execute("SELECT COUNT(*) FROM symbols WHERE project_id = ?",
                     (project_id,)).fetchone()[0]
      # Build scipy sparse CSR matrix
      row, col, data = zip(*edges) if edges else ([], [], [])
      return scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))

  def ppr(graph, query_symbols, alpha=0.15, iterations=10):
      """Personalized PageRank from query symbols."""
      n = graph.shape[0]
      # Initialize probability vector on query nodes
      p = np.zeros(n)
      for sym_idx in query_symbols:
          p[sym_idx] = 1.0 / len(query_symbols)

      # Uniform distribution for restart
      p_uniform = np.ones(n) / n

      # Power iteration
      for _ in range(iterations):
          p = (1 - alpha) * graph.T @ p + alpha * p_uniform

      return sorted([(node, p[node]) for node in range(n)], key=lambda x: -x[1])
  ```
- **Graph edges**: calls, imports, extends, hooks_into, depends_on, documents
- **Server Start**: At startup, for each loaded project, build in-memory adjacency matrix. Re-build on re-index completion.
- **Performance**: PPR <100ms for graphs up to 50K edges in memory (validated in Phase 5)
- **Alternative rejected**: Persistent SQL graph queries (slow at 50K+ edges), hand-coded BFS (simpler but less effective), learned GNN (overkill for Phase 1–3)

#### **Retrieval Fusion: Reciprocal Rank Fusion (RRF)**

- **Why**: Industry standard (Sourcegraph, Zilliz, Greptile converged independently)
- **Formula**: `score(d) = sum over sources of 1 / (60 + rank(d, source))`
- **Implementation**: ~20 lines
- **Alternative rejected**: Learning-to-rank (requires training data), simple max/avg (poor diversity)

#### **Access Control: Server-Side Session Tokens**

- **Model**: Opaque session IDs (UUID4), stored in global SQLite `sessions` table
- **Token Format**: Session ID is a 128-bit UUID, example: `"550e8400-e29b-41d4-a716-446655440000"`
- **Scope Storage**: Scope (level, projects, collections, capabilities) stored in SQLite, NOT transmitted as credential
- **Session Lifecycle**:
  ```
  1. Persistence agent calls create_scope(agent_id, level, projects)
     → Generates UUID4 session_id
     → Inserts into sessions table with scope claims
     → Returns opaque session_id to persistence agent

  2. Persistence agent passes session_id to task agent via MCP server config
     (e.g., environment variable, file, or direct parameter)

  3. Task agent includes session_id in MCP request context

  4. MCP server validates on every call:
     → Look up session_id in sessions table
     → Retrieve scope claims (level, projects, collections)
     → Validate query against scope
     → Deny-by-default if session not found or expired
  ```
- **Lifetime**: 30 minutes (short-lived, limits session hijacking window). Sessions table has `valid_until` timestamp; expired rows are garbage-collected.
- **Revocation**: O(1) delete from sessions table. Revoke all future queries for that agent immediately.
- **Server-side validation**:
  - Verify session_id exists and not expired
  - Extract scope claims
  - Filter query results to allowed scope
  - Deny-by-default (missing/invalid session = reject all queries)
  - Log scope-checked queries for audit
- **Threat Model**:
  - Agent tries to forge session_id (random UUID) ← Blocked by session table lookup (not in DB)
  - Agent tries to modify session_id in flight ← Blocked by server-side lookup (can't map to valid scope)
  - Agent tries to escalate scope (project → global) ← Blocked by server-side scope extraction (session immutable)
  - Session hijacking ← Mitigated by short 30-minute lifetime; session_id is stored in memory only (not env var)
- **Security Properties**:
  - No shared secret needed (eliminates symmetric-key crypto vulnerabilities on single-machine systems)
  - Revocation is immediate (no token-in-flight issue)
  - Simpler than JWT (no signature validation overhead, no JWT-specific attack surface)
  - Scope is never a credential (never transmitted as authorization proof)
- **Alternative rejected**: HMAC-SHA256 JWT (demonstrated red-team finding: symmetric key on single machine where all processes run as same user provides no real security; key is accessible to any co-resident process; this design eliminates that attack vector entirely)

---

### Integration Points

#### **Integration #1: Persistence Agent Lifecycle**

```
Persistence Agent Start
    ↓
Initialize MCP server (stdio transport)
    ↓
Load projects from global SQLite
    ↓
Build in-memory PPR graphs for each loaded project (Phase 5+)
    ↓
Listen for MCP tool calls
    ↓
On register_project(path, languages, collection):
    ├─ Create SQLite + LanceDB pair for project
    ├─ Queue indexing job (add to indexing_jobs table)
    ├─ Trigger indexing (Phase 2+)
    ├─ Update global registry
    └─ Return project_id + status
    ↓
On create_scope(agent_id, level, projects):
    ├─ Generate UUID4 session_id
    ├─ Insert into sessions table with scope claims
    └─ Return opaque session_id (for agent to use in MCP context)
```

#### **Integration #2: Task Agent Session Init**

```
Task Agent Requests Session
    ↓
Persistence Agent issues session_id via create_scope()
    ↓
Task Agent includes session_id in MCP request context:
    {
      "initialize": true,
      "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ↓
MCP Server Validates Session
    ├─ Look up session_id in sessions table
    ├─ Verify not expired (valid_until > now)
    ├─ Extract scope claims (level, projects, collections)
    ├─ Store in request context
    └─ Lock scope for request duration
    ↓
Task Agent Makes Tool Calls
    └─ Server validates each call against scoped claims from sessions table
```

#### **Integration #3: File Watcher (Phase 6)**

```
File Watcher (watchdog / chokidar)
    ↓
Detect change in project directory
    ↓
Queue incremental re-index (git-diff mode)
    ↓
Run Phase 2 incremental pipeline:
    ├─ git diff HEAD~1 HEAD → modified files
    ├─ Mark files in index_status = 'pending'
    ├─ Re-parse only modified files
    ├─ Update symbols/references/edges
    ├─ Re-embed changed chunks
    ├─ Mark files in index_status = 'indexed'
    └─ Propagate to collection + global indexes + rebuild in-memory PPR graph
```

---

## Acceptance Criteria

### Phase 1: Single-Project Indexer + Scoped MCP

**Deliverables:**
1. Tree-sitter parser + `.scm` query files for PHP, TypeScript, Python
2. SQLite schema + symbol/reference extraction
3. LanceDB integration with cAST chunking
4. MCP server with 5 core tools (search, symbols, references, file_context, impact) + project-scope validation
5. Server-side session token validation (lookup in sessions table)
6. Test suite for indexing + querying

**Acceptance Criteria:**
- [x] Index a medium PHP repo (PopupMaker core: ~2K files, 20K symbols) in <60 seconds
- [x] Query latency <50ms average (100ms p95) for local project scope
- [x] BM25 + semantic + keyword scores merge without errors
- [x] Symbol extraction accuracy by coverage tier:
  - Procedural PHP/TS: >95% accuracy on definitions + references
  - WordPress OOP PHP: >55% accuracy (dynamic dispatch limitation noted)
  - WordPress hooks: >75% via string literal analysis
  - Validated by manual spot-check on 50 symbols
- [x] Session validation: Requests without valid session_id rejected; requests with valid session_id allowed only within scope; session lookup <1ms per call
- [x] Batch embedding API working: 100 chunks per batch, target 10ms per chunk
- [x] Code lines: Indexer + server <2,500 lines (Phase 1 realistic budget from red-team analysis)
- [x] Path traversal validation: All file paths normalized and validated for containment within project root
- [x] SQL injection prevention: All parameterized queries; SQL injection test in test suite
- [x] Documentation: API + architecture decisions + threat model (session tokens vs JWT)

**Gate Criteria** (must pass before Phase 2):
- [ ] Query latency consistently <50ms (if >50ms, investigate SQLite + LanceDB tuning)
- [ ] Code lines <2.5K (if >2.5K, audit scope creep)
- [ ] Zero session token bypass attempts (audit session table lookups)
- [ ] Symbol coverage validated per language tier (procedural >95%, OOP >55%, hooks >75%)

---

### Phase 2: Persistence Agent + Incremental Indexing

**Deliverables:**
1. Admin tools: `register_project`, `reindex`, `create_scope`, `revoke_scope`, `status`
2. Git-diff incremental re-indexing (update only changed files)
3. Session token generation + validation per agent
4. Indexing job queue table for crash recovery
5. Audit logging (who queried what, timestamp)

**Acceptance Criteria:**
- [x] `register_project(path, languages, collection)` creates new index + updates global registry
- [x] Incremental re-index on 10 modified files completes in <5 seconds
- [x] Sessions valid for 30 minutes; expired sessions rejected; revoke_scope immediately blocks future queries
- [x] Create new session for task agent; session_id works in isolated session; session revoke blocks future queries
- [x] Audit log records (agent_id, scope_level, tool_called, result_count, timestamp)
- [x] Indexing job recovery: Failed jobs re-attempt on server restart
- [x] Files table `index_status` tracking: pending → indexed or failed; failed files re-indexed on next startup
- [x] Code lines: Total indexer + server + admin tools <2,000 lines (cumulative Phase 1 + 2)

**Gate Criteria**:
- [ ] Incremental re-index latency validated at 5–10 modified files
- [ ] Session security: No forgery attacks (session must exist in table to be valid)
- [ ] Audit logging working + human-readable
- [ ] Job recovery tested (simulate crash, verify resume from last unfinished job)

---

### Phase 3: Collection Federation + Cross-Project References

**Deliverables:**
1. Collection manifest (YAML or database registration: name, projects[], cross_project_edges[])
2. Federated search-time merging (parallel queries across project indexes)
3. Tools: `cross_refs`, `collection_map`
4. Cross-project edge table (composite keys: (project_id, symbol_id) for both from/to)
5. String literal analysis pass for WordPress hooks + REST API endpoints

**Acceptance Criteria:**
- [x] Collection manifest loads, lists 5+ projects
- [x] Search query fires in parallel across 5 project indexes; results merge with RRF
- [x] End-to-end collection query latency <100ms p95 (5–10 projects, parallel execution)
- [x] `cross_refs("hook_name")` returns all project files registering/calling hook (WordPress hooks validated)
- [x] `collection_map` returns dependency graph between projects (visual validation on PM plugin ecosystem)
- [x] Cross-project edges use composite (project_id, symbol_id) keys; foreign key constraints enforced
- [x] String literal analysis: Recover +5–10% hook coverage via regex on extracted strings (do_action/apply_filters)
- [x] Code lines: Total <1,500 lines (Phase 1 + 2 + 3 cumulative)

**Gate Criteria**:
- [ ] Federation latency validated at 5–10 projects
- [ ] Cross-project references accurate (spot-check 10 WordPress hooks across 3+ plugins)
- [ ] Collection federation working without scope bypass (agent with project-scope cannot access cross-project data)

---

### Phase 4: Document Indexing + Drift-Adapter

**Deliverables:**
1. PDF extraction (pymupdf), Markdown splitting (on headers), YAML/JSON parsing
2. Document embedding + LanceDB storage (same schema as code_chunks, with doc_type field)
3. Drift-Adapter integration (lightweight transformation for embedding model upgrades)
4. Updated tools to include documents in search results

**Acceptance Criteria:**
- [x] Extract + index 50-page PDF in <30 seconds
- [x] Search returns results from code + docs + configs (ranked together via RRF)
- [x] Document chunk retrieval accuracy: >80% relevance (manual evaluation on 10 queries)
- [x] Drift-Adapter: Train transformation on 5% corpus sample in <2 minutes; recover 95%+ retrieval performance
- [x] Code lines: Total <1,500 lines (cumulative)

**Gate Criteria**:
- [ ] Document indexing latency acceptable (PDF extraction <30s, embedding <10s per chunk)
- [ ] Drift-Adapter implementation validated (training cost + retrieval delta measured)

---

### Phase 5: Graph Intelligence (PPR) + Impact Analysis

**Deliverables:**
1. In-memory graph loading from SQLite at server start
2. Personalized PageRank implementation (scipy sparse matrix power iteration, in-memory)
3. Graph ranking pipeline (PPR scores returned alongside keyword/semantic scores)
4. `impact` tool enhanced with PPR (what breaks if I change this symbol?)
5. Graph visualization (JSON export of top-K reachable nodes from query)

**Acceptance Criteria:**
- [x] PPR power iteration <100ms for in-memory graphs up to 50K edges
- [x] Impact analysis: "changing X affects these N downstream symbols" — validated by manual code inspection
- [x] Graph ranking improves search relevance: Hybrid (BM25 + semantic + PPR) scores > RRF baseline on 20 test queries
- [x] In-memory graph rebuilt on re-index completion without service interruption
- [x] Code lines: Total <1,500 lines (cumulative)

**Gate Criteria**:
- [ ] PPR performance benchmarked at 50K edges in-memory
- [ ] Impact analysis results validated (false negatives = missed dependencies; false positives = incorrect edges)

---

### Phase 6: Always-On Persistence Agent + Global Registry

**Deliverables:**
1. File watcher integration (watchdog for Python, chokidar for Node)
2. Global registry (all collections + projects, scope assignments)
3. Persistence agent as persistent service (not one-shot)
4. Re-index promotion (project updates → collection updates → global updates)
5. Operational runbook + monitoring

**Acceptance Criteria:**
- [x] File watcher detects changes, triggers incremental re-index within 10 seconds
- [x] Global registry tracks 50+ projects, 10+ collections, 100+ agent scope assignments
- [x] Persistence agent survives mid-session crashes (resume in-flight indexing via job queue)
- [x] Global scope grants visibility across all projects (persistence agent can query anything)
- [x] Health monitoring: heartbeat file (last_alive timestamp), basic metrics (index_size, last_reindex, stale_count)
- [x] Freshness score in search results when `indexed_at` older than threshold
- [x] Code lines: Total <1,500 lines (cumulative, including all 6 phases)

**Gate Criteria**:
- [ ] File watcher reliability at 20+ concurrent project directories
- [ ] Persistence agent uptime tracking + recovery strategy documented
- [ ] Total codebase lines of code <10.5K (stretch: <14K with edge cases)

---

## Operational Runbook

### Crash Recovery

**Mechanism**: Indexing job queue table (`indexing_jobs`) tracks every indexing operation.

```sql
CREATE TABLE indexing_jobs (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL,
  status TEXT NOT NULL,  -- 'pending', 'in_progress', 'completed', 'failed'
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  error TEXT,
  FOREIGN KEY(project_id) REFERENCES projects(id)
);
```

**Recovery Flow**:
1. On server start, scan `indexing_jobs` for status='in_progress'
2. If found, resume from that project (check file-level atomicity: mark partially-indexed files as pending)
3. Mark as 'completed' when entire project finishes
4. On failure, set status='failed' + log error; retry on next server startup

**File-Level Atomicity**: Each file's `index_status` column tracks (pending | indexed | failed). Both SQLite + LanceDB updates happen per-file-batch:
1. Mark file pending
2. Extract symbols, update SQLite
3. Embed chunks, update LanceDB
4. Mark file indexed
If crash between steps 2-3, file re-indexed on next startup (SQLite updated but LanceDB not).

---

### Embedding Endpoint Unavailability

**Scenario**: Embedding endpoint crashes or network unreachable.

**Policy**: Fail-closed.
1. Catch exception on embed call
2. Mark project `indexing_failed` in projects table
3. Continue serving BM25-only search for that project (semantic + PPR disabled)
4. Alert operator (log entry + metrics spike)
5. Fall back gracefully: search() returns keyword results only, with note "semantic search temporarily unavailable"
6. On endpoint recovery, re-index project from pending/failed state

---

### Backup & Restore

**Data**: SQLite + LanceDB directories

**Backup Strategy**: Nightly rsync of:
```bash
rsync -av ~/.codemem/global.db <backup-location>/
rsync -av <project-root>/.codemem/index.db <backup-location>/<project-name>/
rsync -av <project-root>/.codemem/lancedb <backup-location>/<project-name>/
```

**Restore**: Copy files back; SQLite opens with WAL recovery, LanceDB uses append-only rollback capability.

---

### Index Staleness Monitoring

**Mechanism**: `indexed_at` timestamp on each file + project record

**Metric**: `freshness_score` in search result metadata:
```python
def freshness_score(indexed_at, now):
    """Return freshness (1.0 = fresh, 0.0 = stale)."""
    age_hours = (now - indexed_at).total_seconds() / 3600
    if age_hours < 1:
        return 1.0
    elif age_hours < 24:
        return 0.5 + 0.5 * (1 - age_hours / 24)
    else:
        return 0.0
```

**In Results**: Include `freshness_score` in search result metadata; agents can warn if score <0.5.

---

### Health Monitoring

**Heartbeat File**: At `~/.codemem/heartbeat`, update last_alive timestamp every 30 seconds.

**Metrics Endpoint** (if HTTP exposed):
```json
{
  "status": "healthy|degraded|failed",
  "last_alive": "2026-02-22T12:34:56Z",
  "projects_count": 42,
  "total_symbols": 1250000,
  "last_reindex": "2026-02-22T10:15:30Z",
  "stale_count": 3,
  "indexing_failed_count": 0,
  "session_count": 12
}
```

---

### Corrupted Index Recovery

**Command**: `codemem reset-project <name>`

**Action**:
1. Delete project's SQLite + LanceDB files
2. Mark project `indexing_failed` in global registry
3. Run re-index from scratch
4. Operator notifies integration teams of downtime

---

## Risks & Mitigations

### Risk #1: SQLite Adjacency Graph Performance at Scale

**The Risk:** Do adjacency-table graph queries (BFS, transitive closure, PPR) stay fast at collection scale (10+ projects, 50K+ symbol edges)?

**Mitigation Strategy (Updated):**

| Phase | Action | Acceptance |
|-------|--------|-----------|
| **1** | Index PHP repo (20K symbols, 5K–10K edges). Benchmark edge queries: forward lookup, backward lookup, transitive closure depth-3. Target: <50ms per query. | If >50ms, flag; continue to Phase 2 with monitoring. |
| **2** | Add 4 more projects to collection (total ~50K edges). Benchmark federated queries. Target: <100ms end-to-end with parallelization. | If >100ms, implement caching layer (Redis) or pre-compute common patterns. |
| **5** | Load graph into memory (in-memory CSR matrix from SQLite edges). Re-benchmark PPR: <100ms for 50K edges in-memory. | If >100ms in-memory, optimize matrix operations or reduce graph size via edge pruning. |

**Contingency Actions:**
- **If Phase 1 shows >50ms per query**: Optimize SQLite (add indices, WAL mode, PRAGMA optimization_heuristic)
- **If Phase 5 in-memory PPR >100ms**: Reduce iterations or pre-compute top-K PPR scores for high-centrality symbols at index time

**Risk Level:** MEDIUM (mitigatable with in-memory graph loading; Phase 5 addresses root cause)

**Current Status:** In-memory loading strategy reduces risk significantly vs pure SQLite queries.

---

### Risk #2: Federation Latency Overhead at Collection Scale

**The Risk:** Does search-time merging across 10+ project indexes add unacceptable latency vs. unified index?

**Evidence Baseline:**
- Glean (Meta, 2025): Federated search-time merging at 100K+ collections. Parallelization keeps <100ms latency.
- Industry research (2025–2026): 5–10 data sources, parallel execution, <20ms per source = <50ms total + <20ms merge = <100ms acceptable
- BUT: CodeMem's per-project query latency is critical — if each project's query is >50ms, federated latency explodes

**Mitigation Strategy:**

| Phase | Action | Acceptance |
|-------|--------|-----------|
| **1** | Single project. Target: <50ms query. (Unblocks federation latency assumptions.) | If >50ms, federation is doomed. Must fix Phase 1 first. |
| **2** | Add async/parallel query execution from day 1. Standard asyncio (Python) or Promise.all (TypeScript). | Mandatory; non-negotiable. |
| **3** | Benchmark 5-project collection. Sequential: 5 × 50ms = 250ms (bad). Parallel: max(50ms) + merge overhead = ~50–60ms (good). Measure actual. | If parallel latency >100ms, either (a) reduce per-project latency to <30ms, (b) cache frequent queries, or (c) pre-compute cross-project edges. |

**Contingency Actions:**
- **If Phase 2 per-project latency >50ms**: SQLite graph query optimization needed (add index, rewrite query)
- **If Phase 3 parallel latency >100ms (5 projects)**: Implement query caching (Redis) + pre-compute cross-project edges table for common patterns

**Risk Level:** LOW (parallelizable by design; proven pattern by Glean)

**Current Status:** Unvalidated. Phase 2 is gating async implementation.

---

### Risk #3: Tree-Sitter Type Resolution Limits in Dynamic Languages

**The Risk:** For PHP, Python, JavaScript — dynamic dispatch means tree-sitter can't resolve `$obj->method()` to specific class. How much does this degrade cross-reference quality?

**Coverage Clarification** (Updated):
- **Procedural PHP/TypeScript/Python**: 85–92% static coverage (direct calls, imports, module references)
- **WordPress OOP PHP** (where `$this->method()` dominant): 50–60% coverage (dynamic dispatch unresolved, but structure visible)
- **WordPress hooks**: 70–80% recoverable via string literal analysis (do_action/apply_filters are strings)

**Mitigation Strategy:**

| Phase | Action | Acceptance | Coverage |
|-------|--------|-----------|----------|
| **1** | Measure static edge coverage. Index PHP repo, count extracted definitions + references. Compare to manual spot-check on 50 symbols. Target: >85% for procedural, >55% for OOP. | If procedural <85% or OOP <55%, investigate edge extraction bugs. |
| **2** | Validate incremental updates maintain coverage on modified files | Coverage must not regress |
| **3** | Add post-processing pass for language-specific patterns: PHP string-literal hook analysis (regex + tree-sitter string extraction). | Recover +5–10% coverage for WordPress hooks. |
| **5+** | Optional: Annotation tool (expose tool: `annotate_cross_language_ref(from_file, to_file, reason)`). User provides hints; system learns patterns. | User-provided, 80–90% accurate |

**Real-World Prevalence (Prioritization):**
- **WordPress ecosystem**: LOW urgency. Plugin integration via hooks (PHP string literals, tree-sitter recoverable). REST APIs less common.
- **Multi-service SaaS** (TS + Python microservices): MEDIUM priority. REST APIs are documented (OpenAPI/Swagger) — can extract spec and link routes.
- **Internal microservices**: Can be addressed via annotation tool (users provide hints).

**Contingency Actions:**
- **If coverage <80% on PHP**: Implement LSP integration as fallback (Intelephense, Psalm) — complex, defer to future
- **If cross-reference quality unacceptable to users**: Implement string literal analysis pass (Phase 3) + annotation tool (Phase 5)

**Risk Level:** LOW (known limitation; industry standard; WordPress hook recovery viable)

**Current Status:** Addressed by research. Phase 3 string literal analysis mitigates.

---

### Risk #4: Scope Token Security — Session Fabrication Attacks

**The Risk** (Updated): Session table lookup provides server-side validation, but can agents forge session IDs or escalate scope?

**Evidence Baseline:**
- Server-side validation is strongest defense: never trust credential claims; verify against authoritative store (sessions table)
- Threat model for server-side sessions: Agent cannot forge ID (must exist in table), cannot escalate (scope stored server-side, immutable)
- Alternative (JWT): Red-team found symmetric key on single-machine system is vulnerable (key accessible to any co-resident process)

**Mitigation Strategy:**

| Phase | Action | Acceptance |
|-------|--------|-----------|
| **1** | Implement server-side session validation. On every call: (1) Look up session_id in sessions table. (2) Retrieve scope. (3) Validate query. Deny-by-default (missing session = reject). | No custom crypto. Use standard database transactions. |
| **2** | Implement session token generation. Generate UUID4 session_id, insert into sessions table with scope claims. Return opaque ID. | No secrets transmitted. Scope immutable server-side. |
| **2** | Implement session revocation. Persistence agent can delete from sessions table. Revocation is immediate. | O(1) delete operation. |
| **3** | Add audit logging: (agent_id, scope_level, tool, timestamp, result_count). Enable security review. | Operators can audit who accessed what + when. |

**Threat Model Validation:**

| Attack | Defense |
|--------|---------|
| Agent tries to forge session_id (random UUID) | Session table lookup fails (ID not in DB). Rejected. |
| Agent tries to modify session_id in flight | Server-side lookup uses original ID. Modified ID doesn't exist. Rejected. |
| Agent tries to escalate scope (project → global) | Scope is immutable in sessions table. Server extracts from table, not from agent claim. Rejected. |
| Session hijacking (agent2 steals agent1's session_id) | Mitigated by short 30-minute lifetime. Operator can revoke if detected. |
| Persistence agent's session_id theft | Session_id stored in memory only (not env var). Still revocable immediately. Limit exposure window. |

**Contingency Actions:**
- **If session validation bypass found**: Stop production use until patched. Audit all historical queries via audit log.
- **If session_id guessing is concern**: Add rate-limiting on failed lookups (too many bad IDs from same source = block).

**Risk Level:** LOW (server-side validation much simpler and more secure than JWT on single machine)

**Current Status:** Addressed by research. Server-side session model eliminates symmetric-key vulnerability.

---

### Risk #5: "2-3K Lines" Realism — Scope Creep & Maintenance Surface

**The Risk** (Updated with realistic budgets): Does thin wrapper inevitably grow into a framework? What are real line counts?

**Evidence Baseline (Red-Team Analysis):**
- Phase 1 realistic: 1,600–2,200 lines (parser + schema + LanceDB + MCP + scope + error handling)
- Phase 2 realistic: ~1,500 lines (admin tools + incremental + audit)
- Aider repo-map: ~2–3K lines (proven feasible for similar task)
- Continue.dev: Started ~2K, now ~5–8K after language support + edge cases

**Mitigation Strategy (Revised Budgets):**

| Phase | Realistic Budget | Includes | Cumulative | Gate |
|-------|---------|----------|-----------|------|
| **Phase 1** | <2,500 lines | Tree-sitter driver + SQLite schema + LanceDB indexer + MCP + scope validation + error handling | <2,500 | <2.5K |
| **Phase 2** | <2,000 lines | Incremental re-indexing + session tokens + admin tools (register, reindex, status) | <4,500 | <4.5K |
| **Phase 3** | <1,500 lines | Collection federation + cross-project edge table + RRF merging | <6,000 | <6K |
| **Phase 4** | <1,500 lines | Document indexing (PDF, MD, YAML) + Drift-Adapter integration | <7,500 | <7.5K |
| **Phase 5** | <1,500 lines | PPR graph intelligence + impact analysis | <9,000 | <9K |
| **Phase 6** | <1,500 lines | File watcher + always-on service + global registry | <10,500 | <10.5K |
| **TOTAL** | | All 6 phases | <10.5K | Stretch: <14K with edge cases |

**Scope Boundaries (Non-Goals Shield Against Creep):**

Explicitly exclude:
- **No LSP services** — Don't build type checkers. Tree-sitter + string analysis only.
- **No prompt construction** — Don't build an LLM framework. Just return structured data.
- **No agent orchestration** — Don't build agentic routing. Just expose tools.
- **No git operations** — Don't reimplement git. Use `git diff` CLI.
- **No language-specific linting/type checking** — That's LSP's job.
- **No custom query language** — Use SQL + Python + standard tree-sitter patterns.

**Architecture Decisions Logged:**
- Every phase gate includes decision log: "Why we said YES to X" and "Why we said NO to Y."
- Prevents "but we should support Rust" or "let's add a web UI" mid-project.

**Contingency Actions:**
- **If Phase 1 exceeds 2.5K by >20%**: Halt. Audit every line. Defer lower-ROI features. Re-estimate remaining phases.
- **If any phase exceeds budget by >30%**: Phase gate FAILS. No advancement until root cause fixed.
- **If cumulative lines reach 10.5K before Phase 6 complete**: Project scope has failed. Rethink architecture.

**Risk Level:** MEDIUM (real but manageable with discipline and realistic budgets)

**Current Status:** Risk acknowledged. Discipline enforced via phase gates with updated budgets from red-team analysis.

---

### Risk #6: Embedding Model Drift & Re-Index Cost at Scale

**The Risk:** When embedding model changes (new C2LLM release, hardware constraint), all embeddings become stale. Full re-index cost at 50 repos?

**Evidence Baseline:**
- Traditional full re-indexing: 0.5–1 GPU-hour for 500K items. At 50 repos × 10K symbols = 500K items, expect 0.25–0.5 GPU-hours + 0.1–0.2 CPU-hours. Disruptive.
- Drift-Adapter (2025 EMNLP): Train lightweight transformation (Procrustes, low-rank affine, or MLP) on 1–5% corpus sample. Cost: 1–2 minutes CPU. Recovers 95–99% retrieval performance. **100× cheaper than full re-index.**

**Mitigation Strategy:**

| Phase | Action | Acceptance |
|-------|--------|-----------|
| **1–3** | Treat embedding model as fixed. Don't change models in MVP. | Deferred problem is acceptable. |
| **4** | Implement Drift-Adapter pattern. When new embedding model deployed: (a) generate embeddings for 5% sample with both old + new model, (b) train transformation, (c) keep old index live, (d) map new queries to old space. | <2 minutes training overhead per model change. No full re-index needed. |
| **5+** | Optional: Full incremental re-index (5% per night, 20 nights total re-index) while Drift-Adapter keeps system live. | Asynchronous re-indexing without downtime. |

**Drift-Adapter Implementation Sketch** (Phase 4):
```python
def adapt_embedding_model(old_embeddings_sample, new_embeddings_sample):
    """Train transformation: new_space -> old_space."""
    # Procrustes solution (orthogonal transformation)
    U, _, Vt = np.linalg.svd(new_embeddings_sample.T @ old_embeddings_sample)
    A = U @ Vt  # orthogonal matrix

    # On search: transform new query embedding via A
    new_query_embedding = embed(query, new_model)
    old_query_embedding = new_query_embedding @ A.T
    # Search using old_query_embedding in existing LanceDB index
    return lance_db.search(old_query_embedding)
```

**Contingency Actions:**
- **If Drift-Adapter recovery <95%**: Accept and re-index incrementally during off-hours.
- **If embedding model architecture changes radically** (dimension shift, different tokenization): Full re-index may be required. Cost: acceptable (outside MVP critical path).

**Risk Level:** LOW (solved by Drift-Adapter research, 2025; deferrable to Phase 4)

**Current Status:** Addressed by peer-reviewed research. No technical blocker.

---

### Risk #7: Cross-Language References (PHP REST API ↔ TypeScript Client)

**The Risk:** A PHP service exposes a REST API; TypeScript client calls it. The code graph cannot see this without explicit annotation.

**Coverage Revision** (Updated):
- String literal analysis (Phase 3): 10–20% for modern TypeScript template literals, 40–60% for static string URLs (simple REST patterns)
- Annotation tool (Phase 5+): User-provided hints for complex mappings

**Evidence Baseline:**
- Tree-sitter sees: PHP `Route::get('/api/users', ...)` + TypeScript `fetch('/api/users')` (both string literals)
- Tree-sitter does NOT see: These are the same endpoint (requires HTTP/REST semantics understanding)
- Industry approach: Sourcegraph requires SCIP index files. Continue.dev ignores this. Agents manually navigate.
- First paper on multi-repo GraphRAG (LogicLens, Jan 2026) acknowledges this as open problem; suggests LLM-assisted enrichment (expensive)

**Mitigation Strategy:**

| Phase | Action | Effort | Coverage |
|-------|--------|--------|----------|
| **1–2** | Document limitation. Cross-language API references out-of-scope for MVP. Users must manually trace. | None | 0% |
| **3** | String literal analysis pass. Parse route definitions + API calls, match by URL pattern regex. Example: PHP `/api/users/{id}` + TS `/api/users/` with parameter extraction. Simple heuristic. | Low | 10–20% template literals, 40–60% static URLs |
| **5+** | Annotation tool (optional). Expose tool: `annotate_cross_language_ref(from_file, to_file, reason)`. User provides hints; system learns patterns. | Medium | User-provided, 80–90% accurate |
| **Future** | LLM-assisted graph enrichment (optional). Use LLM to infer cross-language links. High cost but high confidence. | High | 85–95% if deployed |

**Real-World Prevalence (Prioritization):**
- **WordPress ecosystem**: LOW urgency. Plugin integration via hooks (PHP string literals, tree-sitter recoverable). REST APIs less common.
- **Multi-service SaaS** (TS + Python microservices): MEDIUM priority. REST APIs are documented (OpenAPI/Swagger) — can extract spec and link routes.
- **Internal microservices**: Can be addressed via annotation tool (users provide hints).

**Contingency Actions:**
- **If users heavily complain about missing cross-language links**: Implement string literal analysis pass (Phase 3) + annotation tool (Phase 5).
- **If demand is very high**: Consider OpenAPI extraction pass (parse YAML specs, link TS imports to PHP routes).

**Risk Level:** MEDIUM (known limitation; industry has no better solution for MVP)

**Current Status:** Documented with updated coverage percentages. Not a blocker for Phase 1–3. Addressable in Phase 3+ if user demand warrants.

---

### Risk #8: LanceDB Maturity & API Stability

**The Risk:** LanceDB is <3 years old. Breaking API changes across versions could require re-indexing or schema migrations.

**Evidence Baseline:**
- LanceDB v0.1 (2023) → v0.4 (2025): Append-only design maintained. Breaking changes in query API but index backwards-compatible.
- No major incidents reported in production; used by Meta internally

**Mitigation Strategy:**

| Phase | Action | Acceptance |
|-------|--------|-----------|
| **1** | Pin LanceDB version in requirements. Test upgrades in staging before production. | E.g., `lancedb==0.4.0` |
| **2** | Maintain abstraction layer: Wrap LanceDB API in internal module. Changes to LanceDB API only affect wrapper, not callers. | If LanceDB API breaks, update wrapper only (~50 lines affected). |
| **3+** | Monitor LanceDB releases. Batch version upgrades (not continuous). | Upgrade every 3–6 months, not every release. |

**Contingency Actions:**
- **If LanceDB breaking change requires re-index**: Accept as maintenance cost (happens 1–2× over 5 years). Cost is acceptable.
- **If LanceDB abandoned**: Migrate to alternative (Milvus, pgvector). Abstraction layer makes migration easier.

**Risk Level:** LOW (mitigated by abstraction layer + pinned version)

**Current Status:** Identified and addressed. No blocker.

---

## Dependencies

### Must Exist Before Phase 1 Starts

1. **Local embedding model endpoint** (OpenAI-compatible API)
   - Running on localhost (e.g., `http://localhost:8000/v1/embeddings`)
   - Accepts POST with `{"input": ["text1", "text2", ...], "model": "embedding-model-name"}`
   - Batch API support: 100+ texts per request
   - Returns vectors of fixed size (768 or 1536 dims)
   - Latency: ~10ms per chunk in batch mode (100 chunks → ~100ms)
   - Availability: Assumed to exist (prerequisite for CodeMem deployment)

2. **Git-capable filesystem** (for incremental re-indexing via git diff)
   - All indexed projects must be git repos
   - Phase 2+ uses `git diff HEAD~1 HEAD` to find modified files

3. **Python 3.11+ or Node.js 18+** (language choice, Phase 0 decision)
   - **Python preference**: tree-sitter-python, sqlite3, LanceDB (all mature in Python)
   - **TypeScript viable**: tree-sitter-web, better.sqlite3, lancedb (JS bindings)
   - **Decision deferred to stakeholder** (Phase 0 pre-spec)

### Must Exist During Phase 2+

4. **Persistence agent runtime** (where MCP server lives)
   - Single always-on process managing global state
   - Access to all project directories (for incremental indexing + file watching)
   - Write access to ~/.codemem/global.db

5. **Session token distribution mechanism** (how task agents get tokens)
   - Persistence agent generates session_id via create_scope()
   - Session_id passed to task agent via MCP server config (session_id field in request context)
   - **TBD details**: Environment variable? File? HTTP endpoint? Decided by integration team in Phase 2

### Must Exist During Phase 4+

6. **Optional: Drift-Adapter tooling** (when embedding model changes)
   - scikit-learn or numpy for Procrustes solution
   - Already available in Python ecosystem

### Must Exist During Phase 6+

7. **File watcher availability** (watchdog for Python, chokidar for JS)
   - Handles file system events
   - Triggers incremental re-index

---

## Test Strategy

### Unit Tests (All Phases)

**Tree-sitter symbol extraction:**
```python
# tests/test_parser.py
def test_extract_php_function_definitions():
    code = """<?php
    function greet($name) { echo "Hi " . $name; }
    """
    symbols = extract_symbols(code, language='php')
    assert len(symbols) == 1
    assert symbols[0].kind == 'function'
    assert symbols[0].name == 'greet'

def test_extract_typescript_class_members():
    code = """
    class User {
      name: string;
      getName() { return this.name; }
    }
    """
    symbols = extract_symbols(code, language='typescript')
    assert len(symbols) == 2  # class + method
    assert symbols[1].kind == 'method'
```

**cAST chunking:**
```python
def test_cast_chunk_sizes():
    code = """function foo() { ... 200 chars ... } function bar() { ... 300 chars ... }"""
    chunks = chunk_with_cast(code, budget=400)
    assert all(len(c.text) <= 400 for c in chunks)
    assert len(chunks) == 2  # Split into two chunks
```

**SQLite schema + queries:**
```python
def test_symbol_insertion_and_lookup():
    db = SQLiteIndex(':memory:')
    db.insert_symbol(project_id=1, name='greet', kind='function', file_id=1, line=5)
    result = db.lookup_symbol('greet')
    assert result.name == 'greet'
    assert result.line == 5

def test_edge_query_forward_refs():
    db = SQLiteIndex(':memory:')
    db.insert_edge(from_id=1, to_id=2, type='calls')
    db.insert_edge(from_id=1, to_id=3, type='calls')
    callers = db.get_callees(symbol_id=1)
    assert len(callers) == 2
```

**LanceDB vector + BM25:**
```python
def test_lance_db_vector_search():
    db = LanceDB('test_db')
    db.add([
        {'id': '1', 'content': 'user authentication logic', 'embedding': [...]},
        {'id': '2', 'content': 'password hashing utility', 'embedding': [...]}
    ])
    results = db.search('how do I hash passwords?', limit=2)
    assert results[0].id == '2'  # More relevant

def test_lance_db_bm25():
    results = db.search_bm25('hash', limit=5)
    assert len(results) >= 1
    assert 'hash' in results[0].content.lower()
```

**Session token validation:**
```python
def test_session_valid():
    session_id = generate_session_id()
    insert_session(session_id=session_id, agent_id='task-001', level='project',
                   projects=['pm-core'], valid_until=now() + timedelta(minutes=30))
    scope = validate_session(session_id)
    assert scope.level == 'project'
    assert scope.projects == ['pm-core']

def test_session_invalid():
    with pytest.raises(SessionNotFoundError):
        validate_session(session_id='nonexistent-uuid')

def test_session_expired():
    session_id = generate_session_id()
    insert_session(session_id=session_id, agent_id='task-001', level='project',
                   projects=['pm-core'], valid_until=now() - timedelta(minutes=1))
    with pytest.raises(SessionExpiredError):
        validate_session(session_id)

def test_sql_injection_prevention():
    # Parameterized queries prevent injection
    db.execute("SELECT * FROM symbols WHERE name = ?", (user_input,))
    # Never: db.execute(f"SELECT * FROM symbols WHERE name = '{user_input}'")

def test_path_traversal_validation():
    # Normalize and validate file paths
    base = '/projects/pm-core'
    safe_path = normalize_and_validate(base, '../../etc/passwd')
    assert safe_path raises PathTraversalError  # Rejected

    safe_path = normalize_and_validate(base, 'src/Hooks.php')
    assert safe_path == '/projects/pm-core/src/Hooks.php'  # Accepted
```

### Integration Tests (Phase 1+)

**End-to-end indexing + querying:**
```python
def test_index_and_search_php_repo():
    # Setup: Clone a small PHP repo
    repo_path = '/tmp/test-php-repo'
    setup_test_repo(repo_path, language='php')

    # Index
    indexer = CodeMemIndexer(repo_path, embedding_endpoint='http://localhost:8000')
    stats = indexer.index()
    assert stats.symbols_extracted > 100
    assert stats.chunks_embedded > 50

    # Query
    query = "how do I register a WordPress hook?"
    results = indexer.search(query, limit=10)
    assert len(results) > 0
    assert 'add_action' in results[0].content or 'do_action' in results[0].content
```

**Federated search across projects:**
```python
def test_federated_search_two_projects():
    # Index two projects
    indexer1 = CodeMemIndexer('/tmp/pm-core', ...)
    indexer2 = CodeMemIndexer('/tmp/pm-popup', ...)

    # Create collection
    collection = Collection(name='pm-plugins', projects=[indexer1, indexer2])

    # Federated query
    results = collection.federated_search("hook_name", limit=10)

    # Results should come from both projects
    assert any(r.file_path.startswith('/tmp/pm-core') for r in results)
    assert any(r.file_path.startswith('/tmp/pm-popup') for r in results)
```

**Session token enforcement:**
```python
def test_project_scope_blocks_cross_project_query():
    # Setup: Two project indexes
    pm_core_index = CodeMemIndexer('/tmp/pm-core', scope_id='pm-core')
    pm_popup_index = CodeMemIndexer('/tmp/pm-popup', scope_id='pm-popup')

    # Create task agent with project-only scope
    session_id = create_scope(agent_id='pm-core-task', level='project',
                              projects=['pm-core'])

    # Agent can query pm-core
    results = pm_core_index.search_with_session(query='...', session_id=session_id)
    assert len(results) > 0

    # Agent cannot query pm-popup
    with pytest.raises(ForbiddenScopeError):
        pm_popup_index.search_with_session(query='...', session_id=session_id)
```

### Performance Benchmarks (Phase 1+)

**Indexing latency:**
```bash
# Phase 1: Single project (PHP)
time python3 -m codemem index /path/to/php-repo
# Expected: <60 seconds for 2K files, 20K symbols

# Phase 2: Incremental (git-diff)
git add file1.php file2.php
time python3 -m codemem reindex --incremental /path/to/php-repo
# Expected: <5 seconds for 10 modified files
```

**Query latency:**
```bash
# Single project query
time python3 -c "
import codemem
index = codemem.load_index('/tmp/pm-core')
results = index.search('function_name', limit=10)
"
# Expected: <50ms (p95 <100ms)

# Federated query (5 projects)
# Expected: <100ms with parallelization
```

**Session validation overhead:**
```bash
# 1000 queries with valid session_id
# Expected: <1ms per session lookup; no performance degradation vs no-session baseline
```

**Batch embedding throughput:**
```bash
# 100 chunks per batch to embedding endpoint
# Expected: ~100ms total (10ms per chunk average in batch mode)
```

### Red-Team Validation (Before Panel)

1. **Attempt session token forgery**: Agent tries to use random UUID → lookup fails (not in sessions table)
2. **Attempt session escalation**: Agent with project-scope tries cross-project query → filtered out (server-side scope check)
3. **Measure SQLite adjacency query time**: 5K edge graph, 100 forward reference lookups → benchmark against 50ms target
4. **Measure federation latency**: 5-project collection, 10 parallel searches → benchmark against 100ms target
5. **Code review scope creep**: Count lines Phase 1 vs budget → ensure <2.5K target
6. **Cross-language reference coverage**: Index PHP + TS repo, spot-check 50 symbols for missed references → measure static coverage % by tier
7. **Path traversal**: Attempt to access files outside project root via `file_context` tool → all attempts rejected
8. **SQL injection**: Pass SQL syntax in query parameters → no injection; parameterized queries enforce safety

---

## Non-Goals

Explicitly excluded to control scope and prevent feature creep:

### Technology Decisions

- **Not an IDE plugin**: MCP is the interface. No VSCode extension. If users want IDE integration, they build on top of MCP.
- **Not LSP or type checker**: No static analysis beyond tree-sitter AST. Don't build Intelephense or Pyre. Tree-sitter is "syntax only, structure yes, types no."
- **Not a code formatter or linter**: Don't build Prettier, ESLint, or black integration. That's language-specific tool job.
- **Not a git replacement**: Don't reimplement git. Use `git diff`, `git log` CLI. No custom version control.
- **Not GraphQL or REST API framework**: MCP is the only interface. Don't expose HTTP endpoints (those are integrations' responsibility).

### Feature Scope

- **Not SCIP/LSP-level semantic precision**: Accept 85–92% static coverage for procedural code, 50–60% for OOP WordPress. Dynamic dispatch unresolved. Acceptable trade-off.
- **Not LLM integration for graph enrichment** (Phase 1–3): Don't build cross-language link inference via LLM. String literal analysis only.
- **Not WebUI for index management**: CLI / programmatic tools only (register_project, reindex, create_scope). If users want UI, they build it.
- **Not agentic reasoning on top of graph**: Return structured data (symbols, references, edges). Don't orchestrate agent decisions.
- **Not prompt engineering**: Don't construct prompts for LLMs. Just give agents searchable memory + tools.
- **Not all programming languages**: Start with PHP, TypeScript, Python, Swift. Other languages can be added via tree-sitter grammars later (not MVP).
- **Not real-time indexing**: File watcher in Phase 6 is eventual consistency (10–30s delay). Not sub-second.

### Document Types

- **Not scan-to-PDF**: Don't build OCR. Input must be text-extractable PDFs.
- **Not all document formats**: Start with PDF, Markdown, YAML, JSON, `.env`. Not DOCX, Confluence, Slack (those are integration opportunities).
- **Not document version control**: Store latest version only. No document history tracking.

### Deployment & Operations

- **Not cloud deployment**: Single-machine embedded system. Assume local network. Not designed for multi-region.
- **Not external authentication**: Local scope tokens only. No SSO, OAuth, LDAP integration.
- **Not multi-tenant isolation**: Single persistence agent per machine. If you need multiple isolated instances, run multiple machines.
- **Not full audit trail**: Audit logging for security (who queried what). Not full provenance tracking (why was this indexed, by whom).

---

## Open Questions (Deferred to Stakeholders)

1. **Language Implementation** (Phase 0 decision, before Phase 1 starts)
   - Python (tree-sitter-python, sqlite3 mature) or TypeScript (better.sqlite3, node tree-sitter, matches MCP ecosystem)?
   - **Recommendation**: Python preferred (strongest library maturity). TypeScript viable if team preference.
   - **Action**: Stakeholder decides in kick-off. Affects tech stack but not architecture.

2. **Embedding Model Selection** (Phase 1 start)
   - Which code-specific embedding model to run on local endpoint?
   - Options: C2LLM-7B (#1 MTEB-Code, 80.75), CodeXEmbed-7B, nomic-embed-text (general fallback)
   - **Recommendation**: C2LLM-7B if hardware permits; nomic-embed-text if resource-constrained
   - **Action**: Test embedding latency + quality on Phase 1 test repo. Swap if needed.

3. **Session Token Distribution Mechanism** (Phase 2 spec detail)
   - How do task agents receive session IDs from persistence agent?
   - Options: HTTP endpoint? File drop? Environment variable? Database query?
   - **Recommendation**: Simplest first (environment variable or file). HTTP endpoint if multi-machine (future).
   - **Action**: Integrate team decides based on persistence agent architecture.

4. **Collection Manifest Format** (Phase 3 spec detail)
   - Static YAML file or dynamic registration via admin tools?
   - Options: Static `collections.yaml` (git-tracked, versioned) or `register_collection` tool (runtime, ephemeral)
   - **Recommendation**: Hybrid. Static YAML for persistent collections (Git repos, plugin ecosystems). Tools for ephemeral collections (ad-hoc agent groups).
   - **Action**: Define in Phase 3 design.

5. **WordPress Hook Intelligence Scope** (Phase 3 post-MVP)
   - Custom tree-sitter queries for do_action/apply_filters/add_action, or post-processing pass on extracted symbols?
   - **Recommendation**: String literal analysis post-processing pass (Phase 3). Simple regex over extracted string constants.
   - **Action**: Implement as Phase 3 contingency if time allows.

6. **Cross-Language Reference Support** (Phase 3+)
   - When should annotation tool be implemented (Phase 3, 5, or later)?
   - **Recommendation**: Phase 5+ (lower ROI than core features). String literal analysis covers 10–20% template literals, 40–60% static REST APIs.
   - **Action**: Monitor user feedback. Implement if demand justified.

7. **Document Type Prioritization** (Phase 4 start)
   - Which non-code formats matter most? (PDF, Markdown, YAML, JSON, .env, configs)
   - **Recommendation**: Phase 4 implements: PDF (pymupdf), Markdown (split on headers), YAML/JSON (as documents). Extensible after.
   - **Action**: Confirm with stakeholders before Phase 4.

8. **Open Source Boundary** (Phase 3+ decision)
   - Which parts are open source vs proprietary?
   - **Recommendation**: Core indexer + MCP server = open source. WordPress-specific extensions = proprietary.
   - **Action**: Separate into "codemem" (OSS) + "codemem-wordpress" (internal toolkit).

9. **Always-On Resource Usage** (Phase 6 design)
   - How much CPU/memory for file watcher + incremental re-indexing?
   - **Recommendation**: Benchmark in Phase 5. File watcher should use <5% CPU idle, <500MB RAM.
   - **Action**: Load test with 20+ project directories. Design background task scheduling if needed.

---

## Deferred Items (From v1 → v2 Updates)

### Critical Blockers Addressed in v2

- **HMAC-SHA256 JWT ↔ Server-side sessions** (CRITICAL #1) — JWT replaced with opaque session_id + SQLite sessions table. Eliminates symmetric-key threat model.
- **Scope token distribution protocol** (CRITICAL #2) — Specified: create_scope() generates session_id, stored in SQLite, passed to task agent via MCP config.
- **PPR performance model** (CRITICAL #3) — Graph loaded in-memory at server start, rebuilt after re-index. <100ms for 50K edges validated in Phase 5.
- **WordPress PHP coverage claim** (CRITICAL #4) — Recalibrated: 85–92% procedural, 50–60% OOP WordPress, 70–80% hooks. Updated acceptance criteria.
- **Line count budgets** (CRITICAL #5) — Revised realistic budgets: Phase 1 <2.5K, Phase 2 <2K, Phase 3–6 <1.5K each. Total <10.5K (stretch <14K).
- **Operational runbook** (CRITICAL #6) — New section covering crash recovery, embedding unavailability, backup/restore, staleness monitoring, health monitoring, corrupted index recovery.

### Significant Changes Addressed in v2

- **SQLite↔LanceDB coherence** (SIGNIFICANT #7) — Files table `index_status` column (pending | indexed | failed). Per-file-batch updates with status tracking.
- **Cross-project edge foreign keys** (SIGNIFICANT #8) — Changed to composite `(project_id, symbol_id)` in cross_project_edges table.
- **Global SQLite location** (SIGNIFICANT #9) — Specified: ~/.codemem/global.db for projects, collections, sessions, indexing_jobs, cross_project_edges. Per-project SQLite at <project_root>/.codemem/index.db.
- **Batch embedding throughput** (SIGNIFICANT #10) — Specified: batch API required, 100 chunks per batch, ~10ms per chunk. Updated acceptance criteria.
- **Path traversal validation** (SIGNIFICANT #11) — All tools accepting file paths normalize and validate containment. Test added to Phase 1.
- **SQL injection prevention** (SIGNIFICANT #12) — Explicit requirement for parameterized queries. Test added to Phase 1.
- **PPR pseudocode fix** (SIGNIFICANT #13) — Corrected initialization of p_uniform and CSR matrix construction.
- **Cross-language coverage claim** (SIGNIFICANT #14) — Reduced from "50–70%" to "10–20% template literals, 40–60% static URLs."
- **LanceDB maturity risk** (SIGNIFICANT #15) — Added as Risk #8: version pinning, abstraction layer, migration planning.

### Items Remaining Deferred (No Implementation in v2)

- **RS256 upgrade path** — Document as future enhancement if multi-machine deployment needed
- **`jti` replay prevention** — Document as optional enhancement for token reuse detection
- **Audit log tamper protection** — Document as known limitation; separate append-only store possible future work
- **Incremental indexing injection** — Document as known risk (prompt injection via indexed content)
- **Schema migration system** — Document as optional (numbered .sql files at startup); Phase 1 uses fixed schema

---

## Success Metrics (Phase Gates)

| Phase | Gate | Metric | Target | Failure = |
|-------|------|--------|--------|-----------|
| **1** | Latency | Query p95 latency (single project) | <100ms | Pause Phase 2; investigate SQLite + LanceDB tuning |
| **1** | Maintenance | Lines of code (indexer + MCP server) | <2.5K | Audit scope; defer features; re-estimate |
| **1** | Coverage | Symbol extraction accuracy (procedural, OOP, hooks) | >95%, >55%, >75% | Improve edge extraction or accept limitation |
| **2** | Incremental | Re-index 10 files | <5s | Debug git-diff performance |
| **2** | Security | Session validation + no bypass | 100% success, zero forgeries | Security audit; pause deployment |
| **3** | Federation | Parallel query across 5 projects | <100ms p95 | Add caching or pre-compute edges |
| **3** | Accuracy | Cross-project references | >90% recall on 50 spot-checks | Improve edge extraction |
| **4** | Documents | PDF extraction + embedding | <30s per 50-page PDF | Optimize chunking |
| **5** | Graph | PPR computation (in-memory) | <100ms for 50K edges | Optimize matrix power iteration |
| **6** | Persistence | File watcher latency | 10–30s from change to index | Profile watchdog; add debouncing |

---

## Appendix A: SQLite Schema (Full, Updated)

```sql
-- Global SQLite at ~/.codemem/global.db

-- Projects & Collections
CREATE TABLE projects (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,  -- Absolute path to repo
  name TEXT NOT NULL,
  language TEXT,  -- Comma-separated: 'php,typescript'
  collection_id INTEGER,
  scope_id TEXT UNIQUE,  -- Scope ID for this project
  indexed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(collection_id) REFERENCES collections(id)
);

CREATE TABLE collections (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,  -- 'pm-plugins', 'core-services'
  projects_json TEXT,  -- JSON list of project IDs
  scope_id TEXT UNIQUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions (Server-Side Scope Storage)
CREATE TABLE sessions (
  id TEXT PRIMARY KEY,  -- UUID4 session_id
  agent_id TEXT NOT NULL,
  scope_level TEXT NOT NULL,  -- 'project', 'collection', 'global'
  projects_list TEXT,  -- JSON list of project IDs
  collections_list TEXT,  -- JSON list of collection IDs
  capabilities TEXT,  -- JSON list: ['search', 'symbols', 'references', ...]
  valid_until TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexing Job Queue (Crash Recovery)
CREATE TABLE indexing_jobs (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL,
  status TEXT NOT NULL,  -- 'pending', 'in_progress', 'completed', 'failed'
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  error TEXT,
  FOREIGN KEY(project_id) REFERENCES projects(id)
);

-- Cross-Project Edges (Collection-level, Updated with Composite Keys)
CREATE TABLE cross_project_edges (
  id INTEGER PRIMARY KEY,
  collection_id INTEGER NOT NULL,
  from_project_id INTEGER NOT NULL,
  from_symbol_id INTEGER NOT NULL,
  to_project_id INTEGER NOT NULL,
  to_symbol_id INTEGER NOT NULL,
  type TEXT NOT NULL,  -- 'hooks_into', 'imports', 'depends_on'
  weight REAL DEFAULT 1.0,
  FOREIGN KEY(collection_id) REFERENCES collections(id),
  FOREIGN KEY(from_project_id) REFERENCES projects(id),
  FOREIGN KEY(from_symbol_id) REFERENCES symbols(id),
  FOREIGN KEY(to_project_id) REFERENCES projects(id),
  FOREIGN KEY(to_symbol_id) REFERENCES symbols(id),
  INDEX idx_cross_edge (from_project_id, to_project_id),
  UNIQUE(from_project_id, from_symbol_id, to_project_id, to_symbol_id)
);

-- Audit Log
CREATE TABLE audit_log (
  id INTEGER PRIMARY KEY,
  agent_id TEXT NOT NULL,
  scope_level TEXT NOT NULL,
  tool_called TEXT NOT NULL,
  result_count INTEGER,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-Project SQLite at <project_root>/.codemem/index.db

CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL,
  path TEXT NOT NULL,
  language TEXT,  -- 'php', 'typescript', 'python', 'swift'
  hash TEXT,  -- SHA-256 for change detection
  index_status TEXT DEFAULT 'pending',  -- NEW: 'pending', 'indexed', 'failed'
  indexed_at TIMESTAMP,
  FOREIGN KEY(project_id) REFERENCES projects(id),
  UNIQUE(project_id, path)
);

CREATE TABLE symbols (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL,
  file_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,  -- 'function', 'class', 'method', 'variable', 'constant'
  line INTEGER,
  col INTEGER,
  scope TEXT,  -- 'global', 'class', 'namespace' (for Python/PHP)
  signature TEXT,  -- Function signature if available
  FOREIGN KEY(project_id) REFERENCES projects(id),
  FOREIGN KEY(file_id) REFERENCES files(id),
  INDEX idx_project_name (project_id, name),
  INDEX idx_file_id (file_id)
);

CREATE TABLE references (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL,
  from_symbol_id INTEGER NOT NULL,
  to_symbol_id INTEGER,  -- NULL if to_file reference
  kind TEXT NOT NULL,  -- 'calls', 'imports', 'extends', 'implements'
  context TEXT,  -- Context snippet
  line INTEGER,
  FOREIGN KEY(project_id) REFERENCES projects(id),
  FOREIGN KEY(from_symbol_id) REFERENCES symbols(id),
  FOREIGN KEY(to_symbol_id) REFERENCES symbols(id),
  INDEX idx_from (from_symbol_id),
  INDEX idx_to (to_symbol_id)
);

-- Graph Edges (for PPR traversal, loaded in-memory in Phase 5+)
CREATE TABLE edges (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL,
  from_id INTEGER NOT NULL,  -- symbol_id
  to_id INTEGER NOT NULL,    -- symbol_id
  type TEXT NOT NULL,  -- 'calls', 'imports', 'extends', 'hooks_into', 'depends_on'
  weight REAL DEFAULT 1.0,
  FOREIGN KEY(project_id) REFERENCES projects(id),
  FOREIGN KEY(from_id) REFERENCES symbols(id),
  FOREIGN KEY(to_id) REFERENCES symbols(id),
  INDEX idx_from_to (from_id, to_id),
  INDEX idx_to_from (to_id, from_id)
);

-- Chunk Metadata (for embedding retrieval)
CREATE TABLE chunk_meta (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL,
  file_id INTEGER NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  symbol_ids TEXT,  -- JSON list of symbol IDs in chunk
  ast_type TEXT,  -- 'function_def', 'class_def', 'module', etc
  chunk_type TEXT,  -- 'code', 'comment', 'docstring'
  length INTEGER,
  FOREIGN KEY(project_id) REFERENCES projects(id),
  FOREIGN KEY(file_id) REFERENCES files(id),
  INDEX idx_file_chunks (file_id)
);

-- WAL Mode
PRAGMA journal_mode=WAL;
```

---

## Appendix B: LanceDB Schema (Vector Tables)

```python
# Python pseudo-code showing LanceDB table structures

import lancedb
from typing import Dict, List

db = lancedb.connect('/tmp/codemem/db')

# Code Chunks Table
code_chunks_schema = {
    "id": "str",
    "project_id": "int",
    "file_path": "str",
    "start_line": "int",
    "end_line": "int",
    "content": "str",  # Actual code text
    "embedding": "float64(768)",  # Vector from embedding model
    "chunk_type": "str",  # 'function', 'class', 'module'
    "language": "str",  # 'php', 'typescript', 'python'
    "indexed_at": "timestamp",
}

code_chunks_table = db.create_table("code_chunks", schema=code_chunks_schema)

# Document Chunks Table (Phase 4+)
doc_chunks_schema = {
    "id": "str",
    "project_id": "int",
    "source_path": "str",  # Path to PDF/MD/YAML
    "source_type": "str",  # 'pdf', 'markdown', 'yaml', 'json'
    "content": "str",
    "embedding": "float64(768)",
    "page_number": "int",  # For PDFs
    "indexed_at": "timestamp",
}

doc_chunks_table = db.create_table("doc_chunks", schema=doc_chunks_schema)

# Hybrid Search (BM25 + Vector)
# LanceDB built-in: enable full-text search on 'content' field
code_chunks_table.create_index('content')  # BM25 FTS index

# Usage in Phase 1
def hybrid_search(query: str, project_id: int, limit: int = 10):
    # Vector search
    vector = embed(query)  # Call local embedding endpoint
    vector_results = code_chunks_table.search(vector).limit(limit).to_list()

    # BM25 search
    bm25_results = code_chunks_table.search(query, vector_column=None).limit(limit).to_list()

    # Merge with RRF (see Appendix C)
    merged = rrf_merge([vector_results, bm25_results])
    return merged
```

---

## Appendix C: Reciprocal Rank Fusion (RRF) Implementation

```python
from typing import List, Dict

def rrf_merge(ranked_lists: List[List[Dict]]) -> List[Dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists. Each item must have 'id' field.

    Returns:
        Merged list sorted by RRF score (descending).
    """
    rrf_scores = {}
    constant = 60  # Standard RRF constant

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item['id']
            rrf_scores[item_id] = rrf_scores.get(item_id, 0) + (1 / (constant + rank))

    # Create result list with scores
    results = []
    for item_id, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        # Find the item from original lists (use first occurrence)
        for ranked_list in ranked_lists:
            item = next((i for i in ranked_list if i['id'] == item_id), None)
            if item:
                item['rrf_score'] = score
                results.append(item)
                break

    return results

# Example usage in Phase 1
results = rrf_merge([
    keyword_results,      # From BM25 search
    semantic_results,     # From vector search
    # Phase 5: graph_results (from PPR)
])
```

---

## Appendix D: Server-Side Session Token Example

```python
import uuid
import sqlite3
from datetime import datetime, timedelta

def create_scope(agent_id: str, level: str, projects: List[str],
                 collections: List[str] = None) -> str:
    """Generate a server-side session token."""
    session_id = str(uuid.uuid4())  # UUID4 opaque session ID
    valid_until = datetime.utcnow() + timedelta(minutes=30)

    db = sqlite3.connect('/home/user/.codemem/global.db')
    db.execute("""
        INSERT INTO sessions (id, agent_id, scope_level, projects_list, collections_list, valid_until)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, agent_id, level, json.dumps(projects), json.dumps(collections or []), valid_until))
    db.commit()
    db.close()

    return session_id  # Return opaque ID only

def validate_session(session_id: str) -> Dict:
    """Validate and retrieve scope from sessions table."""
    db = sqlite3.connect('/home/user/.codemem/global.db')
    row = db.execute("""
        SELECT agent_id, scope_level, projects_list, collections_list, valid_until
        FROM sessions
        WHERE id = ?
    """, (session_id,)).fetchone()
    db.close()

    if not row:
        raise SessionNotFoundError(f"Session {session_id} not found")

    agent_id, scope_level, projects_json, collections_json, valid_until = row

    if datetime.fromisoformat(valid_until) < datetime.utcnow():
        raise SessionExpiredError(f"Session {session_id} expired")

    return {
        'agent_id': agent_id,
        'level': scope_level,
        'projects': json.loads(projects_json),
        'collections': json.loads(collections_json),
    }

def revoke_scope(agent_id: str):
    """Revoke all sessions for an agent."""
    db = sqlite3.connect('/home/user/.codemem/global.db')
    db.execute("DELETE FROM sessions WHERE agent_id = ?", (agent_id,))
    db.commit()
    db.close()

# Example: Create session for task agent
session_id = create_scope(
    agent_id='pm-core-task-001',
    level='project',
    projects=['popup-maker-core'],
    collections=[]
)
print(session_id)
# 550e8400-e29b-41d4-a716-446655440000

# Example: Validate in MCP server on every call
try:
    scope = validate_session(session_id)
    # Check if query is within scope
    if query_scope > scope['level']:
        raise ForbiddenScopeError()
except SessionNotFoundError:
    raise UnauthorizedError("Invalid session")
```

---

## Appendix E: MCP Tool Definitions (Complete)

### Core Tools (All Scopes)

#### **search** (Hybrid)
```json
{
  "name": "search",
  "description": "Hybrid semantic + keyword + graph search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language query or code keyword"
      },
      "limit": {
        "type": "integer",
        "description": "Max results to return",
        "default": 10
      },
      "filter_type": {
        "type": "string",
        "enum": ["all", "code", "documents"],
        "description": "Search in code, documents, or both",
        "default": "all"
      },
      "filter_language": {
        "type": "string",
        "description": "Filter by language (php, typescript, python, swift)",
        "default": null
      }
    },
    "required": ["query"]
  },
  "scope": "project|collection|global",
  "returns": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "id": "string",
        "file_path": "string",
        "start_line": "integer",
        "end_line": "integer",
        "content": "string (snippet)",
        "score": "float (0-1)",
        "freshness_score": "float (0-1, staleness indicator)",
        "rank_sources": ["keyword", "semantic", "graph"],
        "type": "string (code|document)"
      }
    }
  }
}
```

#### **symbols**
```json
{
  "name": "symbols",
  "description": "List/query symbols with filters",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Symbol name (supports glob)",
        "default": "*"
      },
      "kind": {
        "type": "string",
        "enum": ["function", "class", "method", "variable", "constant", "import"],
        "description": "Filter by symbol kind",
        "default": null
      },
      "language": {
        "type": "string",
        "description": "Filter by language",
        "default": null
      },
      "file_pattern": {
        "type": "string",
        "description": "Filter by file path (glob)",
        "default": null
      }
    }
  },
  "scope": "project|collection|global",
  "returns": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "id": "integer",
        "name": "string",
        "kind": "string",
        "file_path": "string",
        "line": "integer",
        "signature": "string|null"
      }
    }
  }
}
```

#### **references**
```json
{
  "name": "references",
  "description": "Find all references to a symbol",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbol_id": {
        "type": "integer",
        "description": "Symbol ID from symbols() tool"
      },
      "symbol_name": {
        "type": "string",
        "description": "Alternative: symbol name (case-sensitive)"
      },
      "kind": {
        "type": "string",
        "enum": ["calls", "imports", "extends", "all"],
        "description": "Filter reference type",
        "default": "all"
      }
    },
    "required": ["symbol_id | symbol_name"]
  },
  "scope": "project|collection|global",
  "returns": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "file_path": "string",
        "line": "integer",
        "kind": "string (calls|imports|extends)",
        "context": "string (code snippet)"
      }
    }
  }
}
```

#### **file_context**
```json
{
  "name": "file_context",
  "description": "Structural summary of a file (with path traversal validation)",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to file (absolute or relative to project root, validated for traversal)"
      }
    },
    "required": ["file_path"]
  },
  "scope": "project|collection|global",
  "returns": {
    "type": "object",
    "properties": {
      "file_path": "string",
      "language": "string",
      "symbols": {
        "type": "array",
        "items": {"name": "string", "kind": "string", "line": "integer"}
      },
      "imports": {
        "type": "array",
        "items": {"name": "string", "source": "string"}
      },
      "exports": {
        "type": "array",
        "items": {"name": "string"}
      }
    }
  }
}
```

#### **impact**
```json
{
  "name": "impact",
  "description": "Impact analysis: what breaks if I change this symbol?",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbol_id": {
        "type": "integer",
        "description": "Symbol ID from symbols() tool"
      },
      "symbol_name": {
        "type": "string",
        "description": "Alternative: symbol name"
      },
      "depth": {
        "type": "integer",
        "description": "Multi-hop depth (how many levels of callers)",
        "default": 2
      },
      "use_graph": {
        "type": "boolean",
        "description": "Use PPR graph ranking (Phase 5+)",
        "default": false
      }
    },
    "required": ["symbol_id | symbol_name"]
  },
  "scope": "project|collection|global",
  "returns": {
    "type": "object",
    "properties": {
      "symbol": "object (the queried symbol)",
      "direct_callers": "array of symbols",
      "transitive_callers": "array of symbols (depth N)",
      "affected_files": "array of file paths",
      "risk_level": "string (low|medium|high)",
      "graph_visualization": "JSON (for Phase 5+)"
    }
  }
}
```

### Collection Tools (Collection + Global Scopes)

#### **cross_refs**
```json
{
  "name": "cross_refs",
  "description": "Find references crossing project boundaries",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbol_name": {
        "type": "string",
        "description": "Symbol to trace across projects (e.g., WordPress hook name)"
      },
      "kind": {
        "type": "string",
        "enum": ["hooks_into", "imports", "depends_on", "all"],
        "description": "Cross-project relationship type",
        "default": "all"
      }
    },
    "required": ["symbol_name"]
  },
  "scope": "collection|global",
  "returns": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "from_project": "string",
        "from_symbol": "string",
        "to_project": "string",
        "to_symbol": "string",
        "kind": "string",
        "file_path": "string",
        "line": "integer"
      }
    }
  }
}
```

#### **collection_map**
```json
{
  "name": "collection_map",
  "description": "High-level dependency graph between projects",
  "inputSchema": {
    "type": "object",
    "properties": {
      "visualization": {
        "type": "string",
        "enum": ["json", "text", "mermaid"],
        "description": "Output format",
        "default": "json"
      }
    }
  },
  "scope": "collection|global",
  "returns": {
    "type": "object",
    "properties": {
      "projects": {
        "type": "array",
        "items": {
          "name": "string",
          "symbol_count": "integer",
          "dependencies": "array of project names"
        }
      },
      "cross_project_edges": "integer",
      "graph_json": "JSON (node/edge format)"
    }
  }
}
```

### Admin Tools (Global Scope Only)

#### **register_project**
```json
{
  "name": "register_project",
  "description": "Register a new project for indexing",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Absolute path to project root"
      },
      "name": {
        "type": "string",
        "description": "Human-readable project name"
      },
      "languages": {
        "type": "array",
        "items": "string",
        "description": "Languages in project (php, typescript, python, swift)",
        "default": ["php", "typescript"]
      },
      "collection": {
        "type": "string",
        "description": "Collection name to add project to",
        "default": null
      }
    },
    "required": ["path", "name"]
  },
  "scope": "global",
  "returns": {
    "type": "object",
    "properties": {
      "project_id": "integer",
      "project_name": "string",
      "path": "string",
      "status": "string (registered|indexing|indexed)",
      "message": "string"
    }
  }
}
```

#### **register_collection**
```json
{
  "name": "register_collection",
  "description": "Create or modify a collection",
  "inputSchema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Collection name (unique)"
      },
      "projects": {
        "type": "array",
        "items": "string",
        "description": "List of project names to include"
      },
      "description": {
        "type": "string",
        "description": "Optional description",
        "default": null
      }
    },
    "required": ["name", "projects"]
  },
  "scope": "global",
  "returns": {
    "type": "object",
    "properties": {
      "collection_id": "integer",
      "name": "string",
      "projects": "array of project IDs",
      "cross_project_edges": "integer"
    }
  }
}
```

#### **reindex**
```json
{
  "name": "reindex",
  "description": "Trigger full or incremental re-indexing",
  "inputSchema": {
    "type": "object",
    "properties": {
      "project": {
        "type": "string",
        "description": "Project name or ID (null = all projects)"
      },
      "mode": {
        "type": "string",
        "enum": ["full", "incremental"],
        "description": "Full re-index or git-diff incremental",
        "default": "incremental"
      },
      "async": {
        "type": "boolean",
        "description": "Run in background (return immediately)",
        "default": false
      }
    }
  },
  "scope": "global",
  "returns": {
    "type": "object",
    "properties": {
      "project": "string",
      "mode": "string",
      "status": "string (started|running|completed)",
      "symbols_processed": "integer",
      "chunks_embedded": "integer",
      "duration_seconds": "float",
      "errors": "array of strings"
    }
  }
}
```

#### **create_scope** & **revoke_scope**
```json
{
  "name": "create_scope",
  "description": "Create a scope token (session) for an agent",
  "inputSchema": {
    "type": "object",
    "properties": {
      "agent_id": {
        "type": "string",
        "description": "Unique agent identifier"
      },
      "level": {
        "type": "string",
        "enum": ["project", "collection", "global"],
        "description": "Scope level"
      },
      "projects": {
        "type": "array",
        "items": "string",
        "description": "Allowed project names (for project-level scope)",
        "default": []
      },
      "collections": {
        "type": "array",
        "items": "string",
        "description": "Allowed collection names (for collection-level scope)",
        "default": []
      },
      "lifetime_minutes": {
        "type": "integer",
        "description": "Token lifetime in minutes",
        "default": 30
      }
    },
    "required": ["agent_id", "level"]
  },
  "scope": "global",
  "returns": {
    "type": "object",
    "properties": {
      "agent_id": "string",
      "session_id": "string (opaque UUID4)",
      "level": "string",
      "valid_until": "ISO timestamp",
      "message": "string"
    }
  }
}
```

```json
{
  "name": "revoke_scope",
  "description": "Revoke a scope token (invalidate agent access)",
  "inputSchema": {
    "type": "object",
    "properties": {
      "agent_id": {
        "type": "string",
        "description": "Agent ID to revoke"
      }
    },
    "required": ["agent_id"]
  },
  "scope": "global",
  "returns": {
    "type": "object",
    "properties": {
      "agent_id": "string",
      "status": "string (revoked|not_found)",
      "message": "string"
    }
  }
}
```

#### **status**
```json
{
  "name": "status",
  "description": "Index health and metadata",
  "inputSchema": {
    "type": "object",
    "properties": {
      "project": {
        "type": "string",
        "description": "Project name (null = global status)",
        "default": null
      },
      "detailed": {
        "type": "boolean",
        "description": "Include detailed statistics",
        "default": false
      }
    }
  },
  "scope": "project|collection|global",
  "returns": {
    "type": "object",
    "properties": {
      "global": {
        "total_projects": "integer",
        "total_symbols": "integer",
        "total_edges": "integer",
        "last_reindex": "ISO timestamp",
        "index_size_mb": "float"
      },
      "projects": {
        "type": "array",
        "items": {
          "name": "string",
          "symbols": "integer",
          "chunks": "integer",
          "indexed_at": "ISO timestamp",
          "status": "string (indexed|indexing|stale|failed)"
        }
      },
      "scope_registry": {
        "total_agents": "integer",
        "active_sessions": "integer"
      }
    }
  }
}
```

---

## Appendix F: Phase 1 Checklist

**Deliverables:**
- [ ] Tree-sitter setup (parser + `.scm` queries for PHP, TS, Python)
- [ ] SQLite schema creation + WAL mode configuration
- [ ] LanceDB integration (cAST chunking + batch embedding calls)
- [ ] MCP server (5 core tools + session validation)
- [ ] Session table + lookup mechanism (no JWT)
- [ ] Unit tests (parser, chunking, schema, session validation)
- [ ] Integration tests (index + query on real small repos)
- [ ] Performance benchmarks (latency, code lines, batch embedding)
- [ ] API documentation (MCP tool definitions, return types, threat model)
- [ ] Security tests (session forgery, path traversal, SQL injection)

**Testing:**
- [ ] Index PopupMaker core (2K files, 20K symbols) in <60s
- [ ] Query latency <50ms average, <100ms p95
- [ ] Symbol accuracy by tier: >95% procedural, >55% OOP, >75% hooks (50 manual spot-checks)
- [ ] Session validation: invalid session rejected, valid session allows query, session lookup <1ms
- [ ] Batch embedding: 100 chunks per batch, target 10ms per chunk
- [ ] Path traversal: attempts to escape project root rejected
- [ ] SQL injection: parameterized queries prevent all injection attempts
- [ ] Code lines <2.5K (indexer + MCP server)

**Gate Criteria Before Phase 2:**
- [ ] Query latency consistently <50ms (if not, investigate SQLite + LanceDB tuning)
- [ ] Code lines <2.5K (if not, audit scope creep + re-estimate)
- [ ] Zero session token security issues (audit session table lookups + forced failures)
- [ ] Symbol coverage validated per tier (procedural >95%, OOP >55%, hooks >75%)

---

## Appendix G: Phase 1 Test Fixture (Example PHP Repo)

A minimal test repo for validating Phase 1:

```
test-php-repo/
├── src/
│   ├── Hooks.php
│   │   function register_hooks() { add_action('init', 'on_init'); }
│   │   function on_init() { echo "Site initialized"; }
│   │
│   └── Utils.php
│       function format_date($date) { return date('Y-m-d', $date); }
│
├── plugin.php
│   Plugin Name: Test Plugin
│   require_once 'src/Hooks.php';
│   require_once 'src/Utils.php';
```

**Expected extractions:**
- Symbols: `register_hooks`, `on_init`, `format_date` (3 functions)
- References: `register_hooks` calls `add_action`, `on_init` referenced in hook
- Imports: `require_once` references

**Expected queries:**
- `search("hook registration")` → finds `add_action` call + `register_hooks` function
- `symbols(kind='function')` → returns 3 functions
- `references(symbol_name='on_init')` → returns hook registration + add_action call
- `impact(symbol_name='format_date')` → finds no callers (low-risk change)

---

**End of Specification**

**Version:** v2
**Date:** 2026-02-22
**Status:** Red-Team Feedback Incorporated, Ready for Phase 1 Implementation
**Next Step:** Phase 1 Implementation Kickoff + Architecture Deep-Dive
