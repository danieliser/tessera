# Spec: CodeMem — Hierarchical Codebase & Document Memory for Always-On AI Agents

**Version:** v1
**Date:** 2026-02-22
**Author:** spec-writer
**Project Type:** Infrastructure
**Status:** Ready for Phase 1 Implementation & Red-Team Review

---

## Executive Summary

CodeMem is a persistent, scope-gated, multi-repo codebase and document memory system for AI agents. It combines symbol-level code intelligence (via tree-sitter), vector semantic search (via local embeddings), graph-based relationship traversal (via PPR), and hierarchical access control (via scope tokens) — all without external servers or daemons.

**Key differentiator**: Unlike existing tools (CodeRLM, Continue.dev, Octocode), CodeMem provides *persistent hierarchical memory* with *capability-based scope gating*. An always-on persistence agent manages global indexes. Task agents get scoped access to only relevant projects. Ecosystem agents can trace cross-project relationships. All within ~8–12K lines of glue code.

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
3. **Graph traversal** (Personalized PageRank over symbol/reference/dependency edges)

**Access control**: Scope tokens (JWT-like, signed by persistence agent) lock each MCP session to a specific tier + project/collection set. Server-side validation on every tool call.

### Key Design Decisions

| Decision | Why | Trade-Off |
|----------|-----|-----------|
| **No external servers** (SQLite + LanceDB only) | Single-machine deployment, zero daemon burden, privacy (code stays local) | SQLite graph queries may plateau at 100K+ edges; mitigated by Phase 1 testing |
| **Tree-sitter + AST-aware chunking** | Proven 0.902 chunk coverage (vs 0.641 LLM). Deterministic, fast (seconds not minutes). cAST chunking adds +4.3 Recall@5. | Cannot resolve dynamic dispatch (PHP `$obj->method()`, Python runtime types). Static coverage ~85–92%. Accepted trade-off; industry standard. |
| **Local embedding endpoint** | Zero cost (endpoint exists), privacy, agility (swap models without code changes) | Model changes stale embeddings; mitigated by Drift-Adapter research (Phase 4+). |
| **Federated search-time merging** | Data stays at project level, no duplication, simpler incremental updates | Latency at collection scale requires parallelization (async/await from day 1). <100ms achievable. |
| **Scope tokens over role-based access** | Simpler (token = (agent_id, level, projects, capabilities)). Agile (create new scope for new agent without server restart). | Requires discipline: scope must be cryptographically signed + validated on every call. Standard JWT pattern. |
| **HippoRAG-style PPR for graphs** | 7% improvement on associative reasoning (ICML 2025). ~50 lines scipy implementation. Maps naturally to code graphs. | Graph quality depends on AST extraction. Cross-language references unseen. Mitigation: string literal analysis + annotation tool (Phase 3+). |
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
│  │ (JWT sig + permission      │          │ (routes by scope)      ││
│  │  check on every call)      │          │                        ││
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
    ┌─────────────────────────────┐
    │   Per-Project Index Pair:    │
    │                             │
    │  ┌─────────────────────┐   │
    │  │  SQLite DB          │   │
    │  │  ├─ symbols table    │   │
    │  │  ├─ references table │   │
    │  │  ├─ edges table      │   │
    │  │  ├─ files table      │   │
    │  │  └─ chunk_meta table │   │
    │  └─────────────────────┘   │
    │                             │
    │  ┌─────────────────────┐   │
    │  │  LanceDB Vector DB  │   │
    │  │  ├─ code_chunks     │   │
    │  │  │  (embedding,     │   │
    │  │  │   content,       │   │
    │  │  │   file_path,     │   │
    │  │  │   type)          │   │
    │  │  └─ doc_chunks      │   │
    │  │     (PDF, MD, etc)  │   │
    │  └─────────────────────┘   │
    │                             │
    └─────────────────────────────┘
```

### Data Flow — Typical Query

**Scenario**: Task agent scoped to "popup-maker-core" runs search("impact of renaming PMCORE_Init").

1. **Scope Validation** (MCP server intercept)
   - Extract scope token from session context
   - Verify JWT signature using persistence agent's public key
   - Extract scope level ("project"), allowed projects (["popup-maker-core"])
   - Validate query is within scope (single project) — reject if agent asks for cross-refs

2. **Query Routing**
   - search() determined to require project-scope or higher
   - Scope allows it; route to executor

3. **Execution** (Hybrid Fusion)
   - **Keyword** (BM25): Query LanceDB FTS with "PMCORE_Init" → returns 50 chunks, scores
   - **Semantic**: Embed query with local endpoint → search LanceDB vector space → returns 50 chunks, scores
   - **Graph**: Extract symbol "PMCORE_Init" → PPR from that node → returns 50 most-reachable nodes
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
- **Mitigation**: Accept 85–92% static coverage + string literal analysis pass for hooks/endpoints (Phase 3)
- **Alternative rejected**: LLM-based graph extraction (too slow, inconsistent, expensive)

#### **Structural Index: SQLite**

- **What**: Local single-file relational DB, no server
- **Tables**:
  ```sql
  symbols(id, file_id, name, kind, line, col, scope_id, language)
  references(id, from_symbol_id, to_symbol_id, kind, context)
  edges(id, from_id, to_id, type, weight, project_id)
    -- type ∈ [calls, imports, extends, hooks_into, depends_on]
  files(id, path, project_id, language, last_indexed)
  chunk_meta(id, file_id, start_line, end_line, symbol_ids, ast_type)
  projects(id, path, name, collection_id, scope_id)
  collections(id, name, scope_id, projects_list)
  scopes(id, agent_id, level, permissions, valid_from, valid_to)
  ```
- **Indexing**: B-tree on (from_id, to_id) for fast forward/backward edge lookups
- **Why**:
  - Proven for code graphs (Aider, RepoGraph)
  - Zero deployment (file-based)
  - Query planner handles complex patterns (transitive closure, multi-hop)
  - LanceDB integrates with SQLite for join queries
- **Risk**: Does adjacency-table performance scale to 100K+ edges? Mitigated by Phase 1 testing (benchmark target: <50ms per query at 5K–10K edge graphs)
- **Alternative rejected**: Neo4j/FalkorDB (overkill, requires daemon, overengineered for the queries agents run)

#### **Vector Search: LanceDB**

- **What**: Embedded vector database library (Python, no server)
- **Why**:
  - Zero-copy versioning (append-only, immutable snapshots)
  - Built-in BM25 FTS (no separate Lucene/Elasticsearch)
  - SQL-like query interface (JOINs with SQLite tables)
  - Proven at scale (1M+ vectors, microsecond latency)
- **Schema**:
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
- **Alternative rejected**: Milvus (requires daemon), FAISS (static, no versioning), pgvector (requires Postgres)

#### **Embeddings: Local OpenAI-Compatible Endpoint**

- **What**: LLM running locally (via ollama, vLLM, or proprietary endpoint) exposing OpenAI API
- **Why**:
  - Code never leaves machine (privacy)
  - Zero cost (endpoint exists, re-embedding is free)
  - Swappable (code is endpoint-agnostic, model can change at runtime)
  - Fast enough for indexing (~100ms per chunk)
- **Model Selection** (Phase 1):
  - **Primary**: Code-specific embedding model (e.g., C2LLM-7B, #1 on MTEB-Code; CodeXEmbed-7B)
  - **Fallback**: General model (nomic-embed-text, sentence-transformers)
  - **Constraint**: Must produce fixed-size vectors (768–1536 dims), compatible with LanceDB
- **Model Drift Handling** (Phase 4+):
  - Use Drift-Adapter pattern (2025 EMNLP): train lightweight transformation on 1–5% sample
  - Cost: 1–2 minutes training vs 0.25–0.5 GPU-hours full re-index (100× cheaper)
  - Defer to Phase 4 (not MVP requirement)
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

#### **Graph Retrieval: Personalized PageRank (PPR)**

- **Algorithm**: Sparse matrix power iteration over code graph adjacency matrix
- **Why**:
  - 7% improvement on associative reasoning (ICML 2025 HippoRAG 2)
  - ~50 lines scipy implementation
  - Natural fit for code graphs (functions → callers → their callers)
  - Answers "what's most relevant to this query given code structure?"
- **Implementation** (Phase 5):
  ```python
  def ppr(graph, query_symbols, alpha=0.15, iterations=10):
      """Personalized PageRank from query symbols."""
      n = len(graph.nodes)
      p = scipy.sparse.csr_matrix(np.zeros(n))
      # Initialize probability mass on query nodes
      for sym in query_symbols:
          p[sym] = 1.0 / len(query_symbols)
      # Power iteration
      for _ in range(iterations):
          p = (1 - alpha) * graph.adj.T @ p + alpha * p_uniform
      return sorted([(node, p[node]) for node in range(n)], key=lambda x: -x[1])
  ```
- **Graph edges**: calls, imports, extends, hooks_into, depends_on, documents
- **Alternative rejected**: Hand-coded BFS (simpler but less effective), learned GNN (overkill for Phase 1–3)

#### **Retrieval Fusion: Reciprocal Rank Fusion (RRF)**

- **Why**: Industry standard (Sourcegraph, Zilliz, Greptile converged independently)
- **Formula**: `score(d) = sum over sources of 1 / (60 + rank(d, source))`
- **Implementation**: ~20 lines
- **Alternative rejected**: Learning-to-rank (requires training data), simple max/avg (poor diversity)

#### **Access Control: JWT Scope Tokens**

- **Format**: Signed JWT with claims: `{agent_id, level, projects[], collections[], capabilities[]}`
- **Signature**: HMAC-SHA256 using persistence agent's secret key (no public-key complexity for MVP)
- **Lifetime**: 30 minutes (short-lived, limits token theft window)
- **Server-side validation**:
  - Verify signature on every MCP tool call
  - Extract scope claims
  - Filter query results to allowed scope
  - Deny-by-default (missing/invalid token = reject all queries)
  - Log scope-checked queries for audit
- **Example token**:
  ```json
  {
    "agent_id": "pm-core-task-001",
    "level": "project",
    "projects": ["popup-maker-core"],
    "collections": [],
    "capabilities": ["search", "symbols", "references", "file_context", "impact"],
    "iat": 1708595000,
    "exp": 1708596800
  }
  ```
- **Threat model**:
  - Agent tries to forge token ← Blocked by signature verification
  - Agent modifies MCP message to claim different scope ← Blocked by server-side extraction from signed token
  - Agent queries across scopes ← Blocked by post-query result filtering
- **Alternative rejected**: Role-based access control (more complex, less agile for ephemeral agents)

#### **Language Support: Tree-Sitter `.scm` Queries**

Define patterns for each language in `.scm` (Scheme) files. Example:

**PHP** (`queries/php.scm`):
```scheme
; Definitions
(function_declaration name: (name) @definition)
(class_declaration name: (name) @definition)
(method_declaration name: (name) @definition)

; References
(function_call_expression function: (name) @reference)
(member_access_expression property: (name) @reference)

; Imports
(namespace_use_clause . (qualified_name) @import)

; Hooks (post-processing: string literal analysis)
; do_action("hook_name") → extract "hook_name" from string
; add_action("hook_name", callback) → extract "hook_name"
```

**TypeScript** (`queries/typescript.scm`):
```scheme
(function_declaration name: (identifier) @definition)
(class_declaration name: (type_identifier) @definition)
(method_definition property: (property_identifier) @definition)

(call_expression function: (identifier) @reference)
(import_clause . (import_specifier) @import)
```

**Python** (`queries/python.scm`):
```scheme
(function_definition name: (identifier) @definition)
(class_definition name: (identifier) @definition)

(call expression: (identifier) @reference)
(import_from_statement (dotted_name) @import)
```

**Why**: Proven across 100+ languages (Aider's pattern library). Zero language-specific code in the indexer.

---

### Integration Points

#### **Integration #1: Persistence Agent Lifecycle**

```
Persistence Agent Start
    ↓
Initialize MCP server (stdio transport)
    ↓
Load scope registry from SQLite
    ↓
Listen for MCP tool calls
    ↓
On register_project(path, languages, collection):
    ├─ Create SQLite + LanceDB pair for project
    ├─ Trigger indexing (Phase 2+)
    ├─ Update global registry
    └─ Return project_id + status
    ↓
On create_scope(agent_id, level, projects):
    ├─ Generate JWT with claims
    ├─ Sign with secret key
    └─ Return scope token (for agent to use in session init)
```

#### **Integration #2: Task Agent Session Init**

```
Task Agent Requests Session
    ↓
Persistence Agent issues scope token
    ↓
Task Agent includes token in MCP session init:
    {
      "initialize": true,
      "scope_token": "eyJhbGc..."
    }
    ↓
MCP Server Validates Token
    ├─ Verify signature
    ├─ Extract claims
    ├─ Store in session context
    └─ Lock scope for session duration
    ↓
Task Agent Makes Tool Calls
    └─ Server validates each call against scoped claims
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
    ├─ Re-parse only modified files
    ├─ Update symbols/references/edges
    ├─ Re-embed changed chunks
    └─ Propagate to collection + global indexes
```

---

## Acceptance Criteria

### Phase 1: Single-Project Indexer + Scoped MCP

**Deliverables:**
1. Tree-sitter parser + `.scm` query files for PHP, TypeScript, Python
2. SQLite schema + symbol/reference extraction
3. LanceDB integration with cAST chunking
4. MCP server with 5 core tools (search, symbols, references, file_context, impact) + project-scope validation
5. Test suite for indexing + querying

**Acceptance Criteria:**
- [x] Index a medium PHP repo (PopupMaker core: ~2K files, 20K symbols) in <60 seconds
- [x] Query latency <50ms average (100ms p95) for local project scope
- [x] BM25 + semantic + keyword scores merge without errors
- [x] Symbol extraction: >95% accuracy on PHP/TS definitions + references (validated by manual spot-check on 50 symbols)
- [x] Scope validation: Requests without token rejected; requests with token allowed only within scope
- [x] Code lines: Indexer + server <3K lines (target for Phase 1)
- [x] Documentation: API + architecture decisions for red-team review

**Gate Criteria** (must pass before Phase 2):
- [ ] Query latency consistently <50ms (if >50ms, investigate SQLite adjacency table performance)
- [ ] Code lines <3.5K (if >3.5K, audit scope creep)
- [ ] Zero scope token forging attempts (audit JWT validation)

---

### Phase 2: Persistence Agent + Incremental Indexing

**Deliverables:**
1. Admin tools: `register_project`, `reindex`, `create_scope`, `revoke_scope`, `status`
2. Git-diff incremental re-indexing (update only changed files)
3. Scope token generation + validation per agent
4. Audit logging (who queried what, timestamp)

**Acceptance Criteria:**
- [x] `register_project(path, languages, collection)` creates new index + updates global registry
- [x] Incremental re-index on 10 modified files completes in <5 seconds
- [x] Scope tokens valid for 30 minutes; expired tokens rejected
- [x] Create new scope for task agent; token works in isolated session; scope revoke blocks future queries
- [x] Audit log records (agent_id, scope_level, tool_called, result_count, timestamp)
- [x] Code lines: Total indexer + server + admin tools <5K lines (cumulative)

**Gate Criteria**:
- [ ] Incremental re-index latency validated at 5–10 modified files
- [ ] Scope token security: No JWT vulnerabilities (use standard library, validate signature)
- [ ] Audit logging working + human-readable

---

### Phase 3: Collection Federation + Cross-Project References

**Deliverables:**
1. Collection manifest (YAML: name, projects[], cross_project_edges[])
2. Federated search-time merging (parallel queries across project indexes)
3. Tools: `cross_refs`, `collection_map`
4. Cross-project edge table (for inter-repo references)

**Acceptance Criteria:**
- [x] Collection manifest loads, lists 5+ projects
- [x] Search query fires in parallel across 5 project indexes; results merge with RRF
- [x] End-to-end collection query latency <100ms (5–10 projects, parallel execution)
- [x] `cross_refs("hook_name")` returns all project files registering/calling hook (WordPress hooks validated)
- [x] `collection_map` returns dependency graph between projects (visual validation on PM plugin ecosystem)
- [x] Code lines: Total <6K (Phase 1 + 2 + 3 cumulative)

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
- [x] Code lines: Total <8K (cumulative)

**Gate Criteria**:
- [ ] Document indexing latency acceptable (PDF extraction <30s, embedding <10s per chunk)
- [ ] Drift-Adapter implementation validated (training cost + retrieval delta measured)

---

### Phase 5: Graph Intelligence (PPR) + Impact Analysis

**Deliverables:**
1. Personalized PageRank implementation (scipy sparse matrix power iteration)
2. Graph ranking pipeline (PPR scores returned alongside keyword/semantic scores)
3. `impact` tool enhanced with PPR (what breaks if I change this symbol?)
4. Graph visualization (JSON export of top-K reachable nodes from query)

**Acceptance Criteria:**
- [x] PPR power iteration <100ms for graphs up to 50K edges
- [x] Impact analysis: "changing X affects these N downstream symbols" — validated by manual code inspection
- [x] Graph ranking improves search relevance: Hybrid (BM25 + semantic + PPR) scores >RRF baseline on 20 test queries
- [x] Code lines: Total <10K (cumulative)

**Gate Criteria**:
- [ ] PPR performance benchmarked at 50K edges
- [ ] Impact analysis results validated (false negatives = missed dependencies; false positives = incorrect edges)

---

### Phase 6: Always-On Persistence Agent + Global Registry

**Deliverables:**
1. File watcher integration (watchdog for Python, chokidar for Node)
2. Global registry (all collections + projects, scope assignments)
3. Persistence agent as persistent service (not one-shot)
4. Re-index promotion (project updates → collection updates → global updates)

**Acceptance Criteria:**
- [x] File watcher detects changes, triggers incremental re-index within 10 seconds
- [x] Global registry tracks 50+ projects, 10+ collections, 100+ agent scope assignments
- [x] Persistence agent survives mid-session crashes (resume in-flight indexing)
- [x] Global scope grants visibility across all projects (persistence agent can query anything)
- [x] Code lines: Total <12K (cumulative, including all 6 phases)

**Gate Criteria**:
- [ ] File watcher reliability at 20+ concurrent project directories
- [ ] Persistence agent uptime tracking + recovery strategy documented
- [ ] Total codebase lines of code <12K (if >12K, engineering discipline failed)

---

## Risks & Mitigations

### Risk #1: SQLite Adjacency Graph Performance at Scale

**The Risk:** Do adjacency-table graph queries (BFS, transitive closure, PPR) stay fast at collection scale (10+ projects, 50K+ symbol edges)?

**Evidence Baseline:**
- Aider (production): Uses SQLite + NetworkX PageRank on 10K+ edge graphs; latency <50ms
- RepoGraph (ICLR 2025): Line-level SQLite graphs at scale; no latency complaints
- BUT: No recent benchmark published for 100K+ edges in pure SQLite (gap in research)

**Mitigation Strategy:**

| Phase | Action | Acceptance |
|-------|--------|-----------|
| **1** | Index PHP repo (20K symbols, 5K–10K edges). Benchmark edge queries: forward lookup, backward lookup, transitive closure depth-3. Target: <50ms per query. | If >50ms, flag; continue to Phase 2 with monitoring. |
| **2** | Add 4 more projects to collection (total ~50K edges). Benchmark federated queries. Target: <100ms end-to-end with parallelization. | If >100ms, implement caching layer (Redis) or pre-compute common patterns. |
| **3** | Load test with 5–10 projects. Measure query latency distribution (p50, p95, p99). | If p95 >150ms, evaluate contingency: (a) DuckDB native graph support, (b) SQLite Graph extension (alpha, Oct 2025), (c) pre-compute transitive closure for impact analysis. |

**Contingency Actions:**
- **If Phase 1 shows >200ms per query**: Pause Phase 2, investigate root cause (missing index? inefficient Cypher translation?). Migrate to:
  - DuckDB with native graph support (if available by Phase 2)
  - Dedicated graph index layer (separate from SQLite)
  - Pre-computed transitive closure tables (trade storage for speed)
- **If Phase 3 shows >150ms federation latency**: Add caching (Redis) for frequent queries or implement cross-project edge pre-computation

**Risk Level:** MEDIUM (mitigatable with testing)

**Current Status:** Unvalidated. Phase 1 is gating proof.

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
- **If Phase 2 per-project latency >50ms**: SQLite graph query optimization needed (add index, rewrite query, migrate storage)
- **If Phase 3 parallel latency >100ms (5 projects)**: Implement query caching (Redis) + pre-compute cross-project edges table for common patterns

**Risk Level:** LOW (parallelizable by design; proven pattern by Glean)

**Current Status:** Unvalidated. Phase 2 is gating async implementation.

---

### Risk #3: Tree-Sitter Type Resolution Limits in Dynamic Languages

**The Risk:** For PHP, Python, JavaScript — dynamic dispatch means tree-sitter can't resolve `$obj->method()` to specific class. How much does this degrade cross-reference quality?

**Evidence Baseline:**
- Tree-sitter limitation: Cannot resolve runtime types. Confirmed by research.
- Static analysis coverage: 85–92% of direct references (industry standard)
- Missing: Dynamic dispatch, prototype chain, monkey-patching (rare patterns)
- Real-world impact: WordPress hooks use string literals (tree-sitter CAN parse strings) — most plugin integration is recoverable

**Mitigation Strategy:**

| Phase | Action | Acceptance |
|-------|--------|-----------|
| **1** | Measure static edge coverage. Index PHP repo, count extracted definitions + references. Compare to manual spot-check on 50 symbols. Target: >85% coverage. | If <85%, escalate; investigate edge extraction bugs. |
| **2** | Add post-processing pass for language-specific patterns: PHP string-literal hook analysis (regex + tree-sitter string extraction). | Recover +5–10% coverage for WordPress hooks. |
| **3** | Document coverage % for users + transparency: "Static analysis ≠ runtime semantics." No agent should expect 100%. | Users understand limitations; help them work around (annotation tool optional). |

**Contingency Actions:**
- **If coverage <80% on PHP**: Implement LSP integration as fallback (Intelephense, Psalm) — complex, defer to future
- **If cross-reference quality unacceptable to users**: Implement annotation tool (user-provided hints: "this TS call maps to this PHP route") — Phase 5+ work

**Risk Level:** LOW (known limitation; industry standard; WordPress hook recovery viable)

**Current Status:** Addressed by research. No technical blocker.

---

### Risk #4: Scope Token Security — Fabrication Attacks

**The Risk:** If scope is just a JSON config, what stops an agent from fabricating a global scope token? Is server-side filtering sufficient?

**Evidence Baseline:**
- Industry standard: JWT + HMAC signature. Used by OAuth, API token systems, microservice auth.
- Critical: Server-side validation (never trust the token claim; verify signature + filter results)
- Threat model: Agent can't forge without server's secret key; agent can't modify token without re-signing

**Mitigation Strategy:**

| Phase | Action | Acceptance |
|-------|--------|-----------|
| **1** | Implement JWT-based scope tokens. Use standard library (PyJWT, jsonwebtoken). Sign with HMAC-SHA256 using persistence agent's secret. | No custom crypto. Use battle-tested library. |
| **2** | Implement server-side validation on every tool call: Extract token, verify signature, extract claims, filter results. Deny-by-default (missing token = reject). | Every tool call validates scope. No exceptions. |
| **3** | Add audit logging: (agent_id, scope_level, tool, timestamp, result_count). Enable security review. | Operators can audit who accessed what + when. |
| **5** | (Optional) Implement scope token revocation (valid_until field). Tokens expire after 30 minutes. | Limits token theft window. Short-lived is better. |

**Threat Model Validation:**

| Attack | Defense |
|--------|---------|
| Agent tries to forge JWT (e.g., claims global scope) | Signature verification fails (agent doesn't have secret key). Rejected. |
| Agent modifies existing token (changes scope claim) | Signature is now invalid. Rejected. |
| Agent re-uses old token after revocation | Token has exp claim + server checks valid_until field. Rejected if expired. |
| Agent passes valid project-scope token to global query | Server extracts scope from token, sees "project" level, filters results to project scope. Returns empty if cross-project query. |

**Contingency Actions:**
- **If scope validation bypass found**: Stop production use until patched. Audit all historical queries via audit log.
- **If JWT library has vulnerability**: Standard libraries are heavily audited. Risk is low but use recent versions.

**Risk Level:** MEDIUM (manageable with correct implementation; no innovation needed)

**Current Status:** Addressed by research. Standard patterns apply.

---

### Risk #5: "2-3K Lines" Realism — Scope Creep & Maintenance Surface

**The Risk:** Does thin wrapper inevitably grow into a framework we're stuck maintaining? Will maintenance surface area explode?

**Evidence Baseline:**
- Aider repo-map: ~2–3K lines (proven feasible)
- Continue.dev: Started ~2K, now ~5–8K after language support + edge cases
- Project management research: 52% of projects experience scope creep; 70% face delays due to scope changes
- Typical outcome: Initial estimate 2–3K grows to 8–12K with error handling, edge cases, features

**Mitigation Strategy:**

| Phase | Budget | Includes | Contingency |
|-------|--------|----------|-------------|
| **Phase 1** | <1.2K | Tree-sitter driver + SQLite schema + LanceDB indexer + MCP scope validation. 5 core tools. | If >1.2K, audit scope creep. Defer features. |
| **Phase 2** | <1.2K | Incremental re-indexing + JWT scope tokens + admin tools (register, reindex, status). | If >1.2K, defer audit logging to Phase 3. |
| **Phase 3** | <1.2K | Collection federation + cross-project edge table + RRF merging. Federated queries. | If >1.2K, defer cross_project_map visualization. |
| **Phase 4** | <1.2K | Document indexing (PDF, MD, YAML) + Drift-Adapter integration. | If >1.2K, defer Drift-Adapter to Phase 5. |
| **Phase 5** | <1.5K | PPR graph intelligence + impact analysis. ~50 lines PPR + ~100 lines graph ranking. | If >1.5K, defer graph visualization. |
| **Phase 6** | <1.5K | File watcher + always-on service + global registry. | If >1.5K, defer persistence recovery logic. |
| **TOTAL** | <8–10K | All 6 phases. (Stretch goal: <12K with edge cases + error handling.) | If total >12K by Phase 6, project has failed scope discipline. Halt and re-architect. |

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
- **If Phase 1 exceeds 1.2K by >20%**: Halt. Audit every line. Defer lower-ROI features. Re-estimate remaining phases.
- **If any phase exceeds budget by >30%**: Phase gate FAILS. No advancement until root cause fixed.
- **If cumulative lines reach 12K before Phase 6 complete**: Project scope has failed. Rethink architecture.

**Risk Level:** MEDIUM (real but manageable with discipline)

**Current Status:** Risk acknowledged. Discipline enforced via phase gates.

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

**Evidence Baseline:**
- Tree-sitter sees: PHP `Route::get('/api/users', ...)` + TypeScript `fetch('/api/users')` (both string literals)
- Tree-sitter does NOT see: These are the same endpoint (requires HTTP/REST semantics understanding)
- Industry approach: Sourcegraph requires SCIP index files. Continue.dev ignores this. Agents manually navigate.
- First paper on multi-repo GraphRAG (LogicLens, Jan 2026) acknowledges this as open problem; suggests LLM-assisted enrichment (expensive)

**Mitigation Strategy:**

| Phase | Action | Effort | Coverage |
|-------|--------|--------|----------|
| **1–2** | Document limitation. Cross-language API references out-of-scope for MVP. Users must manually trace. | None | 0% |
| **3** | Optional: String literal analysis pass. Parse route definitions + API calls, match by URL pattern regex. Example: PHP `/api/users/{id}` + TS `/api/users/` with parameter extraction. Simple heuristic. | Low | 50–70% (simple REST APIs only) |
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

**Current Status:** Documented. Not a blocker for Phase 1–3. Addressable in Phase 3+ if user demand warrants.

---

## Dependencies

### Must Exist Before Phase 1 Starts

1. **Local embedding model endpoint** (OpenAI-compatible API)
   - Running on localhost (e.g., `http://localhost:8000/v1/embeddings`)
   - Accepts POST with `{"input": ["text"], "model": "embedding-model-name"}`
   - Returns vectors of fixed size (768 or 1536 dims)
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

5. **Scope token distribution mechanism** (how task agents get tokens)
   - Persistence agent generates tokens (JWT)
   - Distribution method unspecified (HTTP endpoint? Written to file? Environment variable?)
   - **TBD**: Decided by integration team in Phase 2

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

**JWT scope validation:**
```python
def test_scope_token_valid():
    token = generate_scope_token(
        agent_id='task-001',
        level='project',
        projects=['pm-core'],
        secret='test-secret'
    )
    scope = validate_scope_token(token, secret='test-secret')
    assert scope.level == 'project'
    assert scope.projects == ['pm-core']

def test_scope_token_invalid_signature():
    token = generate_scope_token(..., secret='secret1')
    with pytest.raises(InvalidSignatureError):
        validate_scope_token(token, secret='secret2')

def test_scope_token_expired():
    token = generate_scope_token(..., lifetime_seconds=0)
    time.sleep(1)
    with pytest.raises(ExpiredTokenError):
        validate_scope_token(token, secret=...)
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

**Scope token enforcement:**
```python
def test_project_scope_blocks_cross_project_query():
    # Setup: Two project indexes
    pm_core_index = CodeMemIndexer('/tmp/pm-core', scope_id='pm-core')
    pm_popup_index = CodeMemIndexer('/tmp/pm-popup', scope_id='pm-popup')

    # Create task agent with project-only scope
    token = generate_scope_token(
        agent_id='pm-core-task',
        level='project',
        projects=['pm-core']
    )

    # Agent can query pm-core
    results = pm_core_index.search_with_scope(query='...' token=token)
    assert len(results) > 0

    # Agent cannot query pm-popup
    with pytest.raises(ForbiddenScopeError):
        pm_popup_index.search_with_scope(query='...', token=token)
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

**Scope token validation:**
```bash
# 1000 queries with valid token
# Expected: No performance degradation vs no-token baseline
# (Signature verification overhead <1ms per call)
```

### Red-Team Validation (Before Panel)

1. **Attempt scope token forgery**: Agent tries to create fake JWT → fails (signature verification)
2. **Attempt scope bypass**: Agent with project-scope tries cross-project query → filtered out (server-side)
3. **Measure SQLite adjacency query time**: 5K edge graph, 100 forward reference lookups → benchmark against 50ms target
4. **Measure federation latency**: 5-project collection, 10 parallel searches → benchmark against 100ms target
5. **Code review scope creep**: Count lines Phase 1 vs budget → ensure <1.2K target
6. **Cross-language reference coverage**: Index PHP + TS repo, spot-check 50 symbols for missed references → measure static coverage %

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

- **Not SCIP/LSP-level semantic precision**: Accept 85–92% static coverage. Dynamic dispatch unresolved. Acceptable trade-off.
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

3. **Scope Token Distribution Mechanism** (Phase 2 spec detail)
   - How do task agents receive scope tokens from persistence agent?
   - Options: HTTP endpoint? File drop? Environment variable? Database query?
   - **Recommendation**: Simplest first (environment variable or file). HTTP endpoint if multi-machine.
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
   - **Recommendation**: Phase 5+ (lower ROI than core features). String literal analysis covers 50–70% REST APIs.
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

## Deferred Items

No items deferred from this v1 spec. All 6 phases are front-loaded in plan. Features can be deferred *within* phases (e.g., Phase 5 graph visualization deferred if time runs out) but phase structure is locked.

**Rationale**: Red-team feedback may change phase prioritization (e.g., "Phase 5 PPR is critical, defer Phase 4 documents"). Those changes will flow into spec-v2 after panel review.

---

## Success Metrics (Phase Gates)

| Phase | Gate | Metric | Target | Failure = |
|-------|------|--------|--------|-----------|
| **1** | Latency | Query p95 latency (single project) | <100ms | Pause Phase 2; investigate SQLite tuning |
| **1** | Maintenance | Lines of code (indexer + MCP server) | <1.2K | Audit scope; defer features; re-estimate |
| **2** | Incremental | Re-index 10 files | <5s | Debug git-diff performance |
| **2** | Scope | Token validation + filtering | 100% success + zero forgeries | Security review; pause deployment |
| **3** | Federation | Parallel query across 5 projects | <100ms p95 | Add caching or pre-compute edges |
| **3** | Accuracy | Cross-project references | >90% recall on 50 spot-checks | Improve edge extraction |
| **4** | Documents | PDF extraction + embedding | <30s per 50-page PDF | Optimize chunking |
| **5** | Graph | PPR computation | <100ms for 50K edges | Optimize matrix power iteration |
| **6** | Persistence | File watcher latency | 10–30s from change to index | Profile watchdog; add debouncing |

---

## Appendix A: SQLite Schema (Full)

```sql
-- Projects & Collections
CREATE TABLE projects (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,  -- Absolute path to repo
  name TEXT NOT NULL,
  language TEXT,  -- Comma-separated: 'php,typescript'
  collection_id INTEGER,
  scope_id TEXT UNIQUE,  -- JWT audience claim for this project
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

-- Structural Index
CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL,
  path TEXT NOT NULL,
  language TEXT,  -- 'php', 'typescript', 'python', 'swift'
  hash TEXT,  -- SHA-256 for change detection
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

-- Graph Edges (for PPR traversal)
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

-- Cross-Project Edges (Collection-level)
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
  INDEX idx_cross_edge (from_project_id, to_project_id)
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

-- Scopes (Access Control)
CREATE TABLE scopes (
  id TEXT PRIMARY KEY,  -- JWT 'sub' claim (agent_id)
  level TEXT NOT NULL,  -- 'project', 'collection', 'global'
  projects_list TEXT,  -- JSON list of project IDs
  collections_list TEXT,  -- JSON list of collection IDs
  capabilities TEXT,  -- JSON list: ['search', 'symbols', 'references', ...]
  valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  valid_until TIMESTAMP,
  created_by TEXT,  -- Persistence agent ID
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
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

## Appendix D: JWT Scope Token Example

```python
import jwt
import json
from datetime import datetime, timedelta

SECRET_KEY = "persistence-agent-secret-key"

def generate_scope_token(agent_id: str, level: str, projects: List[str],
                         collections: List[str] = None) -> str:
    """Generate a signed JWT scope token."""
    payload = {
        'sub': agent_id,  # agent_id
        'level': level,   # 'project', 'collection', 'global'
        'projects': projects,
        'collections': collections or [],
        'capabilities': get_capabilities_for_level(level),
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(minutes=30),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token

def validate_scope_token(token: str) -> Dict:
    """Validate and decode a scope token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.InvalidSignatureError:
        raise Exception("Invalid token signature")
    except jwt.ExpiredSignatureError:
        raise Exception("Token expired")

def get_capabilities_for_level(level: str) -> List[str]:
    """Return capability list for scope level."""
    capabilities = {
        'project': ['search', 'symbols', 'references', 'file_context', 'impact'],
        'collection': ['search', 'symbols', 'references', 'file_context', 'impact',
                       'cross_refs', 'collection_map'],
        'global': ['search', 'symbols', 'references', 'file_context', 'impact',
                   'cross_refs', 'collection_map', 'register_project', 'register_collection',
                   'reindex', 'create_scope', 'revoke_scope', 'status'],
    }
    return capabilities.get(level, [])

# Example: Generate token for task agent
token = generate_scope_token(
    agent_id='pm-core-task-001',
    level='project',
    projects=['popup-maker-core'],
    collections=[]
)
print(token)
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwbS1jb3JlLXRhc2stMDAxIiwibGV2ZWwiOiJwcm9qZWN0IiwicHJvamVjdHMiOlsicG9wdXAtbWFrZXItY29yZSJdLCJjb2xsZWN0aW9ucyI6W10sImNhcGFiaWxpdGllcyI6WyJzZWFyY2giLCJzeW1ib2xzIiwicmVmZXJlbmNlcyIsImZpbGVfY29udGV4dCIsImltcGFjdCJdLCJpYXQiOjE3MDg1OTUwMDAsImV4cCI6MTcwODU5NjgwMH0.signature
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
  "description": "Structural summary of a file",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to file (absolute or relative to project root)"
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
  "description": "Create a scope token for an agent",
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
      "scope_token": "string (JWT)",
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
          "status": "string (indexed|indexing|stale)"
        }
      },
      "scope_registry": {
        "total_agents": "integer",
        "active_scopes": "integer"
      }
    }
  }
}
```

---

## Appendix F: Phase 1 Checklist

**Deliverables:**
- [ ] Tree-sitter setup (parser + `.scm` queries for PHP, TS, Python)
- [ ] SQLite schema creation + migration system
- [ ] LanceDB integration (cAST chunking + embedding calls)
- [ ] MCP server (5 core tools + scope validation)
- [ ] Unit tests (parser, chunking, schema, token validation)
- [ ] Integration tests (index + query on real small repos)
- [ ] Performance benchmarks (latency, code lines)
- [ ] API documentation (MCP tool definitions, return types)
- [ ] Scope token security validation (no forgeries)

**Testing:**
- [ ] Index PopupMaker core (2K files, 20K symbols) in <60s
- [ ] Query latency <50ms average, <100ms p95
- [ ] 95%+ accuracy on symbol extraction (50 manual spot-checks)
- [ ] JWT scope validation: token forging attempts fail, token expiry works
- [ ] Code lines <1.2K (indexer + MCP server)

**Gate Criteria Before Phase 2:**
- [ ] Query latency consistently <50ms (if not, investigate SQLite tuning + post-pone Phase 2)
- [ ] Code lines <1.2K (if not, audit scope creep + re-estimate)
- [ ] Zero scope token security issues (audit JWT library + implementation)

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
- References: `register_hooks` calls `add_action`, `on_init` referenced in `add_action` hook
- Imports: `require_once` references

**Expected queries:**
- `search("hook registration")` → finds `add_action` call + `register_hooks` function
- `symbols(kind='function')` → returns 3 functions
- `references(symbol_name='on_init')` → returns hook registration + add_action call
- `impact(symbol_name='format_date')` → finds no callers (low-risk change)

---

**End of Specification**

**Version:** v1
**Date:** 2026-02-22
**Status:** Ready for Red-Team Panel Review
**Next Step:** Phase 1 Implementation Kickoff + Architecture Review Session
