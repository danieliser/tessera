# Tessera

Persistent codebase intelligence for autonomous AI agents. Tessera gives agents bottom-up file access and top-down code understanding — across every project they're authorized to touch, with security from the ground up.

## The Problem

Persistent AI agents — orchestrators like AutoJack, task agents like OpenClaw — need to understand codebases the way a senior developer does. Not just "find this string in a file," but "what calls this function, across which projects, and what breaks if I change it?"

Today's agents burn context window and wall-clock time on repeated `grep` / `find` / `cat` cycles. They lose track of project structure between conversations. They can't safely delegate to sub-agents without leaking access to projects those agents shouldn't see. And they can't search documentation, config files, or assets alongside code.

## What Tessera Does

Tessera indexes everything — code, documents, config files, text assets — into a structured, chunked, searchable database. It exposes that through 18 MCP tools that any agent can call. Responses come back in milliseconds, not seconds.

**For orchestrator agents:** Full system visibility. Register projects, group them into collections, search across all of them. Understand cross-project dependencies. Delegate scoped access to sub-agents via session tokens.

**For task agents:** Deep code intelligence within their authorized scope. Symbol lookup, reference tracing, impact analysis, document search — everything an IDE provides, but through tool calls.

**For security:** Deny-by-default scope gating. Sub-agents only see what the orchestrator explicitly grants. Credentials and secrets are blocked from indexing by un-negatable security patterns. No ambient access, no scope creep.

### Code Intelligence
- **Symbol search** — Functions, classes, methods, hooks by name or pattern
- **Reference tracing** — Call graphs, imports, inheritance chains
- **Impact analysis** — "What breaks if I change this?" — traced N levels deep
- **File context** — Complete structural overview of any file in one call
- **Cross-project references** — Track where project A's exports are used in project B

### Document & Text Search
- **Chunked indexing** — Files are split into focused, searchable chunks with metadata (by header, key path, or line group) — not stored as monolithic blobs
- **Code + docs unified** — Query across everything, or filter by source type
- **Structural formats** — PDF, Markdown (header hierarchy), YAML/JSON (key-path chunking)
- **Markup** — HTML/XML with tag stripping
- **Plaintext** — `.txt`, `.rst`, `.csv`, `.log`, `.ini`, `.cfg`, `.toml`, config files, dotfiles

### Multi-Project Federation
- **Project collections** — Group related projects (e.g., a plugin ecosystem) and query across them
- **Scope-gated access** — Session tokens control what each agent can see. Orchestrators create scoped tokens for sub-agents.
- **Search-time federation** — Data stays at project level, merged at query time. No duplication.

### Security
- **Deny-by-default** — No access without a valid session token
- **`.tesseraignore`** — Per-project ignore config with `.gitignore` syntax
- **Two-tier ignore system** — Security-critical patterns (`.env*`, `*.pem`, `*credentials*`) are locked and cannot be overridden by project config
- **`trusted` field** — Search results from code are marked trusted; document content is marked untrusted so agents can handle prompt injection risk

### Infrastructure
- **Fully embedded** — SQLite + FAISS. No Docker, no daemons, no external servers
- **Incremental indexing** — Git-aware, only re-indexes changed files
- **Schema migration** — Versioned database schema with automatic upgrades
- **Drift adapter** — Switch embedding models without re-indexing (Orthogonal Procrustes)

## Supported Languages

PHP, TypeScript, JavaScript, Python, Swift — via tree-sitter grammars.

## MCP Tools (18)

### Search & Navigation
| Tool | Purpose |
|------|---------|
| `search` | Hybrid keyword + semantic search across code and documents |
| `doc_search_tool` | Document-only search (filterable by format) |
| `symbols` | Look up functions, classes, methods by name/pattern/kind |
| `references` | Find all references to a symbol (calls, imports, extends) |
| `file_context` | Complete context for a file (symbols, refs, structure) |
| `impact` | Trace downstream impact of changing a symbol |
| `cross_refs` | Cross-project references to a symbol |
| `collection_map` | Overview of projects in a collection with stats |

### Administration
| Tool | Purpose |
|------|---------|
| `register_project` | Register a project for indexing |
| `reindex` | Trigger full or incremental re-index |
| `status` | Project indexing status and health |
| `drift_train` | Train embedding drift adapter for model migration |

### Access Control
| Tool | Purpose |
|------|---------|
| `create_scope_tool` | Create scoped session tokens for sub-agents |
| `revoke_scope_tool` | Revoke agent session tokens |
| `create_collection_tool` | Create a project collection |
| `add_to_collection_tool` | Add a project to a collection |
| `list_collections_tool` | List all collections |
| `delete_collection_tool` | Delete a collection |

## Quick Start

### Requirements
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install
```bash
git clone https://github.com/danieliser/tessera.git
cd tessera
uv sync
```

### Run as MCP Server

Add to your `.mcp.json`:
```json
{
  "mcpServers": {
    "tessera": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/tessera",
        "run", "python", "-m", "tessera", "serve"
      ]
    }
  }
}
```

Lock to a specific project (single-project mode):
```bash
uv run python -m tessera serve --project /path/to/your/project
```

### Embedding Setup (Optional)

Tessera works without embeddings (keyword search only via FTS5). For semantic search, point it at any local OpenAI-compatible embedding endpoint. The embedding dimension is auto-detected — no configuration needed.

Recommended: [LM Studio](https://lmstudio.ai) with `nomic-embed-text` or any embedding model serving on `/v1/embeddings`.

### Run Tests
```bash
uv run pytest tests/ -v
```

## Architecture

```
MCP Server (stdio)
├── Scope Validator (session-based, deny-by-default)
├── Query Router (project / collection / global)
│   ├── Search (FTS5 keyword + FAISS semantic + RRF merge)
│   ├── Symbols / References / Impact (SQLite graph)
│   └── Document Search (source_type filtering)
├── Per-Project Indexes
│   ├── SQLite (symbols, references, edges, files, chunk_meta)
│   └── FAISS (vector embeddings)
├── Global SQLite (~/.tessera/global.db)
│   ├── projects, collections, sessions
│   └── indexing_jobs
└── Indexer Pipeline
    ├── Tree-sitter parser (PHP, TS, JS, Python, Swift)
    ├── AST-aware code chunking
    ├── Document extraction (PDF, MD, YAML, JSON, HTML, XML, plaintext)
    └── Ignore filter (.tesseraignore, two-tier security)
```

### Design Principles

- **No external dependencies at runtime** — SQLite + FAISS, fully embedded
- **Tree-sitter for deterministic parsing** — no LLM-extracted graphs, no hallucinated edges
- **Chunked everything** — every file is split into focused, searchable units with structural metadata
- **Security-first scope model** — deny-by-default, session-scoped, un-negatable credential protection
- **Federation over duplication** — data stays at project level, merged at query time

## Project Status

**v0.2.0** — Core system operational. Code intelligence, document indexing, federation, and access control all working.

| Phase | Status | What |
|-------|--------|------|
| 1 | Done | Single-project indexer + scoped MCP server |
| 2 | Done | Incremental indexing + persistence |
| 3 | Done | Collection federation + cross-project refs |
| 4 | Done | Document indexing + drift adapter + ignore config + text formats |
| 4.5 | Planned | Media/binary file metadata catalog |
| 5 | Planned | PPR graph intelligence |
| 6 | Planned | Always-on file watcher |

## License

MIT
