# Tessera

Instant codebase search and navigation for AI agents. Tessera indexes your code, documents, and config files into a structured, searchable database — then exposes it all through MCP tools that any AI agent can call.

## Why Tessera?

AI agents working on codebases spend most of their time searching — finding where a function is defined, understanding what calls what, figuring out which files are related. Generic search tools return noisy results. Tessera gives agents the same level of code understanding that an IDE provides: symbol lookups, reference tracing, impact analysis, and semantic search — across multiple projects, with scope-gated access control.

**What it replaces:** Repeated `grep` / `find` / `cat` cycles, manual file discovery, losing track of project structure across conversations.

**What it enables:** An agent can ask "what calls this function?" and get an answer in milliseconds, across every project it has access to.

## Capabilities

### Code Intelligence
- **Symbol search** — Find functions, classes, methods, hooks by name or pattern across your codebase
- **Reference tracing** — "Who calls this?" / "What does this depend on?" with full call graph edges
- **Impact analysis** — Trace downstream effects of changing a symbol, N levels deep
- **File context** — Get all symbols, references, and structure for any file in one call
- **Cross-project references** — Track symbol usage across federated project collections

### Document & Text Search
- **Chunked indexing** — Documents are split into searchable chunks (by header, key path, or line groups), not stored as monolithic blobs
- **PDF, Markdown, YAML, JSON** — Structural chunking that preserves document hierarchy
- **HTML, XML** — Tag stripping with visible-text extraction
- **Plaintext formats** — `.txt`, `.rst`, `.csv`, `.log`, `.ini`, `.cfg`, `.toml`, config files, dotfiles
- **Unified search** — Query code and docs together, or filter by source type
- **`doc_search`** — Dedicated tool for document-only queries

### Multi-Project Federation
- **Project collections** — Group related projects (e.g., a plugin ecosystem) and search across them
- **Scope-gated access** — Session tokens control which projects/collections an agent can see
- **Cross-project symbol lookup** — Find where a function defined in project A is used in project B

### Infrastructure
- **Incremental indexing** — Only re-indexes changed files (git-aware)
- **Schema migration** — Versioned database schema with automatic upgrades
- **Embedding model migration** — Drift adapter (Orthogonal Procrustes) lets you switch embedding models without re-indexing everything
- **`.tesseraignore`** — Per-project ignore config with `.gitignore` syntax. Security-critical patterns (`.env*`, `*.pem`) are locked and can't be overridden.

## Supported Languages

PHP, TypeScript, JavaScript, Python, Swift — via tree-sitter grammars.

## MCP Tools (18)

| Tool | Purpose |
|------|---------|
| `search` | Hybrid keyword + semantic search across code and documents |
| `doc_search_tool` | Search non-code documents only (filterable by format) |
| `symbols` | Look up functions, classes, methods by name/pattern/kind |
| `references` | Find all references to a symbol (calls, imports, extends) |
| `file_context` | Get complete context for a file (symbols, refs, structure) |
| `impact` | Trace downstream impact of changing a symbol |
| `cross_refs` | Find cross-project references to a symbol |
| `collection_map` | Overview of projects in a collection with stats |
| `register_project` | Register a new project for indexing |
| `reindex` | Trigger full or incremental re-index |
| `status` | Project indexing status and health |
| `drift_train` | Train embedding drift adapter for model migration |
| `create_scope_tool` | Create scoped session tokens for agents |
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

Or lock to a specific project:
```bash
uv run python -m tessera serve --project /path/to/your/project
```

### Index a Project

Indexing happens automatically when you use `register_project` + `reindex` through the MCP tools. For CLI indexing (requires an embedding endpoint):

```bash
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:8000/v1
```

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
    └── Ignore filter (.tesseraignore)
```

### Key Design Decisions

- **No external servers** — fully embedded (SQLite + FAISS), no Docker, no daemons
- **Tree-sitter for parsing** — deterministic, fast, 100+ language support
- **Chunked indexing** — every file is split into focused, searchable chunks with metadata
- **Search-time federation** — data stays at project level, merged at query time
- **Scope gating** — agents only see what they're authorized to see

## Embedding Setup (Optional)

Tessera works without embeddings (keyword search only via FTS5). For semantic search, point it at any local OpenAI-compatible embedding endpoint:

- [Ollama](https://ollama.com) — `ollama serve` + any embedding model
- [LM Studio](https://lmstudio.ai) — local model server
- [vLLM](https://docs.vllm.ai) — production-grade serving

The embedding dimension is auto-detected from the model output. No configuration needed.

## Project Status

**v0.2.0** — Production-ready for single-user, local-machine use.

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
