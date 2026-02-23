# CodeMem — Project Context

## Overview
CodeMem is a hierarchical, scope-gated codebase indexing and persistent memory system for always-on AI agents. It's an MCP server that provides search, symbol navigation, reference tracing, and impact analysis across multi-language, multi-repo projects.

## Project Status
**Phase 1: Active Implementation**

Scaffolding complete. Architecture design documented in `docs/plans/architecture/`. Proceeding with modular implementation, validating CTO conditions via spike tests.

## Tech Stack
- **Language**: Python 3.11+
- **AST Parsing**: tree-sitter (universal, 100+ languages)
- **Structural Index**: SQLite (symbols, references, graph edges)
- **Semantic Index**: FAISS (vector similarity search via faiss-cpu)
- **Graph Retrieval**: SQLite adjacency tables for graph queries
- **Embeddings**: Local OpenAI-compatible model endpoint
- **Interface**: MCP server (stdio transport)

## Key Architectural Decisions
1. **No external servers/daemons** — fully embedded (SQLite + FAISS)
2. **Capability-based scope control** — project/collection/global tiers, locked per MCP session
3. **Search-time federation** — data stays at project level, merged at query time
4. **Thin glue layer** — ~2-3K lines orchestrating proven libraries, not a framework
5. **Tree-sitter deterministic graphs** — validated as beating LLM-extracted graphs

## Package Structure
```
src/codemem/
  __init__.py       - Package metadata and version
  __main__.py       - CLI entry point (index, serve commands)
  db.py             - SQLite schema and connection management
  parser.py         - Tree-sitter parser orchestration
  chunker.py        - AST-aware code chunking strategy
  embeddings.py     - OpenAI-compatible client + caching
  auth.py           - Session tokens and scope gating (C2)
  indexer.py        - Incremental re-indexing orchestration (C3)
  server.py         - MCP server and tool registration
  search.py         - FAISS vector search + keyword + RRF merging (C5)
```

## Running CodeMem

### Development Setup
```bash
uv sync                           # Install dependencies
uv run pytest tests/ -v           # Run test suite
```

### Index a Project
```bash
uv run python -m codemem index /path/to/project \
  --embedding-endpoint http://localhost:8000/v1
```

### Start MCP Server
```bash
uv run python -m codemem serve --project /path/to/project
```

## Code Style
- **Language**: Python 3.11+
- **Type Hints**: Required on all public API functions
- **Docstrings**: Public APIs only (module, class, method level)
- **Testing**: pytest + pytest-asyncio for async code
- **Async**: Use `asyncio.to_thread()` to wrap blocking I/O (FAISS, embeddings, etc.)

## CTO Conditions

All CTO conditions from architecture phase documented in `docs/conditions.md`.

## Languages Supported
PHP, TypeScript, Python, Swift — via tree-sitter grammars.

## Relationship to Other Projects
- Consumed by the persistence agent toolkit (`~/Toolkit/`)
- Independent, open-sourceable project
- Uses the strategic planner (`~/Toolkit/strategic-planner/`) for planning
