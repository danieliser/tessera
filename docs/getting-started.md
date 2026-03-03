# Tessera Getting Started

Tessera is a persistent codebase intelligence MCP server. Index your project once, then search, navigate, and analyze code through 18 MCP tools. Everything runs locally — SQLite + FAISS, no Docker, no external services.

This guide walks you through installation, indexing your first project, and running Tessera as an MCP server.

## Prerequisites

- **Python 3.11+** — Check with `python3 --version`
- **uv** — Package manager. Install from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **Git** — For cloning the repository

No other runtime dependencies. SQLite and FAISS are vendored.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/danieliser/tessera.git
cd tessera
uv sync
```

Verify the installation:

```bash
uv run python -m tessera --help
```

You should see:

```
usage: tessera [-h] {index,serve} ...

CodeMem: Hierarchical codebase indexing and memory system

positional arguments:
  {index,serve}  Command to run
    index        Index a project
    serve        Start the MCP server

optional arguments:
  -h, --help     show this help message and exit
```

## Index Your First Project

Indexing extracts code symbols, documents, configuration, and media files into a searchable SQLite database.

### Full Index (without embeddings)

```bash
uv run python -m tessera index /path/to/your/project
```

Example output:

```
Full index: /path/to/your/project
No embedding endpoint — indexing without embeddings.

Done in 12.3s
  Files: 247 indexed, 18 skipped, 0 failed
  Symbols: 1340 extracted
  Chunks: 3156 (0 embedded)
```

What Tessera indexed:
- **Code symbols** (functions, classes, methods, variables) via tree-sitter parsing
- **References** (where symbols are called, imported, or extended)
- **Documents** (Markdown, YAML, JSON, text files) chunked by structure
- **Configuration files** (`.env*`, `*.toml`, `*.ini`, dotfiles)
- **Media assets** (images, videos, audio, fonts) by metadata

Skipped files include node_modules, `.git`, vendored code, and patterns in `.tesseraignore`.

### Index with Semantic Search (optional)

Skip this if you only need keyword search (FTS5). For semantic search, you need a local embedding server running before indexing.

Set up an embedding endpoint (see [Embedding Server Setup](#embedding-server-setup-optional) below), then index with embeddings:

```bash
uv run python -m tessera index /path/to/your/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings \
  --embedding-model nomic-embed
```

Example output:

```
Full index: /path/to/your/project
Embedding endpoint: http://localhost:8800/v1/embeddings (model: nomic-embed)

Done in 45.2s
  Files: 247 indexed, 18 skipped, 0 failed
  Symbols: 1340 extracted
  Chunks: 3156 (3156 embedded)
```

After indexing completes, the database is stored in `.tessera/` at your project root:

```
/path/to/your/project/
├── .tessera/
│   ├── project.db       # SQLite index
│   ├── embeddings.faiss # Vector search (if embeddings used)
│   └── meta.json        # Parser version and metadata
```

This database persists. You can search it immediately or configure an MCP server to expose it to Claude Code.

### Incremental Indexing

After the first index, you can re-index only changed files:

```bash
uv run python -m tessera index /path/to/your/project \
  --incremental \
  --embedding-endpoint http://localhost:8800/v1/embeddings
```

Tessera uses git to detect changed files. Only modified code is re-parsed and re-embedded.

## Start the MCP Server

The MCP server exposes 18 tools to search, navigate, and analyze code. You can run it in two modes:

### Single-Project Mode (locked to one project)

```bash
uv run python -m tessera serve --project /path/to/your/project
```

This locks the server to a single project. All MCP tools query only that project's index.

### Multi-Project Mode (no lock)

```bash
uv run python -m tessera serve
```

The server can query multiple projects. You must register projects first (via `register_project` tool or from Claude Code).

### With Embedding Support

If you configured an embedding endpoint during indexing, you can enable semantic search on the server:

```bash
uv run python -m tessera serve --project /path/to/your/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings \
  --embedding-model nomic-embed
```

The server starts on stdio (suitable for MCP clients) and is ready for connections.

## Configure MCP in Claude Code

Claude Code and other MCP clients use `.mcp.json` to load MCP servers.

### Basic Configuration (single project, no embeddings)

Create or edit `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "tessera": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/tessera",
        "run", "python", "-m", "tessera", "serve",
        "--project", "/absolute/path/to/your/project"
      ]
    }
  }
}
```

Replace:
- `/absolute/path/to/tessera` — absolute path to the cloned tessera repo
- `/absolute/path/to/your/project` — absolute path to the project you indexed

### With Embeddings

If you indexed with embeddings, enable them in the server config:

```json
{
  "mcpServers": {
    "tessera": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/tessera",
        "run", "python", "-m", "tessera", "serve",
        "--project", "/absolute/path/to/your/project",
        "--embedding-endpoint", "http://localhost:8800/v1/embeddings",
        "--embedding-model", "nomic-embed"
      ]
    }
  }
}
```

Or use environment variables:

```json
{
  "mcpServers": {
    "tessera": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/tessera",
        "run", "python", "-m", "tessera", "serve",
        "--project", "/absolute/path/to/your/project"
      ],
      "env": {
        "TESSERA_EMBEDDING_ENDPOINT": "http://localhost:8800/v1/embeddings",
        "TESSERA_EMBEDDING_MODEL": "nomic-embed"
      }
    }
  }
}
```

### Multi-Project Configuration

To query multiple projects, use multi-project mode:

```json
{
  "mcpServers": {
    "tessera": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/tessera",
        "run", "python", "-m", "tessera", "serve"
      ],
      "env": {
        "TESSERA_EMBEDDING_ENDPOINT": "http://localhost:8800/v1/embeddings",
        "TESSERA_EMBEDDING_MODEL": "nomic-embed"
      }
    }
  }
}
```

After saving `.mcp.json`, restart Claude Code (or toggle the MCP server in Settings). The tessera server will load and its 18 tools become available.

## Embedding Server Setup (Optional)

Embeddings enable semantic search — finding code by concept, not just keywords. Without embeddings, Tessera uses FTS5 keyword search, which is fast and works well for precise queries.

### Option 1: LM Studio (recommended for local development)

[LM Studio](https://lmstudio.ai/) provides a UI for running local models.

1. Download and install LM Studio
2. Search for and load the `nomic-embed-text` model
3. In the Local Server tab, set the model to `nomic-embed-text` and start the server (listens on `http://localhost:8000` by default)
4. Test the endpoint:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "encode this text",
    "model": "nomic-embed"
  }'
```

You should see:

```json
{
  "data": [
    {
      "embedding": [0.123, -0.456, ...],
      "index": 0,
      "object": "embedding"
    }
  ],
  "model": "nomic-embed",
  "object": "list",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

### Option 2: Ollama

[Ollama](https://ollama.ai/) provides a CLI for running models.

1. Install Ollama
2. Pull an embedding model:

```bash
ollama pull nomic-embed-text
```

3. Run the embedding server on port 8800:

```bash
ollama serve --listen 0.0.0.0:8800
```

4. Verify the endpoint is accessible:

```bash
curl http://localhost:8800/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "nomic-embed-text"}'
```

### Option 3: llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp) is lightweight and fast.

1. Clone the repository:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
```

2. Download an embedding model (e.g., nomic-embed-text as a GGUF):

```bash
wget https://huggingface.co/nomic-ai/nomic-embed-text-1.5.5-gguf/resolve/main/nomic-embed-text-v1.5.f16.gguf
```

3. Run the server:

```bash
./server -m nomic-embed-text-v1.5.f16.gguf \
  --embedding \
  --port 8800
```

4. Verify:

```bash
curl http://localhost:8800/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "nomic"}'
```

### Configuring Tessera to Use Your Endpoint

Once your embedding server is running on a local port, pass it to Tessera during indexing and serving:

```bash
# During indexing
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings \
  --embedding-model nomic-embed-text

# When starting the server
uv run python -m tessera serve --project /path/to/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings \
  --embedding-model nomic-embed-text
```

The embedding dimension is auto-detected — no configuration needed beyond the endpoint URL.

## Your First Queries

Once the MCP server is running and connected in Claude Code, you can call Tessera tools. Here are three common queries:

### 1. Search for Symbols

Find all functions named `search`:

```
symbols(query="search")
```

Response:

```json
[
  {
    "file": "src/tessera/search.py",
    "kind": "function",
    "line": 45,
    "name": "search",
    "scope": "module",
    "signature": "def search(query: str, limit: int = 10) -> list[dict]"
  },
  {
    "file": "src/tessera/indexer.py",
    "kind": "function",
    "line": 120,
    "name": "search_project",
    "scope": "IndexerPipeline"
  }
]
```

### 2. Search Code and Docs

Find mentions of "authentication" across code and documents:

```
search(query="authentication", limit=5)
```

Response:

```json
{
  "results": [
    {
      "file": "src/tessera/auth.py",
      "kind": "code",
      "line": 10,
      "text": "def validate_session(conn: sqlite3.Connection, session_id: str)",
      "relevance": 0.92
    },
    {
      "file": "README.md",
      "kind": "document",
      "section": "Security",
      "text": "Session-based authentication with deny-by-default scope gating.",
      "relevance": 0.87
    }
  ]
}
```

### 3. Find References to a Symbol

See everywhere that the `validate_session` function is called:

```
references(symbol_name="validate_session")
```

Response:

```json
{
  "outgoing": [
    {
      "kind": "call",
      "file": "src/tessera/server.py",
      "line": 156,
      "text": "scope = validate_session(conn, session_id)"
    }
  ],
  "callers": [
    {
      "kind": "call",
      "file": "src/tessera/server/_state.py",
      "line": 71,
      "text": "scope = validate_session(_global_db.conn, session_id)"
    }
  ]
}
```

## Development & Testing

### Run Tests

```bash
uv run pytest tests/ -v
```

### Verbose Indexing

See detailed logging during indexing:

```bash
uv run python -m tessera index /path/to/project \
  --verbose \
  --embedding-endpoint http://localhost:8800/v1/embeddings
```

### Check Indexing Status

From Claude Code (or manually), call:

```
status()
```

Response:

```json
{
  "projects": [
    {
      "id": 1,
      "name": "my-project",
      "path": "/path/to/my-project",
      "indexed_at": "2025-03-03T12:34:56",
      "file_count": 247,
      "symbol_count": 1340,
      "has_embeddings": true
    }
  ],
  "global_db": "~/.tessera/global.db"
}
```

### Reindex After Code Changes

After you modify code, re-index incrementally:

```bash
uv run python -m tessera index /path/to/project \
  --incremental \
  --embedding-endpoint http://localhost:8800/v1/embeddings
```

If the parser changes (Tessera updates), force a full reindex:

```bash
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings
```

## Troubleshooting

### Embedding endpoint not reachable

**Problem:** `EmbeddingUnavailableError: Failed to reach embedding endpoint`

**Solution:** Verify your embedding server is running:

```bash
curl http://localhost:8800/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "nomic-embed"}'
```

If the endpoint is unreachable, start your embedding server (LM Studio, Ollama, or llama.cpp) before indexing or serving.

### MCP server not loading in Claude Code

**Problem:** Tessera tools don't appear in Claude Code after adding `.mcp.json`

**Solution:**
1. Verify paths in `.mcp.json` are absolute (not relative)
2. Restart Claude Code completely
3. Check Claude Code logs (Settings > Logs) for MCP startup errors
4. Test the command manually:

```bash
uv --directory /path/to/tessera run python -m tessera serve --project /path/to/project
```

Should output `Listening on stdio` and wait for input.

### Stale index warning

**Problem:** `⚠ Stale index detected for: project-name`

**Solution:** The parser has changed since the index was created. Run:

```bash
uv run python -m tessera index /path/to/project --force
```

This forces a full re-index with the current parser.

### No symbols found

**Problem:** Symbol search returns empty results

**Solution:**
1. Verify the project was indexed: check for `.tessera/project.db`
2. Check file count in status: `status()` should show files indexed > 0
3. Re-index with verbose logging:

```bash
uv run python -m tessera index /path/to/project \
  --verbose
```

Look for parsing errors or files being skipped.

## Next Steps

- **Search Guide** — Advanced keyword and semantic search queries
- **Code Intelligence** — Symbol lookup, reference tracing, impact analysis
- **Access Control** — Creating scoped tokens for sub-agents
- **Architecture** — How Tessera indexes and searches under the hood

Start by searching your project. Try `search(query="your function name")` or `symbols(query="main")` to explore what Tessera has indexed.
