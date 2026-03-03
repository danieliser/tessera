# Indexing & Maintenance Guide

This guide covers how Tessera indexes your projects, what gets indexed, and how to maintain indexes over time.

## What Gets Indexed

Tessera indexes three categories of content: code files, documents, and media assets.

### Code Files

Supported languages are parsed via tree-sitter into symbols, references, and edges:

- **PHP** — functions, classes, methods, properties, interfaces, traits, and hooks
- **TypeScript** — functions, classes, methods, interfaces, type aliases, enums
- **JavaScript** — same as TypeScript (.js and .jsx files)
- **Python** — functions, classes, methods, async definitions
- **Swift** — classes, structs, enums, functions, methods

Each symbol is extracted with its scope, type, and dependencies. References (function calls, imports, inheritance) are cross-linked into a dependency graph.

Code files are split into **AST-aware chunks** using the cAST (Code-Aware Structural) chunker:

- Definition nodes (functions, classes) become their own chunks
- Non-definition nodes (module-level code) are merged up to a 512-character budget
- Each chunk is tagged with its AST type and line range
- Chunks are then embedded (if an embedding endpoint is configured)

### Document Files

Documents are chunked by their structure, not line-based:

| Format | Chunking Strategy |
|--------|-------------------|
| **Markdown** | By heading hierarchy (H1, H2, H3) — preserves section tree structure |
| **PDF** | By page, with text extraction via pymupdf4llm |
| **YAML / JSON** | By key-path (e.g., `config.database.host`) — respects nesting |
| **HTML / XML** | Tag stripping + plaintext chunking — preserves semantic structure |
| **Plaintext** (.txt, .rst, .csv, .log, .ini, .cfg, .toml, .conf, etc.) | Line-based chunking |

### Media & Binary Assets

Assets are **not** extracted for content. Instead, their metadata is indexed:

- **Images** (PNG, JPEG, GIF, BMP) — filename, path, MIME type, dimensions, file size
- **SVG** — indexed both as XML document and as image asset
- **Video** (MP4, WebM, MOV, etc.) — filename, path, MIME type, duration (if extractable), file size
- **Audio** (MP3, WAV, FLAC, etc.) — filename, path, MIME type, duration, file size
- **Fonts** (TTF, OTF, WOFF, etc.) — filename, path, MIME type, file size
- **Archives** (ZIP, TAR, GZ, etc.) — filename, path, MIME type, file size

Assets are searchable by name, path, MIME type, and can be filtered in search results via `source_type: "asset"`.

### Files Excluded from Indexing

The `.tesseraignore` file controls what's indexed. Tessera applies a **two-tier ignore system**:

**Security patterns** (locked, cannot be negated by project config):
```
.env*
*.pem
*.key
*.p12
*.pfx
*credentials*
*secret*
id_rsa
id_ed25519
*.token
service-account.json
```

**Default patterns** (merged with custom `.tesseraignore`):
```
.git/
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.egg-info/
dist/
build/
node_modules/
npm-debug.log
.npm/
vendor/
composer.lock
.next/
out/
.turbo/
.vscode/
.idea/
*.swp
*.swo
.DS_Store
.tsc/
coverage/
.nyc_output/
.tessera/
*.log
.gitignore
```

To customize, create `.tesseraignore` in your project root with [.gitignore syntax](https://git-scm.com/docs/gitignore). Example:

```
# .tesseraignore
# Skip build output
dist/
build/

# Skip test snapshots
__snapshots__/

# Keep node_modules (override default)
!node_modules/app-critical/

# Don't index vendor (keep default)
vendor/
```

**Important:** Attempting to negate a security pattern (e.g., `!.env*`) logs a warning and is ignored.

## Indexing from the CLI

The CLI is the primary way to perform initial or full reindexing of a project.

### Basic Index

Index a project without embeddings (keyword-only search):

```bash
uv run python -m tessera index /path/to/project
```

Output:
```
Full index: /path/to/project
Embedding endpoint: None — indexing without embeddings.

Done in 3.2s
  Files: 42 indexed, 8 skipped, 0 failed
  Symbols: 156
  Chunks: 487 (0 embedded)
```

### With Embeddings

Index with semantic search enabled (requires a running embedding endpoint):

```bash
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings \
  --embedding-model nomic-embed-text
```

Output:
```
Full index: /path/to/project
Embedding endpoint: http://localhost:8800/v1/embeddings (model: nomic-embed-text)

Done in 12.5s
  Files: 42 indexed, 8 skipped, 0 failed
  Symbols: 156
  Chunks: 487 (487 embedded)
```

### Incremental Index

Only re-index files changed since the last indexed commit (requires a git repository):

```bash
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings \
  --incremental
```

Incremental mode is much faster because it:

1. Detects the last indexed commit from the project's global database record
2. Uses `git diff` to find changed files
3. Re-indexes only changed files
4. Deletes index entries for deleted files
5. Resolves cross-file edges for changed symbols

### Verbose Logging

Enable debug logging to see file-by-file progress:

```bash
uv run python -m tessera index /path/to/project -v
```

This logs each file as it's processed, symbol extraction, and any errors.

## How Change Detection Works

Tessera uses **SHA-256 file hashes** to detect changes:

1. **First index**: Computes and stores the hash of every indexed file
2. **Subsequent indexes**: Compares current file hash against stored hash
3. **Unchanged**: File is skipped (default behavior)
4. **Changed**: File is re-indexed, old data cleared, new data inserted
5. **Deleted**: Index entries removed during incremental indexing

**When file hashes are salted:**

In v0.6.0+, file hashes are salted with the Tessera package version. When you upgrade Tessera, hashes no longer match — triggering re-indexing even for unchanged files. This ensures your index always reflects the current parser and chunker behavior.

**Orphan cleanup:**

After incremental indexing, Tessera removes database records for files that no longer exist on disk. This keeps the index synchronized with the project state.

## MCP Reindex Tool

For agents and programmatic control, use the MCP `reindex` tool.

### Full Reindex

Re-indexes all files in a project, regardless of change status:

```python
# MCP tool call
reindex(project_id=1, mode="full")
```

Returns:
```json
{
  "project_id": 1,
  "files_processed": 42,
  "files_skipped": 0,
  "files_failed": 0,
  "symbols_extracted": 156,
  "chunks_created": 487,
  "time_elapsed": 12.5
}
```

Use full reindex when:

- Initial indexing
- After upgrading Tessera (to refresh parser outputs)
- After changing `.tesseraignore` or parser configuration
- To clear stale index data

### Incremental Reindex

Only re-index changed files (requires git history):

```python
reindex(project_id=1, mode="incremental")
```

Much faster for large projects with few changes. Falls back to full index if git history is unavailable.

### Force Reindex

Force re-index all files, bypassing change detection:

```python
reindex(project_id=1, force=True)
```

Sets the parser digest (see below) to match the current Tessera version, clearing the stale index warning. Use after fixing a bug in the indexer that produced incorrect results.

## Stale Index Detection

Tessera automatically detects when an index was built with an older version of the parser and warns you to update it.

### How It Works

On every index run, Tessera computes a **parser digest**: a SHA-256 hash of all parser and chunker source files:

```python
# From _helpers.py
def compute_parser_digest() -> str:
    """Hash of all parser/*.py and chunker*.py files."""
    pkg_root = Path(__file__).resolve().parent.parent
    source_files = sorted([
        *pkg_root.glob("parser/*.py"),
        *pkg_root.glob("chunker*.py"),
    ])
    h = hashlib.sha256()
    for path in source_files:
        h.update(path.read_bytes())
    return h.hexdigest()[:16]
```

This digest is stored in the project database (in the `_meta` table, key: `parser_digest`).

### Stale Index Warning

When the MCP server starts, it:

1. Loads each project's database
2. Retrieves the stored `parser_digest`
3. Compares it against the current digest
4. If mismatch: adds the project to `_stale_projects` set

On any search or navigation call, if a stale project is in scope, a warning is returned:

```
⚠ Stale index detected for: my-project.
The parser has changed since last indexing. Run `reindex(project_id=..., force=True)` to update.
```

### Why This Matters

Parser upgrades (e.g., improvements to symbol extraction, new language support, chunk boundaries) can change how code is indexed. An old index might:

- Miss newly indexed symbols
- Return incorrect reference chains
- Produce chunk boundaries that don't match current code structure

**Fix:** Run `reindex(project_id=1, force=True)` to refresh the index with current parser behavior.

## Embedding Setup (Optional)

Tessera works without embeddings — keyword search via FTS5 is fully functional. For semantic search (query-document similarity), configure an embedding endpoint.

### Requirements

Any OpenAI-compatible `/v1/embeddings` endpoint. No special authentication or model version needed — Tessera auto-detects embedding dimensions.

### Recommended Setups

**Local: LM Studio**

1. Download [LM Studio](https://lmstudio.ai/)
2. Download the `nomic-embed-text` model (small, fast, ~300MB)
3. Start the server: click "Start Server" (default: http://localhost:1234/v1/embeddings)
4. Index with embeddings:

```bash
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:1234/v1/embeddings \
  --embedding-model nomic-embed-text
```

**Local: Ollama**

1. Install [Ollama](https://ollama.ai/)
2. Pull the embedding model:

```bash
ollama pull nomic-embed-text
ollama serve
```

3. Index:

```bash
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:11434/api/embeddings \
  --embedding-model nomic-embed-text
```

### How Embeddings Work

When indexing with embeddings:

1. Each chunk's content is sent to the endpoint
2. The endpoint returns an embedding vector (typically 768-1024 dimensions)
3. Embeddings are stored in FAISS (vector database)
4. At search time, query embeddings are compared against stored embeddings via cosine similarity

### Graceful Degradation

If the embedding endpoint is down during indexing:

- Chunks are indexed without embeddings
- A warning is logged
- Search falls back to keyword-only mode
- No errors or failures

Restart the endpoint and reindex to add embeddings to existing chunks.

## Embedding Dimension Auto-Detection

Tessera automatically detects the embedding dimension from your model:

1. Embeds a short sample text: `"test"`
2. Records the response vector dimension
3. Creates the FAISS index with that dimension
4. Stores dimension in project metadata

**If dimensions change** (e.g., you switch from `nomic-embed-text` (768D) to `bge-base` (768D) or mismatch): the search will fail. Use the drift adapter (below) to migrate.

## Drift Adapter: Switching Embedding Models

If you want to change embedding models without re-indexing, use the drift adapter to train a rotation matrix that maps old embeddings to the new model's space.

### When to Use

- Switching to a better model (e.g., `nomic-embed-text` → `bge-large`)
- Upgrading model versions (e.g., `nomic-embed-text-v2` → `nomic-embed-text-v2.5`)
- Fixing a dimension mismatch

### Train the Adapter

```python
drift_train(sample_size=200)
```

1. Samples 200 random chunks from the index
2. Re-embeds them with the new endpoint + model
3. Trains an Orthogonal Procrustes rotation matrix to align old and new embeddings
4. Saves the matrix to `~/.tessera/data/{project-slug}/drift_matrix.npy`

### Adapter Performance

- **Per-query overhead**: <10 microseconds (negligible)
- **Accuracy**: Typically 95%+ cosine similarity between old and new embeddings
- **Valid for**: Same project, any embedding model

### Example: Upgrade Models

```bash
# Current setup: nomic-embed-text on localhost:1234
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:1234/v1/embeddings \
  --embedding-model nomic-embed-text

# Later: switch to bge-base (better quality, same 768D)
# Kill LM Studio, load bge-base in Ollama instead
ollama pull bge-base
ollama serve

# Don't re-index. Train drift adapter:
drift_train(sample_size=200)

# Search now uses the rotation matrix to map old embeddings → bge-base space
```

Drift training typically takes 1-2 seconds. Searches are unaffected.

## Index Storage Location

Indexes are stored in:

```
~/.tessera/data/{project-slug}/
├── index.db          # SQLite: symbols, references, chunks, files
├── index.db-shm      # SQLite write-ahead log (WAL)
├── index.db-wal      # SQLite write-ahead log
├── embeddings.idx    # FAISS index (vector database)
├── embeddings.dat    # FAISS data
└── drift_matrix.npy  # Drift adapter (if trained)
```

`{project-slug}` is derived from the project path: `/Users/you/Projects/my-app` → `-Users-you-Projects-my-app`.

Size estimates for a medium project (500 files, 1000 symbols):

- `index.db`: 5-20MB (depends on chunk count, reference density)
- `embeddings.idx + .dat`: 50-200MB (depends on chunk count and embedding dimension)
- **Total**: ~60-220MB per project

Large projects (2000+ files) can consume 500MB+ per project.

## Performance Expectations

Indexing times vary by hardware, file count, and whether embeddings are computed. These are approximate baselines on a modern laptop (M1, 8GB RAM):

| Project Size | Files | Symbols | No Embeddings | With Embeddings |
|--------------|-------|---------|---------------|-----------------|
| Small | <100 | <500 | 1-3s | 5-15s |
| Medium | 200-500 | 1-2K | 5-15s | 20-60s |
| Large | 1000+ | 3-5K | 15-45s | 60-180s |

**Key factors:**

- **File count**: Most indexing time is file I/O and parsing
- **Embeddings**: ~1ms per chunk for local models (adds 20-80% overhead)
- **Language**: PHP is slightly slower than Python/TS due to grammar complexity
- **Disk I/O**: Indexing is I/O-bound; SSD vs. HDD makes a big difference

**Incremental indexing** is 10-50x faster because it only touches changed files.

## Troubleshooting

### Index Creation Fails

**Symptom**: `Files: 0 indexed, 0 skipped, 1 failed`

**Check:**

1. Project path exists: `ls -la /path/to/project`
2. Tessera data directory is writable: `ls -la ~/.tessera/data/`
3. Verbose logging shows the error: `uv run python -m tessera index /path/to/project -v`

### Files Marked as Skipped

**Symptom**: `Files: 10 indexed, 32 skipped, 0 failed`

**Reason:** Files have unchanged hashes (incremental mode). To force re-index:

```bash
uv run python -m tessera index /path/to/project --incremental
# OR
reindex(project_id=1, force=True)
```

### Stale Index Warning After Upgrade

**Symptom**: Search results include warning: `Stale index detected for: my-project`

**Fix:** Tessera's parser changed. Update the index:

```bash
reindex(project_id=1, force=True)
```

This re-computes the parser digest and clears the stale flag.

### Embedding Endpoint Unavailable

**Symptom**: `Embedding endpoint unavailable, storing document chunks without embeddings`

**Reason:** Embedding server is down or unreachable.

**Fix:**

1. Start the embedding server (LM Studio, Ollama, etc.)
2. Verify the endpoint: `curl http://localhost:8800/v1/embeddings` (adjust URL/port as needed)
3. Re-index to add embeddings:

```bash
uv run python -m tessera index /path/to/project \
  --embedding-endpoint http://localhost:8800/v1/embeddings
```

### Index Database Locked

**Symptom**: `database is locked`

**Reason:** Multiple indexing processes or stale file locks.

**Fix:**

1. Stop all indexing processes
2. Remove stale WAL files (if indexing crashed):

```bash
rm ~/.tessera/data/{project-slug}/index.db-*
```

3. Retry indexing

## Index Maintenance

### Periodic Full Reindex

For long-lived projects, do a full reindex occasionally to ensure consistency:

```bash
reindex(project_id=1, force=True)
```

Recommended: After major Tessera version upgrades, after significant refactoring, or monthly for active projects.

### Monitor Index Health

Check the `status` tool to see project metadata:

```python
status(project_id=1)
```

Returns project metadata including:
- Last indexed commit (for incremental indexing)
- Files indexed, chunks created
- Index size on disk

### Clean Up Old Indexes

If you no longer need a project's index:

```bash
rm -rf ~/.tessera/data/{project-slug}/
```

This frees up disk space. The next time you index the project, a fresh index is created.

### Backup Indexes

To preserve indexes across machine changes:

```bash
# Backup
cp -r ~/.tessera/data /backup/tessera-indexes

# Restore
cp -r /backup/tessera-indexes ~/.tessera/data
```

Tessera automatically upgrades old schema versions on startup.

## Best Practices

1. **Run initial index once** — Use CLI for first index, then incremental for updates
2. **Enable embeddings early** — Switching later requires drift training or re-indexing
3. **Use `.tesseraignore` proactively** — Add exclusions for build artifacts, vendor, test snapshots early
4. **Reindex after major refactors** — Parser behavior may have changed; force reindex to keep index accurate
5. **Monitor stale warnings** — Upgrade stale indexes promptly with `force=True` reindex
6. **Backup before major upgrades** — If you're using Tessera in production, back up `~/.tessera` before upgrading to a new major version

## FAQ

**Q: Can I index multiple projects in parallel?**

A: Yes. Each project has its own database, so concurrent indexing is safe. Use separate CLI commands or MCP tool calls for each project.

**Q: Does incremental indexing miss changes?**

A: No. It uses `git diff` to detect all file changes (added, modified, deleted). Files with unchanged hashes are skipped.

**Q: What if I move a project to a new path?**

A: The index is stored by path slug, so moving a project breaks the index link. The next index attempt creates a new index at the new path. To preserve the index, update the project's path in the global database (global.db).

**Q: Can I index a large codebase incrementally?**

A: Yes, incremental is designed for large projects. On a 5000-file project, incremental indexing typically completes in 5-10 seconds if only a few files changed.

**Q: How much disk space does an index use?**

A: Roughly:
- 10KB per symbol
- 100KB per 100 chunks (without embeddings)
- 500KB per 100 chunks (with embeddings, depending on embedding dimension)

A medium project (500 files, 1K symbols, 2K chunks) uses ~100-150MB.

**Q: Can I use a cloud embedding endpoint?**

A: Yes, any OpenAI-compatible endpoint works. However, network latency will slow indexing. For production, a local endpoint is recommended.

**Q: What if my embedding model changes dimensions?**

A: Use the drift adapter to migrate without re-indexing. If dimensions change and you don't use drift, searches will fail. Fix with `drift_train()` or re-index.
