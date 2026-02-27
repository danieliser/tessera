# Spec: Phase 4 — Document Indexing + Drift-Adapter + Per-Project Ignore Config

**Version:** v2
**Date:** 2026-02-27
**Author:** spec-writer
**Project:** Tessera
**Status:** Phase 4 Deliverables Ready for Implementation

---

## Changes from v1

This revision incorporates structured feedback from the panel review (consensus and critical items). Key changes:

1. **Security Risk Register (NEW)** — Added comprehensive security section addressing credential file indexing, prompt injection via document content, numpy load validation, and YAML safety.

2. **Two-Tier Ignore System** — IgnoreFilter now has security-critical patterns (un-negatable) and user-configurable defaults. Security patterns cannot be overridden via `.tesseraignore`.

3. **Drift-Adapter Operational Workflow (EXPANDED)** — Detailed startup activation, training trigger via new `drift_train()` MPC tool, validation loop, and rollback mechanism. Explicit limitation: infrastructure-grade tooling.

4. **Incremental Re-Indexing for Documents (EXTENDED)** — `_get_changed_files()` and `_get_deleted_files()` now handle document extensions. Explicit requirement in indexer.py updates.

5. **FAISS Source-Type Filtering Strategy (NEW)** — Over-fetch + post-filter approach to avoid code-only results when filtering documents. Fetch 3x limit, post-filter, return what remains.

6. **DB Migration Strategy (NEW)** — Concrete schema_version table approach with conditional migrations on database open.

7. **Error Isolation in Document Indexing (NEW)** — Document extraction errors caught, logged, and indexed file status set to 'failed' without breaking pipeline.

8. **LOC Budget Clarification** — 1,500 LOC is PRODUCTION CODE ONLY. Tests budgeted separately at 200–300 LOC. Revised breakdown realistic by module.

Preserved from v1: Unified chunk_meta approach, yaml.safe_load usage, parameterized SQL, scope gating, IgnoreFilter defaults, FTS-only fallback, and all non-goals.

---

## Executive Summary

Phase 4 extends Tessera's indexing pipeline to handle non-code documents (PDF, Markdown, YAML, JSON) alongside existing code intelligence. Three orthogonal modules are added:

1. **Document Indexing** — Extract text from PDFs (via PyMuPDF4LLM), split Markdown by headers, parse YAML/JSON structurally. Chunks flow through the same embedding and search pipeline as code, enabling unified search across code + docs.

2. **Drift-Adapter** — Handle embedding model migration without full re-indexing via Orthogonal Procrustes transformation (scipy). Train on 1–5% corpus sample in <2 minutes; recover 95%+ retrieval recall with zero query latency overhead.

3. **Per-Project Ignore Config** — `.tesseraignore` file using `.gitignore` syntax (via pathspec library) to exclude files/patterns from indexing. Two-tier system: security-critical patterns (credentials, secrets) cannot be negated by users; standard defaults are user-overridable.

**Key architectural decision**: Extend existing `chunk_meta` SQLite table with nullable `source_type`, `section_heading`, `key_path`, `page_number`, `parent_section` columns. Single FAISS index and search pipeline handles both code and documents. No separate doc-specific tables or schemas needed.

**Success metrics**:
- Extract + index 50-page PDF in <30 seconds
- `search()` returns mixed code + document results ranked via RRF
- `doc_search()` tool returns document-only results
- Per-project `.tesseraignore` excludes specified patterns; security patterns cannot be negated
- Drift-Adapter trains on 5% sample in <2 minutes, recovers 95%+ recall
- Document search federates across collections (reuses Phase 3 RRF pattern)
- Total Phase 4 code: <1,500 LOC (production only; tests 200–300 LOC separately)

---

## Problem Statement

### What's the problem?

Developers need persistent searchable access to non-code context: API docs (PDFs), architecture decisions (Markdown), configuration schemas (YAML/JSON). Currently, Tessera indexes code only, leaving 30% of the knowledge base unsearchable.

Additionally:
- When embedding models upgrade (e.g., from nomic-embed-text to a newer model), re-embedding 1M+ chunks is prohibitively expensive (0.25–0.5 GPU-hours). A lightweight transformation-based approach (Drift-Adapter) enables model migration in minutes, not hours.
- Per-project ignore patterns (`.tesseraignore`) are needed to exclude vendor code, generated files, and confidential data — currently Tessera skips these via hardcoded defaults only.
- Security risk: Credential files (`.env`, `.pem`, `service-account.json`) must never be indexed, even if accidentally included in project directory.

### Who has it?

1. **Always-on agents** — Need searchable memory of code + docs to answer "how does this API work?" without re-reading documentation
2. **Development teams** — Want to exclude `vendor/`, `node_modules/`, and custom paths from indexing without polluting the shared schema
3. **Operations teams** — Need to swap embedding models without full re-indexing when performance/cost requirements change
4. **Security teams** — Need guarantees that credential files are never indexed, regardless of user configuration

### How painful is it?

**Without Phase 4:**
- Agents must manually navigate between code search results and external docs
- Document searches happen outside Tessera (separate vector DB, LLM context, manual retrieval)
- Embedding model upgrades force expensive re-indexing or stale vectors (poor recall)
- Ignore patterns are hardcoded; projects cannot exclude domain-specific data
- Credential leakage risk if Tessera is exposed to projects containing `.env`, API keys, or certificates

**With Phase 4:**
- `search("how do I handle webhook failures?")` returns code examples + relevant API docs in a single ranked list
- Model migration: train Procrustes on 50 samples (2 minutes), query-time transformation adds <1μs overhead
- Projects create `.tesseraignore` to exclude confidential configs, third-party bundles, generated code
- Security-critical patterns (`.env*`, `*.pem`, `*credentials*`, etc.) are always excluded, un-negatable by user configuration

---

## Proposed Solution

### High-Level Architecture

Document indexing extends the existing Phase 1–3 pipeline:

```
Project File Walk
    ↓
    ├─ IgnoreFilter (security + user patterns)
    │  (always excludes: .env*, *.pem, *.key, *credentials*, id_rsa, etc.)
    │  (user can override: node_modules, vendor, custom paths)
    │
    ├─ Code files (.php, .ts, .py, etc.)
    │  ↓
    │  [Existing: parse + extract symbols]
    │  ↓
    │  chunk_meta table
    │  (source_type='code')
    │  ↓
    │  FAISS embeddings
    │
    └─ Non-code files (.pdf, .md, .yaml, .json)
       ↓
       [NEW Phase 4: extract + chunk documents]
       ↓
       chunk_meta table
       (source_type='markdown'/'pdf'/'yaml'/'json')
       ↓
       FAISS embeddings (same index)

Drift-Adapter (on startup & migration):
    ├─ Check for .tessera/data/{slug}/drift_matrix.npy
    ├─ If present: load rotation matrix, inject adapt_query() into search pipeline
    └─ If absent: use raw embeddings

Unified Search (Phase 1 modified):
    ├─ Keyword search (FTS5, with source_type filter)
    ├─ Semantic search (FAISS, with over-fetch + post-filter for source_type)
    └─ RRF merge (handles source_type filtering + federated collections)
       ↓
       Results: mixed code + doc chunks, ranked by RRF
```

### Core Deliverables

**1. Document Indexing Module** (`src/tessera/document.py`)
- PDF extraction via PyMuPDF4LLM (0.12s per document)
- Markdown splitting by headers with hierarchy metadata
- YAML/JSON structural parsing (key-path chunks)
- Error isolation: all extraction failures wrapped in `DocumentExtractionError`, logged, indexed as 'failed'
- Unified `DocumentChunk` dataclass for all document types

**2. Drift-Adapter Module** (`src/tessera/drift_adapter.py`)
- Orthogonal Procrustes transformation via scipy.linalg
- Training on corpus sample (1–5%)
- Query-time transformation for migrated embeddings
- Persistent storage of rotation matrix as numpy .npy file
- Validation: array dimension and dtype checks on load

**3. Ignore Config Module** (`src/tessera/ignore.py`)
- Two-tier ignore system:
  - Security patterns: un-negatable (`.env*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*credentials*`, `*secret*`, `id_rsa`, `id_ed25519`, `*.token`, `service-account.json`)
  - Default patterns: user-overridable (node_modules, vendor, .git, build artifacts, etc.)
- `.tesseraignore` file parsing (gitignore syntax via pathspec)
- Per-project override + merge logic
- Log warning if user attempts to negate a security pattern

**4. Schema Extensions** (`src/tessera/db.py`)
- Add `schema_version` integer to new `_meta` table
- ALTER TABLE chunk_meta to add document-specific columns
- Backward-compatible with existing code indexing
- Migration on database open: check schema version, run conditional ALTER statements wrapped in try/except

**5. Indexer Pipeline Updates** (`src/tessera/indexer.py`)
- Document discovery: `_get_changed_files()` includes document extensions (`.pdf`, `.md`, `.yaml`, `.yml`, `.json`)
- Document file deletion: `_get_deleted_files()` removes chunks from both DB and FAISS
- Document extraction orchestration via IgnoreFilter (security patterns checked first)
- Embedding calls for document chunks
- Error isolation: wrap `_index_document_file()` in try/except, log, increment `IndexStats.files_failed`, continue

**6. Search Pipeline Updates** (`src/tessera/search.py`)
- New `source_type` filter parameter in `search()`
- New `doc_search()` convenience tool
- FTS5 search with source_type filtering (existing code)
- Semantic search (FAISS) with **over-fetch + post-filter strategy**:
  - When `source_type` filter active: fetch `limit * 3` from FAISS
  - Post-filter by source_type using chunk_meta lookup
  - If <limit results remain, return what's available
  - Avoids separate FAISS indices while ensuring adequate document results

**7. MCP Server Updates** (`src/tessera/server.py`)
- Register `doc_search()` tool
- Register new `drift_train(sample_size: int = 50)` tool for infrastructure operators
- `drift_train()` returns `{"recall_at_10": float, "matrix_path": str, "sample_size": int}`

---

## Architecture Details

### 1. Document Extraction & Chunking (Revised in v2 — error handling)

#### PDF Extraction

```python
class DocumentExtractionError(Exception):
    """Raised when document extraction fails."""
    pass

async def extract_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF to markdown using PyMuPDF4LLM.

    Preserves headers (detected via font size), lists, tables, code blocks.

    Args:
        pdf_path: Absolute path to .pdf file

    Returns:
        Markdown string

    Raises:
        DocumentExtractionError: If PDF is corrupted, unreadable, or extraction fails
    """
```

**Implementation notes:**
- Uses `pymupdf4llm.to_markdown()` (v0.2.0+)
- Runs in thread pool to avoid blocking event loop
- All exceptions caught and wrapped in `DocumentExtractionError`
- Caches results if file not modified (hash-based)

#### Markdown Chunking

```python
@dataclass
class MarkdownChunk:
    content: str                  # Text content of chunk
    headers: dict[str, str]       # Section hierarchy
    start_line: int               # Line number in original file
    end_line: int                 # Line number in original file
    source_type: str = "markdown" # Constant

def chunk_markdown(
    markdown_text: str,
    max_chunk_size: int = 1024,
    overlap: int = 128,
    split_headers: list[str] = ["#", "##", "###"]
) -> list[MarkdownChunk]:
    """Split markdown by headers, preserving hierarchy metadata."""
```

#### YAML/JSON Chunking

```python
@dataclass
class ConfigChunk:
    content: str              # YAML/JSON fragment (pretty-printed)
    key_path: str             # "server.port" or "database.credentials"
    parent_section: str       # "server" or "database"
    source_type: str = "yaml" # 'yaml' or 'json'
    source_file: str = ""     # e.g., "docker-compose.yml"

def chunk_yaml(yaml_path: str, max_chunk_size: int = 2048) -> list[ConfigChunk]:
    """
    Parse YAML and chunk by top-level keys (sections).

    Args:
        yaml_path: Absolute path to .yaml or .yml file
        max_chunk_size: Max chunk size in characters (default 2048)

    Returns:
        List of ConfigChunk objects with content, key_path, section hierarchy

    Raises:
        DocumentExtractionError: If YAML is malformed or unreadable
    """

def chunk_json(json_path: str, max_chunk_size: int = 2048) -> list[ConfigChunk]:
    """Same as chunk_yaml but for JSON files using json.loads()."""
```

**Implementation notes:**
- Uses `yaml.safe_load()` (no code execution risk)
- Uses `json.loads()` (stdlib, safe)
- Error handling: all exceptions wrapped in `DocumentExtractionError`

### 2. Drift-Adapter Module (Revised in v2 — validation + workflow)

#### DriftAdapter Class

```python
import numpy as np
from scipy.linalg import orthogonal_procrustes

class DriftAdapter:
    """Handle embedding model migration via Procrustes transformation."""

    def __init__(self, old_dim: int, new_dim: int):
        self.old_dim = old_dim
        self.new_dim = new_dim
        self.rotation_matrix = None

    def train(
        self,
        old_embeddings: np.ndarray,
        new_embeddings: np.ndarray
    ) -> np.ndarray:
        """Train Procrustes rotation matrix on embedding pairs."""
        old_centered = old_embeddings - old_embeddings.mean(axis=0)
        new_centered = new_embeddings - new_embeddings.mean(axis=0)

        self.rotation_matrix, scale = orthogonal_procrustes(
            old_centered,
            new_centered
        )
        return self.rotation_matrix

    def adapt_query(self, new_query_embedding: np.ndarray) -> np.ndarray:
        """Transform new model's query embedding to old model's space."""
        if self.rotation_matrix is None:
            raise ValueError("Must train() before adapt_query()")
        return new_query_embedding @ self.rotation_matrix.T

    def save(self, path: str):
        """Save rotation matrix to disk as float32 numpy array."""
        np.save(path, self.rotation_matrix.astype(np.float32))

    @staticmethod
    def load(path: str) -> "DriftAdapter":
        """
        Load rotation matrix from disk.

        Security: Loads with allow_pickle=False to prevent code execution.
        Validates shape and dtype.
        """
        embedding_dim = 768
        adapter = DriftAdapter(embedding_dim, embedding_dim)

        # Security: use allow_pickle=False
        matrix = np.load(path, allow_pickle=False)

        # Validate shape and dtype
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
        if matrix.dtype not in (np.float32, np.float64):
            raise ValueError(f"Expected float32/float64, got {matrix.dtype}")

        adapter.rotation_matrix = matrix.astype(np.float32)
        return adapter
```

#### Drift-Adapter Operational Workflow (Revised in v2 — NEW detailed section)

**Activation on Startup:**
- Server initialization checks for `.tessera/data/{slug}/drift_matrix.npy`
- If file exists and valid (shape + dtype checks pass): load adapter, inject `adapt_query()` into search pipeline
- If file absent or invalid: use raw embeddings from FAISS, no transformation
- On load failure: log warning, continue with raw embeddings

**Training Trigger:**
- New MCP tool `drift_train(sample_size: int = 50)` available to operators
- Implementation:
  1. Sample N random chunks from existing index
  2. Get old embeddings from FAISS for those chunks
  3. Embed the same chunks with the new model (via embedding endpoint)
  4. Train Procrustes on (old, new) pairs
  5. Compute recall@10 on held-out validation set (10% of samples)
  6. Save rotation matrix to `.tessera/data/{slug}/drift_matrix.npy`
  7. Return `{"recall_at_10": 0.97, "matrix_path": "...", "sample_size": 50}`

**Validation Loop:**
- Operator runs `drift_train()`, receives recall@10 metric
- If recall >= 0.95: operator manually updates embedding model in config, restarts server
- Server loads new matrix on next startup
- If recall < 0.90: escalate to Phase 5 alternatives or full re-indexing

**Rollback:**
- Delete `.tessera/data/{slug}/drift_matrix.npy`
- On next server startup, no adapter is loaded
- Queries use raw embeddings directly (old model space)

**Explicit Limitation:**
- This is infrastructure-grade tooling for ops teams, not production UX
- Requires manual model configuration changes and server restart
- Phase 5+ may add automated migration triggers

### 3. Per-Project Ignore Config (Revised in v2 — two-tier security)

#### IgnoreFilter Class

```python
import pathspec
from pathlib import Path

class IgnoreFilter:
    """Per-project ignore filter with two-tier security control."""

    # Security-critical patterns: CANNOT be negated by user configuration
    SECURITY_PATTERNS = [
        '.env*',
        '*.pem',
        '*.key',
        '*.p12',
        '*.pfx',
        '*credentials*',
        '*secret*',
        'id_rsa',
        'id_ed25519',
        '*.token',
        'service-account.json',
    ]

    # User-overridable default patterns
    DEFAULT_PATTERNS = [
        '.git/',
        '.gitignore',
        '__pycache__/',
        '*.pyc',
        '*.pyo',
        '.venv/',
        'venv/',
        '.egg-info/',
        'dist/',
        'build/',
        'node_modules/',
        'npm-debug.log',
        '.npm/',
        'vendor/',
        'composer.lock',
        '.next/',
        'out/',
        '.turbo/',
        '.vscode/',
        '.idea/',
        '*.swp',
        '*.swo',
        '.DS_Store',
        '.tsc/',
        'coverage/',
        '.nyc_output/',
        '.tessera/',
        '*.log',
    ]

    def __init__(self, project_root: Path, ignore_file: str = ".tesseraignore"):
        self.project_root = Path(project_root)
        self.ignore_file = self.project_root / ignore_file
        self.spec_security = None
        self.spec_user = None
        self.load()

    def load(self):
        """Load .tesseraignore from disk and merge with defaults."""
        # Security patterns are always applied (non-overridable)
        self.spec_security = pathspec.PathSpec.from_lines('gitwildmatch', self.SECURITY_PATTERNS)

        # User patterns (defaults + .tesseraignore overrides)
        patterns = self.DEFAULT_PATTERNS.copy()

        if self.ignore_file.exists():
            with open(self.ignore_file, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#'):
                        # Check if user is trying to negate a security pattern
                        if stripped.startswith('!'):
                            negated_pattern = stripped[1:].strip()
                            if any(negated_pattern == sp for sp in self.SECURITY_PATTERNS):
                                logger.warning(
                                    f"Attempted to negate security pattern '{negated_pattern}' in .tesseraignore; ignoring"
                                )
                                continue
                        patterns.append(stripped)

        self.spec_user = pathspec.PathSpec.from_lines('gitwildmatch', patterns)

    def should_ignore(self, rel_path: str) -> bool:
        """Check if a path should be ignored.

        Returns True if:
        - Path matches a security pattern (always enforced), OR
        - Path matches a user-overridable default pattern
        """
        # Always check security patterns first
        if self.spec_security.match_file(rel_path):
            return True

        # Then check user patterns
        return self.spec_user.match_file(rel_path)
```

### 4. Schema Extensions (Revised in v2 — migration strategy)

```python
class ProjectDB:
    CURRENT_SCHEMA_VERSION = 2

    def __init__(self, project_path: str):
        # ... existing init code ...
        self._run_migrations()

    def _run_migrations(self):
        """Run schema migrations if needed."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT value FROM _meta WHERE key = 'schema_version'")
            row = cursor.fetchone()
            current_version = int(row[0]) if row else 1
        except Exception:
            current_version = 1

        # Run migrations up to current version
        if current_version < 2:
            self._migrate_to_v2()

        # Update schema version
        try:
            cursor.execute("INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
                          ('schema_version', str(self.CURRENT_SCHEMA_VERSION)))
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update schema version: {e}")

    def _migrate_to_v2(self):
        """Migrate from schema v1 to v2: add document columns to chunk_meta."""
        cursor = self.conn.cursor()
        migrations = [
            "ALTER TABLE chunk_meta ADD COLUMN source_type TEXT DEFAULT 'code'",
            "ALTER TABLE chunk_meta ADD COLUMN section_heading TEXT",
            "ALTER TABLE chunk_meta ADD COLUMN key_path TEXT",
            "ALTER TABLE chunk_meta ADD COLUMN page_number INTEGER",
            "ALTER TABLE chunk_meta ADD COLUMN parent_section TEXT",
        ]

        for sql in migrations:
            try:
                cursor.execute(sql)
                logger.info(f"Schema migration: {sql}")
            except sqlite3.OperationalError as e:
                if "already exists" in str(e):
                    logger.debug(f"Column already exists, skipping: {sql}")
                else:
                    logger.error(f"Migration failed: {e}")
                    raise

        self.conn.commit()
```

### 5. Indexer Pipeline Updates (Revised in v2 — document support)

#### File Discovery for Documents

```python
def _discover_files(self) -> List[str]:
    """Walk project, return code AND document files."""
    code_extensions = {
        'php': ['.php'],
        'typescript': ['.ts', '.tsx'],
        'javascript': ['.js', '.jsx'],
        'python': ['.py'],
        'swift': ['.swift'],
    }
    document_extensions = ['.pdf', '.md', '.yaml', '.yml', '.json']

    allowed_exts = set()
    for lang in self.languages:
        allowed_exts.update(code_extensions.get(lang, []))
    allowed_exts.update(document_extensions)

    files = []
    for root, dirs, filenames in os.walk(self.project_path):
        dirs[:] = [
            d for d in dirs
            if not d.startswith('.') and d not in ('node_modules', 'vendor', '__pycache__', '.tessera')
        ]
        for f in filenames:
            if f.endswith('.d.ts'):
                continue
            if any(f.endswith(ext) for ext in allowed_exts):
                files.append(os.path.join(root, f))

    return sorted(files)
```

#### Changed Files Detection for Documents

```python
def _get_changed_files(self, since_commit: str) -> List[str]:
    """Get files changed since a commit, including document files."""
    # ... existing git diff code ...
    # Add document_extensions = ['.pdf', '.md', '.yaml', '.yml', '.json']
    # Merge with code_extensions when filtering results
```

#### Deleted Files for Documents

```python
def _get_deleted_files(self, since_commit: str) -> List[str]:
    """Get files deleted since a commit, including documents."""
    # Returns relative paths that were deleted
    # indexer removes chunks from both SQLite and FAISS
```

#### Document Indexing with Error Isolation

```python
def _index_document_file(self, file_path: str) -> Dict[str, Any]:
    """Index a single document file with error isolation."""
    rel_path = os.path.relpath(file_path, self.project_path)

    try:
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            markdown_text = asyncio.run(extract_pdf(file_path))
            chunks = chunk_markdown(markdown_text)
        elif ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            chunks = chunk_markdown(markdown_text)
        elif ext in ['.yaml', '.yml']:
            chunks = chunk_yaml(file_path)
        elif ext == '.json':
            chunks = chunk_json(file_path)
        else:
            return {'status': 'skipped', 'reason': f'unknown document type: {ext}'}

        # Embed and store chunks...
        return {
            'status': 'indexed',
            'chunks': len(chunks),
            'embedded': len(embeddings) if embeddings else 0
        }

    except DocumentExtractionError as e:
        logger.error(f"Document extraction failed for {rel_path}: {e}")
        self.project_db.update_file_status(file_id, 'failed')
        return {'status': 'failed', 'reason': f'extraction error: {e}'}
    except Exception as e:
        logger.error(f"Unexpected error indexing document {rel_path}: {e}")
        self.project_db.update_file_status(file_id, 'failed')
        return {'status': 'failed', 'reason': f'unexpected error: {e}'}
```

### 6. Search Pipeline Updates (Revised in v2 — over-fetch strategy)

#### Unified Search with Source-Type Filtering

```python
def search(
    query: str,
    limit: int = 10,
    source_type: list[str] = None,
    scope: str = "project",
    semantic: bool = True,
    keyword_only: bool = False
) -> list[dict]:
    """
    Unified search across code and documents.

    Returns list of dicts with:
    - source_type: 'code', 'markdown', 'pdf', 'yaml', 'json'
    - trusted: bool (True for code, False for documents)
    - rrf_score, file_path, content, section_heading, key_path, page_number, etc.
    """
```

#### Semantic Search with Over-Fetch + Post-Filter

```python
def semantic_search(
    query_embedding: np.ndarray,
    limit: int = 10,
    scope: str = "project",
    source_type: list[str] = None
) -> list[dict]:
    """
    FAISS vector search with optional source_type post-filtering.

    When source_type filter is active:
    - Fetch limit * 3 from FAISS (over-fetch multiplier)
    - Post-filter results by source_type using chunk_meta lookup
    - Return up to limit results (may be fewer if filtering reduces results)
    """
    fetch_limit = limit * 3 if source_type else limit

    chunk_ids, scores = cosine_search(
        query_embedding,
        fetch_limit,
        scope=scope
    )

    # Post-filter by source_type if specified
    if source_type:
        filtered = []
        for chunk_id, score in zip(chunk_ids, scores):
            chunk_meta = db.get_chunk_meta(chunk_id)
            if chunk_meta.get('source_type') in source_type:
                filtered.append({'id': chunk_id, 'score': score})

        return filtered[:limit]
    else:
        return [{'id': cid, 'score': s} for cid, s in zip(chunk_ids, scores)]
```

#### Document Search Tool

```python
async def doc_search(
    query: str,
    limit: int = 10,
    formats: list[str] = None,
    scope: str = "project"
) -> list[dict]:
    """
    Search non-code documents only.

    Convenience wrapper for search(query, source_type=['markdown', 'pdf', 'yaml', 'json']).
    """
    if formats is None:
        formats = ['markdown', 'pdf', 'yaml', 'json']

    return search(query, limit=limit, source_type=formats, scope=scope)
```

### 7. MCP Server Updates (Revised in v2 — drift_train tool)

```python
async def doc_search(
    query: str,
    limit: int = 10,
    formats: Optional[list[str]] = None,
    scope: str = "project"
) -> list[dict]:
    """
    Search non-code documents only (markdown, PDF, YAML, JSON).

    Args:
        query: Search query string
        limit: Max results to return
        formats: Document formats to search ['markdown', 'pdf', 'yaml', 'json']
        scope: 'project', 'collection', or 'global'

    Returns:
        List of document chunks with source_type, section_heading, key_path, etc.
    """

async def drift_train(sample_size: int = 50) -> dict:
    """
    Train Drift-Adapter on a sample of indexed chunks.

    For infrastructure operators during embedding model migrations.

    Implementation:
    1. Sample N random chunks from existing index
    2. Get old embeddings from FAISS
    3. Embed same chunks with new model (from embedding endpoint)
    4. Train Procrustes on (old, new) pairs
    5. Compute recall@10 on validation set (10% of sample)
    6. Save rotation matrix to .tessera/data/{slug}/drift_matrix.npy

    Args:
        sample_size: Number of chunks to sample (default 50)

    Returns:
        {
            "recall_at_10": 0.97,
            "matrix_path": "/path/to/drift_matrix.npy",
            "sample_size": 50,
            "validation_set_size": 5
        }

    Raises:
        ValueError: If insufficient chunks indexed or embedding endpoint unavailable
        DriftTrainingError: If training fails
    """
```

---

## Acceptance Criteria

### Document Indexing

- **AC-D1**: Extract 50-page PDF in <30 seconds (PyMuPDF4LLM + chunking + embedding)
- **AC-D2**: Markdown chunks preserve section hierarchy (headers dict in chunk_meta)
- **AC-D3**: YAML/JSON chunks include key_path and parent_section in chunk_meta
- **AC-D4**: All document extraction errors caught, logged, and marked as 'failed' in files table
- **AC-D5**: Document chunks indexed to same FAISS index as code chunks

### Drift-Adapter

- **AC-DA1**: Procrustes training on 50-sample corpus completes in <2 minutes
- **AC-DA2**: Rotation matrix saved as .npy file (float32, shape d×d)
- **AC-DA3**: `adapt_query()` adds <10 microsecond latency per query
- **AC-DA4**: Recall recovery 95%+ on validation set (EMNLP 2025 baseline)
- **AC-DA5**: `drift_train()` MCP tool returns recall@10 + matrix path
- **AC-DA6**: Server loads and applies drift matrix on startup if present

### Ignore Config

- **AC-I1**: `.tesseraignore` with `.gitignore` syntax correctly merged with defaults
- **AC-I2**: Security patterns (`.env*`, `*.pem`, `*credentials*`, etc.) are un-negatable
- **AC-I3**: User attempts to negate security patterns logged as warning
- **AC-I4**: Patterns applied during both full index and incremental re-index

### Search

- **AC-S1**: `search(query, source_type=['markdown'])` returns document-only results
- **AC-S2**: When source_type filter active, over-fetch 3x limit from FAISS, post-filter by source_type
- **AC-S3**: `doc_search()` convenience tool returns document results only
- **AC-S4**: RRF merge handles code + document results with proper ranking
- **AC-S5**: Search results include `source_type` and `trusted` fields

### Schema & Migration

- **AC-SM1**: Phase 4 database opens and auto-migrates Phase 3 databases
- **AC-SM2**: Migration idempotent (columns already exist case handled)
- **AC-SM3**: Existing code chunks have source_type='code' (default)
- **AC-SM4**: New document chunks have source_type set correctly

### General

- **AC-G1**: Total Phase 4 code <1,500 LOC (production only, excluding tests)
- **AC-G2**: All SQL queries parameterized (no injection risk)
- **AC-G3**: All new functions have type hints and docstrings
- **AC-G4**: Document + code search federate across collections

---

## Risks & Mitigations (Revised in v2 — security additions)

### Security Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **Credential file indexing** | Medium | Critical | Two-tier IgnoreFilter: SECURITY_PATTERNS (`.env*`, `*.pem`, `*.key`, `*credentials*`, `id_rsa*`, `*.token`, `service-account.json`) un-negatable. Log warning if user tries to override. |
| **Prompt injection via documents** | Low | Medium | Add `trusted` field to search results: `true` for code, `false` for documents. Agents must treat document content as external/untrusted. |
| **Array validation in DriftAdapter** | Low | High | Validate loaded array shape (must be square d×d) and dtype (float32/float64 only). Raise ValueError on mismatch. Use allow_pickle=False. |
| **YAML code execution** | Low | Critical | Using `yaml.safe_load()` (no `Loader=yaml.UnsafeLoader`). Explicitly documented. |

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **PyMuPDF4LLM quality varies** | Medium | High | Test on representative PDFs. Document supported types. Fallback to base PyMuPDF. |
| **Procrustes insufficient for drift** | Low | High | Monitor recall@10 during training. If <90%, escalate to Phase 5 alternatives. |
| **YAML/JSON edge cases** | Medium | Medium | Use yaml.safe_load() + json.loads(). Test on real config files. Document limitations. |
| **Ignore patterns exclude legitimate files** | Low | High | Conservative defaults. Users override via .tesseraignore. Document format. |
| **Over-fetch multiplier too aggressive** | Low | Medium | Monitor FAISS memory during search. Configurable in Phase 5 if needed. |
| **Schema migration fails** | Low | High | Wrap ALTER in try/except. Log results. Test on Phase 3 databases. |
| **Document extraction blocks event loop** | Medium | Medium | Use asyncio.to_thread() for I/O. Test with large PDFs. |

---

## Dependencies

### Libraries

- **pymupdf4llm** >=0.2.0 (PDF extraction)
- **pathspec** >=0.12.0 (gitignore patterns)
- **scipy** >=1.10.0 (Procrustes)
- **numpy** >=1.24.0 (matrix ops)
- **faiss-cpu** >=1.7.0 (vector search, already in Phase 1)
- **sqlite3** (built-in)
- **pyyaml** (for YAML parsing, check Phase 1 deps)

### Infrastructure

- Embedding endpoint (existing EmbeddingClient)
- `.tessera/` directory with `data/{slug}/` for drift matrices
- Git repository (optional; fallback to full index)

---

## Test Strategy

### Unit Tests (~150 LOC)

- **IgnoreFilter**: Security patterns, user overrides, negation warnings
- **DriftAdapter**: Procrustes training, matrix save/load with validation
- **Document extraction**: PDF/Markdown/YAML/JSON chunking; error handling
- **Schema migration**: schema_version checks, idempotent ALTER statements

### Integration Tests (~100 LOC)

- Full indexing pipeline: code + documents, verify chunk_meta fields
- Search: source_type filter (code-only, doc-only, mixed), over-fetch strategy
- Incremental re-index: document file change/deletion detection
- Drift-adapter e2e: train on sample, apply transformation, verify recall

### Performance Benchmarks (~50 LOC)

- PDF extraction: 50-page doc <30s
- Procrustes training: 50-sample <2min
- Semantic search with over-fetch: <50ms for limit=10, fetch=30

### Manual Testing

- Operator scenario: Create `.tesseraignore` with security pattern negation, verify warning
- Integration scenario: Index project with mixed code + docs, search for query returning both, verify RRF ranking
- Migration scenario: Phase 3 database → open with Phase 4 code, verify schema migration

---

## Non-Goals

Explicitly out of scope for Phase 4 (defer to Phase 5+):

1. Document versioning — History of document changes. (Phase 5+)
2. Graph edges from documents to code — Link sections to symbols. (Phase 5+)
3. OCR for scanned PDFs — Images in PDFs skipped. (Explicitly out of scope.)
4. DOCX, Excel, PowerPoint support — Only PDF, Markdown, YAML, JSON. (Phase 5+)
5. Confluence, Notion, wiki integration — External systems need connectors. (Phase 5+)
6. Semantic chunking for documents — Headers sufficient. Sentence-level deferred. (Phase 5+)
7. Low-Rank Affine or MLP drift adapters — Procrustes only. Complex drift Phase 5+. (Phase 5+)
8. Automated drift matrix training — Manual operator workflow. Auto-trigger Phase 5+. (Phase 5+)
9. `--dry-run` flag for .tesseraignore validation — Test patterns without indexing. (Future work)
10. Disk footprint projections and eviction policy — Acceptable for local tool. (Future work)

---

## LOC Budget (Revised in v2 — realistic breakdown)

**PRODUCTION CODE ONLY** (new + modified lines in `src/tessera/`):

| Module | LOC | Notes |
|--------|-----|-------|
| `document.py` (new) | 400 | 4 extractors + chunking + error handling |
| `drift_adapter.py` (new) | 200 | Procrustes + validation + persistence |
| `ignore.py` (new) | 120 | Two-tier system + pathspec integration |
| `db.py` (updates) | 100 | Schema version + migration logic |
| `indexer.py` (updates) | 300 | Document discovery + incremental re-index + error isolation |
| `search.py` (updates) | 150 | Over-fetch + post-filter + doc_search() |
| `server.py` (updates) | 80 | Register tools |
| **TOTAL PRODUCTION** | **1,350** | Within 1,500 LOC cap |

**TESTS** (separate budget):

| Category | LOC |
|----------|-----|
| Unit tests | 150 |
| Integration tests | 100 |
| Performance benchmarks | 50 |
| **TOTAL TESTS** | **300** |

---

## Implementation Checklist

### Phase 4 Module Structure

```
src/tessera/
  document.py              (400 LOC)
    ├─ extract_pdf()
    ├─ chunk_markdown()
    ├─ chunk_yaml()
    ├─ chunk_json()
    ├─ DocumentChunk dataclass
    └─ DocumentExtractionError

  drift_adapter.py        (200 LOC)
    ├─ DriftAdapter class
    ├─ train()
    ├─ adapt_query()
    ├─ save() / load() with validation
    └─ Array validation

  ignore.py               (120 LOC)
    ├─ IgnoreFilter class
    ├─ SECURITY_PATTERNS
    ├─ DEFAULT_PATTERNS
    ├─ load()
    ├─ should_ignore()
    └─ Warning logging

  db.py (updates)         (100 LOC)
    ├─ _meta table
    ├─ ALTER TABLE chunk_meta
    └─ _run_migrations()

  indexer.py (updates)    (300 LOC)
    ├─ _discover_files() with docs
    ├─ _get_changed_files() with docs
    ├─ _get_deleted_files() with docs
    ├─ _index_document_file() with error isolation
    └─ IgnoreFilter integration

  search.py (updates)     (150 LOC)
    ├─ search(..., source_type)
    ├─ semantic_search() with over-fetch
    ├─ keyword_search() with source_type
    └─ doc_search()

  server.py (updates)     (80 LOC)
    ├─ Register doc_search()
    ├─ Register drift_train()
    └─ Drift matrix loading

  tests/                  (300 LOC)
    ├─ test_ignore.py
    ├─ test_drift_adapter.py
    ├─ test_document.py
    ├─ test_schema_migration.py
    ├─ test_search.py
    └─ test_incremental.py
```

---

## Sources

- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [Drift-Adapter EMNLP 2025](https://aclanthology.org/2025.emnlp-main.805/)
- [Orthogonal Procrustes (SciPy)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html)
- [Pathspec Library (PyPI)](https://pypi.org/project/pathspec/)
- [Git .gitignore Documentation](https://git-scm.com/docs/gitignore)
- [LangChain Markdown Header Splitter](https://python.langchain.com/v0.2/docs/how_to/markdown_header_metadata_splitter/)
- [OWASP: Deserialization of Untrusted Data](https://owasp.org/www-community/deserialization-of-untrusted-data)
