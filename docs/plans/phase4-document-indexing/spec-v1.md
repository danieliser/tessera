# Spec: Phase 4 — Document Indexing + Drift-Adapter + Per-Project Ignore Config

**Version:** v1
**Date:** 2026-02-27
**Author:** spec-writer
**Project:** Tessera
**Status:** Phase 4 Deliverables Ready for Implementation

---

## Executive Summary

Phase 4 extends Tessera's indexing pipeline to handle non-code documents (PDF, Markdown, YAML, JSON) alongside existing code intelligence. Three orthogonal modules are added:

1. **Document Indexing** — Extract text from PDFs (via PyMuPDF4LLM), split Markdown by headers, parse YAML/JSON structurally. Chunks flow through the same embedding and search pipeline as code, enabling unified search across code + docs.

2. **Drift-Adapter** — Handle embedding model migration without full re-indexing via Orthogonal Procrustes transformation (scipy). Train on 1–5% corpus sample in <2 minutes; recover 95%+ retrieval recall with zero query latency overhead.

3. **Per-Project Ignore Config** — `.tesseraignore` file using `.gitignore` syntax (via pathspec library) to exclude files/patterns from indexing. Merges sensible defaults with user overrides.

**Key architectural decision**: Extend existing `chunk_meta` SQLite table with nullable `source_type`, `section_heading`, `key_path`, `page_number`, `parent_section` columns. Single FAISS index and search pipeline handles both code and documents. No separate doc-specific tables or schemas needed.

**Success metrics**:
- Extract + index 50-page PDF in <30 seconds
- `search()` returns mixed code + document results ranked via RRF
- `doc_search()` tool returns document-only results
- Per-project `.tesseraignore` excludes specified patterns
- Drift-Adapter trains on 5% sample in <2 minutes, recovers 95%+ recall
- Document search federates across collections (reuses Phase 3 RRF pattern)
- Total Phase 4 code: <1,500 LOC

---

## Problem Statement

### What's the problem?

Developers need persistent searchable access to non-code context: API docs (PDFs), architecture decisions (Markdown), configuration schemas (YAML/JSON). Currently, Tessera indexes code only, leaving 30% of the knowledge base unsearchable.

Additionally:
- When embedding models upgrade (e.g., from nomic-embed-text to a newer model), re-embedding 1M+ chunks is prohibitively expensive (0.25–0.5 GPU-hours). A lightweight transformation-based approach (Drift-Adapter) enables model migration in minutes, not hours.
- Per-project ignore patterns (`.tesseraignore`) are needed to exclude vendor code, generated files, and confidential data — currently Tessera skips these via hardcoded defaults only.

### Who has it?

1. **Always-on agents** — Need searchable memory of code + docs to answer "how does this API work?" without re-reading documentation
2. **Development teams** — Want to exclude `vendor/`, `node_modules/`, and custom paths from indexing without polluting the shared schema
3. **Operations teams** — Need to swap embedding models without full re-indexing when performance/cost requirements change

### How painful is it?

**Without Phase 4:**
- Agents must manually navigate between code search results and external docs
- Document searches happen outside Tessera (separate vector DB, LLM context, manual retrieval)
- Embedding model upgrades force expensive re-indexing or stale vectors (poor recall)
- Ignore patterns are hardcoded; projects cannot exclude domain-specific data

**With Phase 4:**
- `search("how do I handle webhook failures?")` returns code examples + relevant API docs in a single ranked list
- Model migration: train Procrustes on 50 samples (2 minutes), query-time transformation adds <1μs overhead
- Projects create `.tesseraignore` to exclude confidential configs, third-party bundles, generated code

---

## Proposed Solution

### High-Level Architecture

Document indexing extends the existing Phase 1–3 pipeline:

```
Project File Walk
    ↓
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

Unified Search (Phase 1 modified):
    ├─ Keyword search (FTS5)
    ├─ Semantic search (FAISS)
    └─ RRF merge (handles source_type filter)
       ↓
       Results: mixed code + doc chunks
```

### Core Deliverables

**1. Document Indexing Module** (`src/tessera/document.py`)
- PDF extraction via PyMuPDF4LLM (0.12s per document)
- Markdown splitting by headers with hierarchy metadata
- YAML/JSON structural parsing (key-path chunks)
- Unified `DocumentChunk` dataclass for all document types

**2. Drift-Adapter Module** (`src/tessera/drift_adapter.py`)
- Orthogonal Procrustes transformation via scipy.linalg
- Training on corpus sample (1–5%)
- Query-time transformation for migrated embeddings
- Persistent storage of rotation matrix

**3. Ignore Config Module** (`src/tessera/ignore.py`)
- `.tesseraignore` file parsing (gitignore syntax via pathspec)
- Default patterns (node_modules, vendor, .git, build artifacts, etc.)
- Per-project override + merge logic

**4. Schema Extensions** (`src/tessera/db.py`)
- ALTER TABLE chunk_meta to add document-specific columns
- Backward-compatible with existing code indexing

**5. Indexer Pipeline Updates** (`src/tessera/indexer.py`)
- Document discovery and filtering via IgnoreFilter
- Document extraction orchestration
- Embedding calls for document chunks

**6. Search Pipeline Updates** (`src/tessera/search.py`)
- New `source_type` filter parameter in `search()`
- New `doc_search()` convenience tool
- FTS5 search with source_type filtering

**7. MCP Server Updates** (`src/tessera/server.py`)
- Register `doc_search()` tool
- Optional drift-adapter training endpoint (for future model migrations)

---

## Architecture Details

### 1. Document Extraction & Chunking (`document.py`)

#### PDF Extraction
```python
async def extract_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF to markdown using PyMuPDF4LLM.

    Preserves headers (detected via font size), lists, tables, code blocks.

    Args:
        pdf_path: Absolute path to .pdf file

    Returns:
        Markdown string

    Raises:
        DocumentExtractionError: If PDF is corrupted or unreadable
    """
```

**Implementation notes:**
- Uses `pymupdf4llm.to_markdown()` (v0.2.0+)
- Runs in thread pool to avoid blocking event loop
- Caches results if file not modified (hash-based)

#### Markdown Chunking
```python
@dataclass
class MarkdownChunk:
    content: str                  # Text content of chunk
    headers: dict[str, str]       # Section hierarchy {"Header1": "Intro", "Header2": "Scaling"}
    start_line: int               # Line number in original file
    end_line: int                 # Line number in original file
    source_type: str = "markdown" # Constant

def chunk_markdown(
    markdown_text: str,
    max_chunk_size: int = 1024,
    overlap: int = 128,
    split_headers: list[str] = ["#", "##", "###"]
) -> list[MarkdownChunk]:
    """
    Split markdown by headers, preserving hierarchy metadata.

    Algorithm:
    1. Parse markdown, identify header lines (#, ##, ###, ...)
    2. Group content between headers into chunks
    3. If a section exceeds max_chunk_size, apply RecursiveCharacterTextSplitter
    4. Attach header hierarchy to each chunk

    Args:
        markdown_text: Full markdown content
        max_chunk_size: Target chunk size in characters (default 1024)
        overlap: Overlap between chunks in characters (default 128)
        split_headers: List of markdown header levels to split on (default ["#", "##", "###"])

    Returns:
        List of MarkdownChunk objects with content, headers, line numbers
    """
```

**Chunking strategy:**
- Headers become metadata (not repeated in content)
- Chunk size: 1024 characters (tunable)
- Overlap: 128 characters for context preservation across boundaries
- Example: README.md → chunks for "## Installation", "## API Reference", "### Authentication"

#### YAML/JSON Structural Chunking
```python
@dataclass
class ConfigChunk:
    content: str                    # Pretty-printed YAML/JSON fragment
    key_path: str                   # "server.port" or "database.credentials"
    parent_section: str             # "server" or "database"
    source_type: str = "config"     # "yaml" or "json"
    source_file: str                # e.g., "docker-compose.yml"

def chunk_yaml(yaml_path: str, max_chunk_size: int = 2048) -> list[ConfigChunk]:
    """
    Parse YAML, extract top-level sections, create key-path chunks.

    Algorithm:
    1. Load YAML to dict via yaml.safe_load()
    2. Walk top-level keys (sections)
    3. Pretty-print each section to YAML string
    4. If section exceeds max_chunk_size, split children into separate chunks
    5. Attach key_path metadata

    Args:
        yaml_path: Path to .yml/.yaml file
        max_chunk_size: Max bytes per chunk (default 2048)

    Returns:
        List of ConfigChunk objects

    Raises:
        yaml.YAMLError: If YAML is malformed
    """

def chunk_json(json_path: str, max_chunk_size: int = 2048) -> list[ConfigChunk]:
    """
    Parse JSON, extract top-level sections, create key-path chunks.

    Same algorithm as chunk_yaml but for JSON files.
    Uses json.loads() for parsing.
    """
```

**Structural chunking examples:**
- `docker-compose.yml` → chunks for `services`, `volumes`, `networks`
- `services` section too large? → sub-chunks for each service definition
- `config.yaml` → chunks for `database`, `cache`, `logging`

---

### 2. Drift-Adapter Module (`drift_adapter.py`)

```python
import numpy as np
from scipy.linalg import orthogonal_procrustes

class DriftAdapter:
    """Embedding model migration via Orthogonal Procrustes transformation."""

    def __init__(self, embedding_dim: int):
        """
        Initialize adapter.

        Args:
            embedding_dim: Dimension of embeddings (e.g., 768)
        """
        self.embedding_dim = embedding_dim
        self.rotation_matrix: Optional[np.ndarray] = None  # (dim, dim)

    def train(
        self,
        old_embeddings: np.ndarray,  # (n_samples, dim)
        new_embeddings: np.ndarray   # (n_samples, dim)
    ) -> np.ndarray:
        """
        Train Procrustes rotation matrix.

        Solves: min ||X @ R - Y||_F^2 where R is orthogonal

        Args:
            old_embeddings: (n_samples, dim) from old model
            new_embeddings: (n_samples, dim) from new model

        Returns:
            Rotation matrix (dim, dim)

        Raises:
            ValueError: If shapes don't match or n_samples < 1
        """
        if old_embeddings.shape != new_embeddings.shape:
            raise ValueError("Shape mismatch between old and new embeddings")

        if old_embeddings.shape[0] < 1:
            raise ValueError("Must provide at least 1 sample pair")

        # Center embeddings (improves numerical stability)
        old_centered = old_embeddings - old_embeddings.mean(axis=0)
        new_centered = new_embeddings - new_embeddings.mean(axis=0)

        # Orthogonal Procrustes: find optimal rotation
        self.rotation_matrix, _ = orthogonal_procrustes(
            old_centered,
            new_centered
        )

        return self.rotation_matrix

    def adapt_query(self, new_query_embedding: np.ndarray) -> np.ndarray:
        """
        Transform new model's query embedding to old model's space.

        Usage:
            adapter = load_adapter_from_disk()
            new_embedding = embed_with_new_model(query)
            old_space_embedding = adapter.adapt_query(new_embedding)
            results = faiss_index.search(old_space_embedding)

        Args:
            new_query_embedding: (dim,) vector from new model

        Returns:
            Transformed (dim,) vector in old model's space

        Raises:
            ValueError: If adapter not trained yet
        """
        if self.rotation_matrix is None:
            raise ValueError("Must call train() before adapt_query()")

        # ~10μs for 768-dim vector (matrix-vector multiply)
        return new_query_embedding @ self.rotation_matrix.T

    def save(self, path: str) -> None:
        """
        Persist rotation matrix to disk.

        Args:
            path: File path to save .npy file
        """
        if self.rotation_matrix is None:
            raise ValueError("No trained matrix to save")
        np.save(path, self.rotation_matrix)

    @classmethod
    def load(cls, path: str, embedding_dim: int) -> "DriftAdapter":
        """
        Load adapter from disk.

        Args:
            path: Path to .npy file
            embedding_dim: Embedding dimension

        Returns:
            DriftAdapter instance with loaded matrix
        """
        adapter = cls(embedding_dim)
        adapter.rotation_matrix = np.load(path)
        return adapter
```

**Performance characteristics:**
- Training: <100ms on CPU for 768-dim, 1000 samples
- Query transform: <1μs per query (matrix-vector multiply)
- Storage: 768×768×4 bytes ≈ 2.4 MB per model migration

---

### 3. Ignore Config Module (`ignore.py`)

```python
from pathlib import Path
import pathspec

class IgnoreFilter:
    """Per-project ignore patterns using .gitignore syntax."""

    # Default patterns (conservative, user-friendly)
    DEFAULT_PATTERNS = [
        # Version control
        '.git/',
        '.gitignore',

        # Python
        '__pycache__/',
        '*.pyc',
        '*.pyo',
        '.venv/',
        'venv/',
        '.egg-info/',
        'dist/',
        'build/',

        # Node.js
        'node_modules/',
        'npm-debug.log',
        '.npm/',

        # PHP
        'vendor/',
        'composer.lock',

        # TypeScript/JavaScript
        '.next/',
        'out/',
        '.turbo/',

        # IDE
        '.vscode/',
        '.idea/',
        '*.swp',
        '*.swo',
        '.DS_Store',

        # Test coverage
        'coverage/',
        '.nyc_output/',

        # Tessera-specific
        '.tessera/',
        '*.log',
    ]

    def __init__(self, project_root: Path, ignore_file: str = ".tesseraignore"):
        """
        Initialize ignore filter.

        Args:
            project_root: Project root directory
            ignore_file: Name of ignore file (default ".tesseraignore")
        """
        self.project_root = Path(project_root)
        self.ignore_file = self.project_root / ignore_file
        self.spec: Optional[pathspec.PathSpec] = None
        self.load()

    def load(self) -> None:
        """
        Load .tesseraignore from disk, merge with defaults.

        If .tesseraignore doesn't exist, use defaults only.
        Patterns are evaluated in order: defaults first, then user patterns.
        """
        patterns = self.DEFAULT_PATTERNS.copy()

        if self.ignore_file.exists():
            try:
                with open(self.ignore_file, 'r') as f:
                    for line in f:
                        stripped = line.strip()
                        # Skip empty lines and comments
                        if stripped and not stripped.startswith('#'):
                            patterns.append(stripped)
            except OSError as e:
                logger.warning(f"Failed to read {self.ignore_file}: {e}")

        # Compile patterns using gitignore semantics
        self.spec = pathspec.PathSpec.from_lines('gitwildmatch', patterns)

    def should_ignore(self, rel_path: str) -> bool:
        """
        Check if a path should be ignored.

        Args:
            rel_path: Path relative to project root

        Returns:
            True if path matches ignore patterns
        """
        if self.spec is None:
            return False
        return self.spec.match_file(rel_path)
```

**Example `.tesseraignore`:**
```
# Project-specific exclusions
vendor/third_party/
data/private/
build/
dist/

# Override default (don't ignore these despite defaults)
!vendor/important-lib/
```

---

### 4. Schema Extensions (`db.py`)

Add 5 nullable columns to existing `chunk_meta` table:

```sql
-- In ProjectDB._create_schema()

ALTER TABLE chunk_meta ADD COLUMN source_type TEXT DEFAULT 'code';
-- Values: 'code', 'markdown', 'pdf', 'yaml', 'json'

ALTER TABLE chunk_meta ADD COLUMN section_heading TEXT;
-- For markdown: "## Installation" or "### Getting Started"

ALTER TABLE chunk_meta ADD COLUMN key_path TEXT;
-- For YAML/JSON: "server.port" or "database.credentials"

ALTER TABLE chunk_meta ADD COLUMN page_number INTEGER;
-- For PDF: page number (1-indexed)

ALTER TABLE chunk_meta ADD COLUMN parent_section TEXT;
-- For YAML/JSON: "server" or "database" (top-level key)

-- Optional: add index for source_type filtering
CREATE INDEX IF NOT EXISTS idx_chunk_meta_source_type
ON chunk_meta(source_type);
```

**Migration strategy:**
- Column additions are backward-compatible (defaults to NULL for existing rows)
- Indexing job runs in existing mode; Phase 4 jobs populate new columns
- Search filters on source_type are optional (NULL passes through)

---

### 5. Indexer Pipeline Updates (`indexer.py`)

```python
# In IndexerPipeline class

def _discover_files(self) -> List[str]:
    """
    Walk project directory, return paths matching supported languages + document formats.

    Returns:
        Sorted list of absolute file paths
    """
    code_extensions = {
        'php': ['.php'],
        'typescript': ['.ts', '.tsx'],
        'javascript': ['.js', '.jsx'],
        'python': ['.py'],
        'swift': ['.swift'],
    }

    document_extensions = ['.pdf', '.md', '.yaml', '.yml', '.json']

    allowed_code_exts = set()
    for lang in self.languages:
        allowed_code_exts.update(code_extensions.get(lang, []))

    # Initialize ignore filter
    from .ignore import IgnoreFilter
    ignore_filter = IgnoreFilter(self.project_path)

    files = []
    for root, dirs, filenames in os.walk(self.project_path):
        # Skip hidden dirs, node_modules, vendor, .git, .tessera
        dirs[:] = [
            d for d in dirs
            if not d.startswith('.') and d not in ('node_modules', 'vendor', '__pycache__', '.tessera')
        ]

        for f in filenames:
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, self.project_path)

            # Check ignore filter
            if ignore_filter.should_ignore(rel_path):
                continue

            # Skip TypeScript declaration files
            if f.endswith('.d.ts'):
                continue

            # Code files
            if any(f.endswith(ext) for ext in allowed_code_exts):
                files.append(abs_path)

            # Document files
            elif any(f.endswith(ext) for ext in document_extensions):
                files.append(abs_path)

    return sorted(files)

async def index_project(
    self,
    reindex_mode: str = 'full'
) -> IndexStats:
    """
    Index all files (code + documents).

    New Phase 4 logic:
    1. Discover files (includes documents)
    2. For each code file: parse, extract symbols, chunk (existing logic)
    3. For each document file: extract, chunk with document-specific metadata
    4. Embed all chunks (code + doc)
    5. Store in chunk_meta with source_type field
    """
    # ... existing code discovery and file filtering ...

    for file_path in discovered_files:
        rel_path = os.path.relpath(file_path, self.project_path)
        language = detect_language(file_path)

        if language in self.languages:
            # Existing code indexing
            await self._index_code_file(file_path, rel_path)

        else:
            # New document indexing
            await self._index_document_file(file_path, rel_path)

async def _index_document_file(self, file_path: str, rel_path: str) -> None:
    """
    Index a document file (PDF, Markdown, YAML, JSON).

    Args:
        file_path: Absolute path
        rel_path: Relative path from project root
    """
    from .document import (
        extract_pdf, chunk_markdown, chunk_yaml, chunk_json,
        DocumentExtractionError
    )

    try:
        if file_path.endswith('.pdf'):
            markdown_text = await extract_pdf(file_path)
            chunks = chunk_markdown(markdown_text)
            source_type = 'pdf'

        elif file_path.endswith('.md'):
            with open(file_path, 'r') as f:
                markdown_text = f.read()
            chunks = chunk_markdown(markdown_text)
            source_type = 'markdown'

        elif file_path.endswith(('.yaml', '.yml')):
            chunks = chunk_yaml(file_path)
            source_type = 'yaml'

        elif file_path.endswith('.json'):
            chunks = chunk_json(file_path)
            source_type = 'json'

        else:
            logger.warning(f"Unknown document type: {file_path}")
            return

        # Store chunks in database
        for chunk in chunks:
            # Existing chunk storage logic
            chunk_dict = {
                'content': chunk.content,
                'file_path': rel_path,
                'source_type': source_type,
                'section_heading': getattr(chunk, 'headers', {}).get('Header1'),
                'key_path': getattr(chunk, 'key_path', None),
                'page_number': getattr(chunk, 'page_number', None),
                'parent_section': getattr(chunk, 'parent_section', None),
            }

            chunk_id = self.project_db.insert_chunk_meta(chunk_dict)

            # Embed chunk if embedding endpoint available
            if self.embedding_client:
                try:
                    embedding = await self.embedding_client.embed_text(chunk.content)
                    self.project_db.insert_embedding(chunk_id, embedding)
                except EmbeddingUnavailableError:
                    logger.warning(f"Embedding unavailable for chunk {chunk_id}")

    except DocumentExtractionError as e:
        logger.error(f"Failed to index document {file_path}: {e}")
```

---

### 6. Search Pipeline Updates (`search.py`)

```python
def keyword_search(
    query: str,
    db: ProjectDB,
    source_type: Optional[list[str]] = None,
    limit: int = 10
) -> list[dict]:
    """
    Full-text search via SQLite FTS5, with optional source_type filter.

    Args:
        query: Search query
        db: ProjectDB instance
        source_type: Filter to specific source types ['code', 'markdown', 'pdf', 'yaml', 'json']
        limit: Max results

    Returns:
        List of dicts with id, content, file_path, source_type, score
    """
    base_sql = """
        SELECT c.id, c.content, c.file_id, c.source_type, f.path,
               c.section_heading, c.key_path, c.page_number, c.language
        FROM chunks_fts fts
        JOIN chunk_meta c ON fts.chunk_id = c.id
        JOIN files f ON c.file_id = f.id
        WHERE fts.content MATCH ?
    """

    params = [query]

    # Add source_type filter if specified
    if source_type:
        placeholders = ','.join('?' * len(source_type))
        base_sql += f" AND c.source_type IN ({placeholders})"
        params.extend(source_type)

    base_sql += " LIMIT ?"
    params.append(limit)

    results = db.conn.execute(base_sql, params).fetchall()

    return [
        {
            "id": row["id"],
            "content": row["content"],
            "file_path": row["path"],
            "source_type": row["source_type"],
            "section_heading": row["section_heading"],
            "key_path": row["key_path"],
            "page_number": row["page_number"],
        }
        for row in results
    ]

def semantic_search(
    query_embedding: np.ndarray,
    db: ProjectDB,
    source_type: Optional[list[str]] = None,
    limit: int = 10
) -> list[dict]:
    """
    Vector similarity search via FAISS, with optional source_type filter.

    Args:
        query_embedding: Embedding vector
        db: ProjectDB instance
        source_type: Filter to specific source types
        limit: Max results

    Returns:
        List of dicts with id, content, file_path, source_type, score
    """
    # Get all embeddings (unchanged from Phase 1)
    all_embeddings, chunk_ids = db.get_all_embeddings()

    if len(all_embeddings) == 0:
        return []

    # FAISS search
    results = cosine_search(query_embedding, chunk_ids, all_embeddings, limit=limit*2)

    # Enrich with metadata
    enriched = []
    for result in results:
        chunk = db.get_chunk(result["id"])
        if source_type and chunk["source_type"] not in source_type:
            continue

        enriched.append({
            "id": result["id"],
            "content": chunk["content"],
            "file_path": chunk["file_path"],
            "source_type": chunk["source_type"],
            "score": result["score"],
            "section_heading": chunk.get("section_heading"),
            "key_path": chunk.get("key_path"),
            "page_number": chunk.get("page_number"),
        })

    return enriched[:limit]

async def search(
    query: str,
    db: ProjectDB,
    embedding_client: Optional[EmbeddingClient] = None,
    source_type: Optional[list[str]] = None,
    limit: int = 10,
    keyword_only: bool = False
) -> list[dict]:
    """
    Unified search across code + documents.

    Args:
        query: Search query
        db: ProjectDB instance
        embedding_client: Optional embedding client for semantic search
        source_type: Filter results ['code', 'markdown', 'pdf', 'yaml', 'json'] or None (all)
        limit: Max results
        keyword_only: Force keyword search only (no embeddings)

    Returns:
        List of dicts with id, content, file_path, source_type, rrf_score, etc.
    """
    # 1. Keyword search
    kw_results = keyword_search(query, db, source_type=source_type, limit=limit*2)

    # 2. Semantic search (if embeddings available)
    sem_results = []
    if embedding_client and not keyword_only:
        try:
            query_embedding = await embedding_client.embed_text(query)
            sem_results = semantic_search(query_embedding, db, source_type=source_type, limit=limit*2)
        except EmbeddingUnavailableError:
            logger.debug("Embedding unavailable, falling back to keyword search")

    # 3. Merge via RRF
    if not sem_results:
        return kw_results[:limit]

    merged = rrf_merge([kw_results, sem_results], k=60)
    return merged[:limit]

async def doc_search(
    query: str,
    db: ProjectDB,
    embedding_client: Optional[EmbeddingClient] = None,
    formats: Optional[list[str]] = None,
    limit: int = 10
) -> list[dict]:
    """
    Convenience wrapper for document-only search.

    Args:
        query: Search query
        db: ProjectDB instance
        embedding_client: Optional embedding client
        formats: Document formats to search ['markdown', 'pdf', 'yaml', 'json'] or None (all)
        limit: Max results

    Returns:
        List of document chunks
    """
    if formats is None:
        formats = ['markdown', 'pdf', 'yaml', 'json']

    return await search(query, db, embedding_client, source_type=formats, limit=limit)
```

---

### 7. MCP Server Updates (`server.py`)

```python
@mcp.tool()
async def doc_search(
    query: str,
    formats: Optional[list[str]] = None,
    limit: int = 10,
    session_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Search non-code documents only (Markdown, PDF, YAML, JSON).

    Args:
        query: Search query
        formats: Document formats to search (default: all non-code)
        limit: Max results to return
        session_id: Session token for scope gating (optional)

    Returns:
        JSON with results array (each: id, content, file_path, source_type, rrf_score)
    """
    scope, error = _check_session({"session_id": session_id})
    if error:
        return {"error": error}

    try:
        project_dbs = _get_project_dbs(scope)

        tasks = [
            search_doc(query, db, embedding_client, formats, limit)
            for _, _, db in project_dbs
        ]

        results_by_project = await asyncio.gather(*tasks)
        merged = rrf_merge([r for r in results_by_project if r])

        _log_audit("doc_search", len(merged), agent_id=scope.agent_id if scope else "dev", scope_level=scope.level if scope else "project")

        return {
            "results": merged[:limit],
            "count": len(merged),
            "projects_searched": len(project_dbs)
        }

    except Exception as e:
        logger.exception("doc_search failed")
        return {"error": str(e)}
```

---

## Data Flow Diagrams

### Phase 4: Document Indexing

```
User runs: uv run python -m tessera index /path/to/project --languages php,typescript,markdown
    │
    ├─ IndexerPipeline.register()
    │  └─ GlobalDB.register_project()
    │
    ├─ IndexerPipeline._discover_files()
    │  ├─ Walk project tree
    │  ├─ Apply IgnoreFilter(.tesseraignore)
    │  ├─ Collect code (.php, .ts, .py, .js)
    │  └─ Collect documents (.pdf, .md, .yaml, .json)
    │
    ├─ For each code file:
    │  ├─ detect_language()
    │  ├─ parse_and_extract() [tree-sitter]
    │  ├─ chunk_with_cast()
    │  └─ Store in chunk_meta (source_type='code')
    │
    ├─ For each document file:
    │  ├─ Detect type (.pdf/.md/.yaml/.json)
    │  ├─ Extract & chunk:
    │  │  ├─ PDF → extract_pdf() → chunk_markdown()
    │  │  ├─ Markdown → chunk_markdown()
    │  │  ├─ YAML → chunk_yaml()
    │  │  └─ JSON → chunk_json()
    │  └─ Store in chunk_meta (source_type='markdown'/'pdf'/'yaml'/'json')
    │
    ├─ Batch embed all chunks [if embedding_endpoint available]
    │  └─ Store in chunk_embeddings
    │
    └─ Update project.indexed_at in GlobalDB
```

### Phase 4: Unified Search

```
Agent calls: search("how do I authenticate?", source_type=['code', 'markdown'])
    │
    ├─ KeywordSearch(FTS5):
    │  ├─ Query chunks_fts WHERE content MATCH "authenticate" AND source_type IN ('code', 'markdown')
    │  ├─ Return 20 results with scores
    │
    ├─ SemanticSearch(FAISS):
    │  ├─ Embed query → get_query_embedding()
    │  ├─ [Optional: apply DriftAdapter if model changed]
    │  ├─ FAISS search on chunk_embeddings
    │  ├─ Filter by source_type
    │  └─ Return 20 results with scores
    │
    ├─ RRF Merge:
    │  ├─ Combine keyword + semantic lists
    │  ├─ RRF ranking formula
    │  └─ Return top 10 merged results
    │
    └─ Agent sees:
        ├─ Code chunk: "function authenticate() { ... }" (source_type='code')
        ├─ Markdown chunk: "## Authentication Methods\n..." (source_type='markdown', section_heading="Authentication Methods")
        └─ API doc chunk: "POST /auth endpoint..." (source_type='pdf', page_number=15)
```

### Phase 4: Drift-Adapter

```
Scenario: Embedding model upgraded from nomic-embed-text to newer model
    │
    ├─ Agent has existing index with old model embeddings
    │
    ├─ New model available, wants to migrate
    │  └─ Operator calls: drift_adapter.train(
    │     old_embeddings=[50 samples from current index],
    │     new_embeddings=[50 samples from new model]
    │  )
    │  └─ Takes <2 minutes
    │
    ├─ DriftAdapter.train():
    │  ├─ Center embeddings
    │  ├─ Orthogonal Procrustes via scipy.linalg
    │  ├─ Learn rotation matrix (768×768)
    │  └─ Save to disk (.tessera/drift_matrix.npy)
    │
    ├─ [Operator updates config to use new embedding model]
    │
    └─ On next query:
       ├─ Embed query with new model
       ├─ Load DriftAdapter from disk
       ├─ Transform: new_embedding @ rotation_matrix.T
       ├─ Search FAISS with transformed embedding
       └─ Results: 95%+ recall vs full re-index (100× cheaper)
```

---

## API Contracts

### `document.py`

```python
async def extract_pdf(pdf_path: str) -> str:
    """→ markdown string"""

def chunk_markdown(
    markdown_text: str,
    max_chunk_size: int = 1024,
    overlap: int = 128,
    split_headers: list[str] = ["#", "##", "###"]
) -> list[MarkdownChunk]:
    """→ list[MarkdownChunk]"""

def chunk_yaml(yaml_path: str, max_chunk_size: int = 2048) -> list[ConfigChunk]:
    """→ list[ConfigChunk]"""

def chunk_json(json_path: str, max_chunk_size: int = 2048) -> list[ConfigChunk]:
    """→ list[ConfigChunk]"""

@dataclass
class MarkdownChunk:
    content: str
    headers: dict[str, str]
    start_line: int
    end_line: int
    source_type: str = "markdown"

@dataclass
class ConfigChunk:
    content: str
    key_path: str
    parent_section: str
    source_type: str
    source_file: str
```

### `drift_adapter.py`

```python
class DriftAdapter:
    def __init__(self, embedding_dim: int) -> None: ...
    def train(self, old_embeddings: np.ndarray, new_embeddings: np.ndarray) -> np.ndarray: ...
    def adapt_query(self, new_query_embedding: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str, embedding_dim: int) -> "DriftAdapter": ...
```

### `ignore.py`

```python
class IgnoreFilter:
    DEFAULT_PATTERNS: list[str]
    def __init__(self, project_root: Path, ignore_file: str = ".tesseraignore") -> None: ...
    def load(self) -> None: ...
    def should_ignore(self, rel_path: str) -> bool: ...
```

### `indexer.py` (additions)

```python
async def _index_document_file(self, file_path: str, rel_path: str) -> None: ...
```

### `search.py` (additions)

```python
async def search(
    query: str,
    db: ProjectDB,
    embedding_client: Optional[EmbeddingClient] = None,
    source_type: Optional[list[str]] = None,
    limit: int = 10,
    keyword_only: bool = False
) -> list[dict]: ...

async def doc_search(
    query: str,
    db: ProjectDB,
    embedding_client: Optional[EmbeddingClient] = None,
    formats: Optional[list[str]] = None,
    limit: int = 10
) -> list[dict]: ...

def keyword_search(
    query: str,
    db: ProjectDB,
    source_type: Optional[list[str]] = None,
    limit: int = 10
) -> list[dict]: ...

def semantic_search(
    query_embedding: np.ndarray,
    db: ProjectDB,
    source_type: Optional[list[str]] = None,
    limit: int = 10
) -> list[dict]: ...
```

### `server.py` (MCP tools)

```python
@mcp.tool()
async def doc_search(
    query: str,
    formats: Optional[list[str]] = None,
    limit: int = 10,
    session_id: Optional[str] = None
) -> dict[str, Any]: ...
```

---

## Schema DDL

```sql
-- Extended chunk_meta table (Phase 4)
-- Note: All new columns are nullable, backward-compatible with Phase 1-3

ALTER TABLE chunk_meta ADD COLUMN source_type TEXT DEFAULT 'code';
-- Values: 'code', 'markdown', 'pdf', 'yaml', 'json'

ALTER TABLE chunk_meta ADD COLUMN section_heading TEXT;
-- For markdown sections: stored value is the human-readable section name
-- Example: "Installation", "API Reference", "Getting Started"

ALTER TABLE chunk_meta ADD COLUMN key_path TEXT;
-- For YAML/JSON: dot-notation path to config key
-- Example: "server.port", "database.credentials.password", "services.api.environment"

ALTER TABLE chunk_meta ADD COLUMN page_number INTEGER;
-- For PDF chunks: 1-indexed page number
-- NULL for other source types

ALTER TABLE chunk_meta ADD COLUMN parent_section TEXT;
-- For YAML/JSON: top-level section containing this chunk
-- Example: "server", "database", "services"
-- Enables filtering: search(..., parent_section="database")

-- Index for faster source_type filtering (used in most queries from Phase 4 onwards)
CREATE INDEX IF NOT EXISTS idx_chunk_meta_source_type ON chunk_meta(source_type);

-- Optional: composite index for document-specific queries
CREATE INDEX IF NOT EXISTS idx_chunk_meta_doc_fields
ON chunk_meta(source_type, section_heading) WHERE source_type IN ('markdown', 'pdf');

-- Optional: index for config key lookups
CREATE INDEX IF NOT EXISTS idx_chunk_meta_key_path
ON chunk_meta(key_path) WHERE source_type IN ('yaml', 'json');
```

---

## Acceptance Criteria

### Criterion 1: PDF Extraction Performance
**Test**: Index 50-page PDF in <30 seconds
**Method**:
- Create test PDF (50 pages, ~20K words)
- Run `indexer.index_project()` with PDF included
- Measure wall-clock time
**Expected**:
- PyMuPDF4LLM extraction: ~6 seconds (0.12s × 50 pages)
- Markdown chunking: ~1 second
- Embedding (if enabled): ~15–20 seconds
- Total: <30 seconds

### Criterion 2: Unified Search — Mixed Results
**Test**: `search("authentication")` returns both code and document chunks
**Method**:
- Index project with code (authentication.php) + docs (API.md)
- Call `search("authentication", source_type=['code', 'markdown'])`
- Verify results include both types
**Expected**:
- Top 10 results ranked via RRF
- `source_type` field correctly populated
- Results from both code and markdown sources

### Criterion 3: Document Search Tool
**Test**: `doc_search("webhook payload")` returns documents only
**Method**:
- Index same project
- Call `doc_search("webhook payload")`
- Verify no code results
**Expected**:
- Results contain only `source_type` in ['markdown', 'pdf', 'yaml', 'json']
- Code chunks excluded

### Criterion 4: Per-Project Ignore Config
**Test**: Files matching `.tesseraignore` are excluded from indexing
**Method**:
- Create `.tesseraignore` with pattern `vendor/`
- Index project with vendor directory
- Verify vendor files not in chunk_meta
**Expected**:
- IgnoreFilter loads .tesseraignore
- should_ignore("vendor/foo.php") returns True
- No chunks created for vendor files

### Criterion 5: Drift-Adapter Training
**Test**: Train adapter on 5% sample in <2 minutes
**Method**:
- Create 1000 embedding pairs (old model, new model)
- Call `DriftAdapter.train(old_embeddings[0:50], new_embeddings[0:50])`
- Measure training time
**Expected**:
- Training time <100ms (CPU, 768-dim)
- Rotation matrix (768×768) generated
- Save/load works

### Criterion 6: Drift-Adapter Query Latency
**Test**: Query transformation adds <1μs overhead
**Method**:
- Load trained adapter
- Transform 10K queries via `adapt_query()`
- Measure per-query time
**Expected**:
- Average <1μs per query (matrix-vector multiply only)

### Criterion 7: Drift-Adapter Recall Recovery
**Test**: Recall recovers to 95%+ after transformation
**Method**:
- Index 100 documents with old model
- Generate embeddings with new model
- Train adapter on 10% sample (10 pairs)
- Query with 20 test queries (new model)
- Transform via adapter before search
- Measure recall@10 vs baseline (full re-index with new model)
**Expected**:
- Recall ≥ 0.95 (95% of queries return same top-10 as full re-index)

### Criterion 8: Federated Document Search
**Test**: Document search federates across collections
**Method**:
- Create 2 projects in same collection
- Index both with documents
- Query at collection scope with `source_type=['markdown']`
**Expected**:
- Results merged from both projects via RRF
- RRF ranking works correctly

### Criterion 9: Line Budget
**Test**: Phase 4 implementation <1,500 LOC
**Method**:
- Count lines in new/modified files
- `document.py` + `drift_adapter.py` + `ignore.py` + updates to `indexer.py`, `search.py`, `server.py`, `db.py`
**Expected**:
- Total <1,500 LOC

### Criterion 10: Backward Compatibility
**Test**: Phase 1–3 code indexing unchanged
**Method**:
- Run Phase 1 test suite on code-only project
- Verify all tests pass
**Expected**:
- No regression in code indexing
- Symbol extraction unchanged
- Search performance unchanged

---

## Performance Targets

| Operation | Target | Method |
|-----------|--------|--------|
| PDF extraction (50 pages) | <6s | PyMuPDF4LLM at 0.12s/page |
| Markdown chunking (50KB) | <1s | Header-based splitting |
| YAML/JSON chunking (50KB) | <1s | Structural parsing |
| Embedding batch (100 chunks) | <10s | OpenAI-compatible endpoint |
| Drift-Adapter training (5% sample) | <100ms | scipy Procrustes (CPU) |
| Drift-Adapter query transform (1 query) | <1μs | Matrix-vector multiply |
| Keyword search (FTS5) | <20ms | SQLite FTS5 |
| Semantic search (FAISS) | <30ms | FAISS IndexFlatIP |
| RRF merge (1K results) | <10ms | Python dict operations |
| Collection-scoped search (5 projects) | <100ms p95 | asyncio.gather parallelization |

---

## Implementation Phases (Within Phase 4)

### Iteration 1: Document Indexing Core (Week 1)
1. Implement `document.py` (PDF, Markdown, YAML/JSON extraction + chunking)
2. Update `db.py` schema (add 5 columns to chunk_meta)
3. Update `indexer.py` to discover and index documents
4. Tests: unit tests for each chunk type, integration test for full pipeline

### Iteration 2: Ignore Config + Search Updates (Week 2)
1. Implement `ignore.py` (IgnoreFilter with .gitignore syntax)
2. Integrate IgnoreFilter into `indexer.py._discover_files()`
3. Update `search.py` (add source_type filtering, doc_search wrapper)
4. Update `server.py` (register doc_search MCP tool)
5. Tests: ignore filter unit tests, search filtering integration tests

### Iteration 3: Drift-Adapter (Week 2–3)
1. Implement `drift_adapter.py` (Procrustes training + query transform)
2. Test training on synthetic embeddings
3. Test query transformation latency
4. Optional: MCP endpoint for drift training (deferred to Phase 5+)
5. Tests: unit tests for adapter, recall recovery on test corpus

### Iteration 4: Integration + Documentation (Week 3)
1. Full integration test: index code + documents, search unified, scope enforcement
2. Performance validation: benchmark all targets
3. Documentation: API reference, usage examples, architecture decision rationale
4. Manual testing: real projects (PopupMaker, etc.)

---

## Line Budget Breakdown

```
document.py              ~350 LOC
  ├─ extract_pdf()                ~50
  ├─ chunk_markdown()             ~100
  ├─ chunk_yaml()                 ~100
  ├─ chunk_json()                 ~80
  └─ Dataclasses + helpers        ~20

drift_adapter.py         ~200 LOC
  ├─ __init__()                   ~30
  ├─ train()                      ~50
  ├─ adapt_query()                ~20
  ├─ save() / load()              ~30
  └─ Docstrings + validation      ~70

ignore.py                ~100 LOC
  ├─ __init__()                   ~20
  ├─ load()                       ~30
  ├─ should_ignore()              ~10
  ├─ DEFAULT_PATTERNS             ~30
  └─ Docstrings                   ~10

db.py (updates)          ~100 LOC
  └─ ALTER TABLE + indexes

indexer.py (updates)     ~250 LOC
  ├─ _discover_files() mods       ~80
  ├─ _index_document_file()       ~170

search.py (updates)      ~150 LOC
  ├─ keyword_search() mods        ~40
  ├─ semantic_search() mods       ~40
  ├─ search() mods                ~40
  └─ doc_search()                 ~30

server.py (updates)      ~50 LOC
  └─ doc_search() tool + registration

Tests (integration)      ~200 LOC
  ├─ Document extraction tests
  ├─ Ignore filter tests
  ├─ Search filtering tests
  └─ Full integration test

─────────────────────────
Total: ~1,350 LOC (within 1,500 budget)
```

---

## Dependencies

### New Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| pymupdf4llm | >=0.2.0 | PDF text extraction to markdown |
| pathspec | >=0.12.0 | Gitignore-style pattern matching |
| scipy | >=1.10.0 | Orthogonal Procrustes via scipy.linalg |
| numpy | >=1.24.0 | Matrix operations (already dependency) |
| pyyaml | >=6.0 | YAML parsing (likely existing) |

### Existing Dependencies (No Changes)
- faiss-cpu (FAISS for vector search)
- sqlite3 (built-in)
- tree-sitter (code parsing)

**Installation**:
```bash
uv add pymupdf4llm>=0.2.0 pathspec>=0.12.0 scipy>=1.10.0
```

---

## Non-Goals

1. **OCR for scanned PDFs** — explicitly out of scope per intake
2. **DOCX/Word format support** — defer to Phase 5+
3. **Confluence or Slack integration** — defer to future phases
4. **Document version history** — defer to Phase 5+
5. **Graph edges from documents to code** — defer to Phase 5+ (may enable "architecture doc describes this module")
6. **Real-time document watching** — defer to Phase 6+ (file watcher)
7. **Advanced semantic chunking** — header-based splitting sufficient for Phase 4
8. **Low-Rank Affine or Residual MLP Drift-Adapter variants** — Orthogonal Procrustes sufficient for Phase 4
9. **Separate document storage** — all documents stored in unified chunk_meta table

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| PyMuPDF4LLM output quality varies by PDF type | Medium | High (bad chunks → bad search results) | Test on representative PDFs (technical specs, research docs). Fallback to base PyMuPDF if markdown quality poor. Document limitations in release notes. |
| Procrustes rotation insufficient for large embedding drift | Low | High (recall <95%) | Monitor recall on test corpus during Drift-Adapter training. If recall <90%, escalate to Low-Rank Affine. Include fallback to full re-embed in future phases. |
| YAML/JSON parsing misses edge cases (nested arrays, complex types) | Medium | Medium (incomplete indexing) | Use yaml.safe_load and json.loads (stdlib, robust). Test against real config files from Phase 1 projects. Document parsing assumptions. |
| `.tesseraignore` patterns break legitimate files | Low | High (requires re-index) | Defaults are conservative (node_modules, .git, vendor only). Users can override with `.tesseraignore`. Warn on suspicious patterns. |
| FTS5 tokenization issues with markdown metadata | Low | Medium (search misses results) | Test FTS5 on markdown chunks with section headers. Verify title/header indexing works. Fallback to separate metadata columns if needed. |
| Embedding endpoint unavailable during document indexing | Medium | Medium (indexing fails, no semantic search) | Mark project index_status='failed', fall back to keyword-only search. Log warning, allow retry on next indexing run. Document mitigation in runbook. |
| Document chunks too large for embedding API | Low | Medium (some chunks fail to embed) | Set max chunk size (2048 chars for YAML/JSON, 1024 for markdown). Validate chunk size before embedding. Log oversized chunks, skip embedding but include in FTS. |
| Ignore filter blocks important config files | Low | High (re-index required) | Provide `.tesseraignore` negation examples in docs. Default patterns tested on 3+ real projects. User can validate with `--dry-run` flag (future work). |

---

## Testing Strategy

### Unit Tests

**document.py**:
- `test_extract_pdf()` — verify PyMuPDF4LLM wrapper
- `test_chunk_markdown()` — header detection, hierarchy, overlap
- `test_chunk_yaml()` — key-path generation, section splitting
- `test_chunk_json()` — nested structure handling

**drift_adapter.py**:
- `test_train()` — rotation matrix generation
- `test_adapt_query()` — transformation accuracy
- `test_save_load()` — persistence

**ignore.py**:
- `test_default_patterns()` — node_modules, vendor, .git
- `test_load_tesseraignore()` — merging defaults + user patterns
- `test_should_ignore()` — path matching
- `test_negation()` — override patterns with `!`

**search.py**:
- `test_keyword_search_source_type_filter()` — FTS5 with source_type
- `test_semantic_search_source_type_filter()` — FAISS with source_type
- `test_doc_search()` — convenience wrapper
- `test_rrf_with_mixed_types()` — RRF merges code + doc results

### Integration Tests

- `test_index_project_with_documents()` — full pipeline: code + PDF + Markdown + YAML
- `test_search_mixed_results()` — code + doc chunks in single search
- `test_ignore_filter_integration()` — .tesseraignore excludes files from index
- `test_collection_search_with_documents()` — federated search across projects with docs
- `test_drift_adapter_migration()` — train on sample, query with transformed embeddings

### Performance Tests (Benchmarks)

- `bench_pdf_extraction()` — PyMuPDF4LLM on 50-page PDF
- `bench_markdown_chunking()` — 50KB markdown split time
- `bench_drift_training()` — 1000 embeddings training time
- `bench_drift_query_transform()` — 10K query transformations
- `bench_search_latency()` — keyword + semantic + RRF for 5-project collection

### Manual Testing

- Index PopupMaker core + 3 plugins with `.tesseraignore`
- Verify cross-project references still work
- Search for "webhook handling" and verify code + docs results
- Manually test Drift-Adapter on real embedding samples

---

## Dependencies on Prior Phases

| Phase | Feature | How Phase 4 Uses It |
|-------|---------|-------------------|
| Phase 1 | Code parsing + search | search() extends keyword + semantic search to documents |
| Phase 2 | Incremental indexing | Document files included in indexing job queue |
| Phase 3 | Collection federation | Document search federates via same RRF merge pattern |

---

## Future Consideration Points (Not Phase 4)

1. **Phase 5**: Graph edges from documents to code (e.g., architecture doc links to module)
2. **Phase 5**: Advanced semantic chunking (embedding similarity across sentences)
3. **Phase 5**: Document version history tracking
4. **Phase 5**: Low-Rank Affine Drift-Adapter variant for large embedding drift
5. **Phase 6**: Real-time document watching + incremental re-indexing
6. **Phase 6+**: DOCX, Confluence, Slack format support
7. **Phase 6+**: Custom metadata extraction from documents

---

## Appendix A: Configuration Examples

### Example `.tesseraignore`

```
# Default patterns are applied automatically.
# Override them here:

# Custom project exclusions
data/private/
.env
.env.local
secrets/

# Don't index third-party bundles in vendor/
vendor/third_party/
vendor/legacy/

# But DO index this important vendor library (override default vendor/)
!vendor/our-fork/

# Node projects: exclude specific build outputs
.turbo/
out/
dist/

# Don't index test coverage reports
coverage/
.nyc_output/
```

### Example: Index with Ignore

```bash
# Tessera will automatically load .tesseraignore from project root
uv run python -m tessera index /path/to/project \
  --languages php,typescript,markdown \
  --embedding-endpoint http://localhost:8000/v1

# .tesseraignore patterns applied automatically
# Hardcoded defaults + project patterns merged
```

### Example: Drift-Adapter Migration

```python
# Operator code: migrate embedding models

from tessera.drift_adapter import DriftAdapter
import numpy as np

# Load 50 samples from old index, embed with new model
old_embeddings = load_from_faiss_index()  # (50, 768)
new_embeddings = embed_samples_with_new_model()  # (50, 768)

# Train adapter
adapter = DriftAdapter(embedding_dim=768)
adapter.train(old_embeddings, new_embeddings)
adapter.save(".tessera/drift_matrix.npy")

# On next server start, operator updates config to new model
# Server loads adapter: DriftAdapter.load(".tessera/drift_matrix.npy", 768)
# Query-time: transform new embeddings before FAISS search
```

---

## References

1. [Drift-Adapter EMNLP 2025](https://aclanthology.org/2025.emnlp-main.805/)
2. [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
3. [Pathspec Library](https://pypi.org/project/pathspec/)
4. [SciPy Orthogonal Procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html)
5. [LangChain Markdown Header Splitter](https://python.langchain.com/v0.2/docs/how_to/markdown_header_metadata_splitter/)
6. [Tessera Phase 1–3 Documentation](../architecture/spec-v2.md)
