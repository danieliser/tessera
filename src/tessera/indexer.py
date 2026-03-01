"""Orchestration layer for incremental re-indexing.

Phase 1: Orchestrates parsing, chunking, embedding, and storage into a complete indexing pipeline.
Implements:
  - Project discovery and file enumeration
  - Incremental updates (new/modified/deleted files)
  - Symbol extraction and storage to SQLite
  - Chunk generation and embedding batch calls
  - Progress tracking and error recovery
"""

import json
import os
import hashlib
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from .db import ProjectDB, GlobalDB
from .parser import detect_language, parse_and_extract, Symbol, Reference, Edge
from .chunker import chunk_with_cast
from .embeddings import EmbeddingClient, EmbeddingUnavailableError
from .search import hybrid_search
from .document import (
    DocumentExtractionError, DocumentChunk,
    extract_pdf, chunk_markdown, chunk_yaml, chunk_json,
    chunk_html, chunk_xml, chunk_text_file,
)
from .ignore import IgnoreFilter
from .assets import (
    ASSET_EXTENSIONS, is_asset_file, get_asset_metadata, build_asset_synthetic_content,
)

logger = logging.getLogger(__name__)

# Structured document formats (format-specific chunking)
DOCUMENT_EXTENSIONS = ['.pdf', '.md', '.yaml', '.yml', '.json']

# Text formats (plaintext line-based chunking)
TEXT_EXTENSIONS = [
    '.txt', '.rst', '.csv', '.tsv', '.log',
    '.ini', '.cfg', '.toml', '.conf',
    '.htaccess', '.env.example', '.env.sample',
    '.editorconfig', '.prettierrc', '.eslintignore',
    '.gitattributes', '.npmrc', '.nvmrc',
    '.dockerignore', '.browserslistrc',
]

# Markup formats (tag stripping + plaintext chunking)
MARKUP_EXTENSIONS = ['.html', '.htm', '.xml', '.xsl', '.xslt', '.svg']

ALL_DOCUMENT_EXTENSIONS = DOCUMENT_EXTENSIONS + TEXT_EXTENSIONS + MARKUP_EXTENSIONS


def _detect_package_name(file_dir: str, project_root: str, cache: dict) -> str:
    """Detect package name by walking upward from file_dir to project_root.

    Checks for package.json, pyproject.toml, or composer.json and extracts
    the package name. Results are cached per directory.

    Returns:
        Package name string, or empty string if not found.
    """
    current = os.path.abspath(file_dir)
    root = os.path.abspath(project_root)

    while current.startswith(root):
        if current in cache:
            return cache[current]

        for manifest, extractor in [
            ("package.json", lambda c: json.loads(c).get("name", "")),
            ("composer.json", lambda c: json.loads(c).get("name", "")),
            ("pyproject.toml", _extract_pyproject_name),
        ]:
            path = os.path.join(current, manifest)
            if os.path.isfile(path):
                try:
                    with open(path, "r") as f:
                        name = extractor(f.read())
                    if name:
                        cache[current] = name
                        return name
                except (json.JSONDecodeError, OSError, KeyError):
                    pass

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    cache[file_dir] = ""
    return ""


def _extract_pyproject_name(content: str) -> str:
    """Extract project name from pyproject.toml without tomllib dependency."""
    in_project = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if in_project:
            if stripped.startswith("[") and stripped != "[project]":
                break
            if stripped.startswith("name"):
                # name = "foo"
                _, _, value = stripped.partition("=")
                return value.strip().strip('"').strip("'")
    return ""


@dataclass
class IndexStats:
    """Statistics for an indexing run."""
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    symbols_extracted: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    time_elapsed: float = 0.0


class IndexerPipeline:
    """Orchestrates the full indexing pipeline: parse → chunk → embed → store."""

    def __init__(
        self,
        project_path: str,
        project_db: Optional[ProjectDB] = None,
        global_db: Optional[GlobalDB] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        languages: Optional[List[str]] = None,
    ):
        """
        Initialize the indexer pipeline.

        Args:
            project_path: Root path of the project to index
            project_db: Optional ProjectDB instance (creates new if not provided)
            global_db: Optional GlobalDB for multi-project management
            embedding_client: Optional EmbeddingClient for semantic indexing
            languages: List of supported languages (default: PHP, TypeScript, Python, JavaScript)
        """
        self.project_path = os.path.abspath(project_path)
        self.project_db = project_db or ProjectDB(project_path)
        self.global_db = global_db
        self.embedding_client = embedding_client
        self.languages = languages or ['php', 'typescript', 'python', 'javascript']
        self.project_id = None  # set during register
        self._package_cache: Dict[str, str] = {}  # dir → package name
        self.ignore_filter = IgnoreFilter(self.project_path)

    def register(self, name: Optional[str] = None) -> int:
        """
        Register this project in the global DB.

        Args:
            name: Project name (defaults to directory name)

        Returns:
            Project ID
        """
        if self.global_db:
            name = name or os.path.basename(self.project_path)
            self.project_id = self.global_db.register_project(
                path=self.project_path,
                name=name,
                language=','.join(self.languages)
            )
        else:
            self.project_id = 1  # default for standalone usage

        return self.project_id

    def _discover_files(self) -> List[str]:
        """
        Walk project directory, return paths matching supported languages.

        Returns:
            Sorted list of absolute file paths
        """
        extensions = {
            'php': ['.php'],
            'typescript': ['.ts', '.tsx'],
            'javascript': ['.js', '.jsx'],
            'python': ['.py'],
            'swift': ['.swift'],
        }
        allowed_exts = set()
        for lang in self.languages:
            allowed_exts.update(extensions.get(lang, []))
        allowed_exts.update(ext for ext in ALL_DOCUMENT_EXTENSIONS)
        allowed_exts.update(ASSET_EXTENSIONS)

        files = []
        for root, dirs, filenames in os.walk(self.project_path):
            # Skip hidden dirs, node_modules, vendor, .git, .tessera
            dirs[:] = [
                d for d in dirs
                if not d.startswith('.') and d not in ('node_modules', 'vendor', '__pycache__', '.tessera')
            ]
            for f in filenames:
                # Skip TypeScript declaration files — generated from source .ts
                if f.endswith('.d.ts'):
                    continue
                if any(f.endswith(ext) for ext in allowed_exts):
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, self.project_path)
                    if not self.ignore_filter.should_ignore(rel_path):
                        files.append(abs_path)

        return sorted(files)

    def _get_git_head(self) -> Optional[str]:
        """Get current HEAD commit hash, or None if not a git repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_path,
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _get_changed_files(self, since_commit: str) -> List[str]:
        """Get files changed since a commit using git diff.

        Args:
            since_commit: Commit hash to diff against

        Returns:
            List of absolute paths to changed files (filtered to supported languages)
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMR", since_commit, "HEAD"],
                cwd=self.project_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                logger.warning("git diff failed: %s", result.stderr.strip())
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        extensions = {
            'php': ['.php'], 'typescript': ['.ts', '.tsx'],
            'javascript': ['.js', '.jsx'], 'python': ['.py'], 'swift': ['.swift'],
        }
        allowed_exts = set()
        for lang in self.languages:
            allowed_exts.update(extensions.get(lang, []))
        allowed_exts.update(ext for ext in ALL_DOCUMENT_EXTENSIONS)
        allowed_exts.update(ASSET_EXTENSIONS)

        changed = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            if any(line.endswith(ext) for ext in allowed_exts):
                abs_path = os.path.join(self.project_path, line)
                if os.path.exists(abs_path):
                    changed.append(abs_path)
        return changed

    def _get_deleted_files(self, since_commit: str) -> List[str]:
        """Get files deleted since a commit.

        Returns:
            List of relative paths that were deleted
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=D", since_commit, "HEAD"],
                cwd=self.project_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return [line for line in result.stdout.strip().split('\n') if line]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return []

    def index_changed(self, since_commit: Optional[str] = None) -> IndexStats:
        """Incremental reindex using git diff.

        If since_commit is provided, uses git diff to find changed files.
        If not, falls back to full index_project().

        Args:
            since_commit: Commit hash to diff against (uses project's last_indexed_commit if None)

        Returns:
            IndexStats with aggregate metrics
        """
        if not self.project_id:
            self.register()

        # Try to get since_commit from GlobalDB if not provided
        if since_commit is None and self.global_db:
            project = self.global_db.get_project(self.project_id)
            if project:
                since_commit = project.get("last_indexed_commit")

        # If still no commit, fall back to full index
        if since_commit is None:
            return self.index_project()

        start = time.perf_counter()
        stats = IndexStats()

        # Handle deleted files
        deleted = self._get_deleted_files(since_commit)
        for rel_path in deleted:
            self.project_db.delete_file_data(rel_path)

        # Handle changed/added files
        changed_files = self._get_changed_files(since_commit)
        for file_path in changed_files:
            result = self.index_file(file_path)
            if result['status'] == 'indexed':
                stats.files_processed += 1
                stats.symbols_extracted += result.get('symbols', 0)
                stats.chunks_created += result.get('chunks', 0)
                stats.chunks_embedded += result.get('embedded', 0)
            elif result['status'] == 'skipped':
                stats.files_skipped += 1
            elif result['status'] == 'failed':
                stats.files_failed += 1

        # Resolve cross-file edges
        cross_file_edges = self._resolve_cross_file_edges()
        logger.info("Cross-file edge resolution: %d edges created", cross_file_edges)

        stats.time_elapsed = time.perf_counter() - start

        # Update last indexed commit
        head = self._get_git_head()
        if head and self.global_db:
            self.global_db.update_last_indexed_commit(self.project_id, head)

        return stats

    def _file_hash(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of file contents for change detection.

        Args:
            file_path: Absolute path to file

        Returns:
            SHA-256 hex digest
        """
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _is_document_file(self, file_path: str) -> bool:
        """Check if file is a document (not code)."""
        return any(file_path.endswith(ext) for ext in ALL_DOCUMENT_EXTENSIONS)

    def _index_document_file(self, file_path: str) -> Dict[str, Any]:
        """Index a single document file with error isolation."""
        rel_path = os.path.relpath(file_path, self.project_path)

        try:
            # Determine file type and extract content
            if file_path.endswith('.pdf'):
                # Use pymupdf4llm directly for synchronous extraction
                try:
                    import pymupdf4llm
                except ImportError:
                    raise DocumentExtractionError(
                        "pymupdf4llm not installed. Install with: pip install pymupdf4llm"
                    )
                markdown_text = pymupdf4llm.to_markdown(file_path)
                chunks = chunk_markdown(markdown_text)
            elif file_path.endswith('.md'):
                content = Path(file_path).read_text(encoding='utf-8', errors='replace')
                chunks = chunk_markdown(content)
            elif file_path.endswith(('.yaml', '.yml')):
                chunks = chunk_yaml(file_path)
            elif file_path.endswith('.json'):
                chunks = chunk_json(file_path)
            elif file_path.endswith(('.html', '.htm')):
                chunks = chunk_html(file_path)
            elif file_path.endswith(('.xml', '.xsl', '.xslt', '.svg')):
                chunks = chunk_xml(file_path)
            elif any(file_path.endswith(ext) for ext in TEXT_EXTENSIONS):
                # Determine source_type from extension
                ext = os.path.splitext(file_path)[1].lstrip('.')
                source_type = ext if ext else 'text'
                chunks = chunk_text_file(file_path, source_type=source_type)
            else:
                return {'status': 'skipped', 'reason': 'unsupported document type'}

            # Compute file hash
            file_hash = self._file_hash(file_path)

            # Upsert file record (language='document')
            file_id = self.project_db.upsert_file(
                project_id=self.project_id,
                path=rel_path,
                language='document',
                file_hash=file_hash
            )

            # Check if file changed
            old_hash = self.project_db.get_old_hash() if hasattr(self.project_db, 'get_old_hash') else None
            existing = self.project_db.get_file(file_id=file_id)

            if (
                existing
                and existing.get('index_status') == 'indexed'
                and old_hash is not None
                and old_hash == file_hash
            ):
                return {'status': 'skipped', 'reason': 'unchanged'}

            # Clear old data and mark as pending
            with self.project_db.conn:
                self.project_db.clear_file_data(file_id)
                self.project_db.update_file_status(file_id, 'pending')

            # Convert DocumentChunk objects to chunk dicts
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    'project_id': self.project_id,
                    'file_id': file_id,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'symbol_ids': [],
                    'ast_type': 'document',
                    'chunk_type': 'document',
                    'content': chunk.content,
                    'source_type': chunk.source_type,
                    'section_heading': chunk.section_heading,
                    'key_path': chunk.key_path,
                    'page_number': chunk.page_number,
                    'parent_section': chunk.parent_section,
                    'file_path': rel_path,
                }
                chunk_dicts.append(chunk_dict)

            # Embed chunks if client available
            embeddings = None
            if self.embedding_client:
                try:
                    texts = [c['content'] for c in chunk_dicts]
                    embeddings = self.embedding_client.embed(texts) if texts else []
                except EmbeddingUnavailableError:
                    logger.warning("Embedding endpoint unavailable, storing document chunks without embeddings")
                    embeddings = None

            # Add embeddings to chunk dicts
            for i, chunk_dict in enumerate(chunk_dicts):
                if embeddings and i < len(embeddings):
                    chunk_dict['embedding'] = embeddings[i]

            # Store chunks
            if chunk_dicts:
                self.project_db.insert_chunks(chunk_dicts)

            # Mark file indexed
            self.project_db.update_file_status(file_id, 'indexed')

            return {
                'status': 'indexed',
                'chunks': len(chunks),
                'embedded': len(embeddings) if embeddings else 0
            }

        except DocumentExtractionError as e:
            logger.error(f"Document extraction error in {file_path}: {e}")
            try:
                file_id = self.project_db.upsert_file(
                    project_id=self.project_id,
                    path=rel_path,
                    language='document',
                    file_hash=self._file_hash(file_path)
                )
                self.project_db.update_file_status(file_id, 'failed')
            except Exception:
                pass
            return {'status': 'failed', 'reason': f'document extraction error: {e}'}
        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {e}")
            try:
                file_id = self.project_db.upsert_file(
                    project_id=self.project_id,
                    path=rel_path,
                    language='document',
                    file_hash=self._file_hash(file_path)
                )
                self.project_db.update_file_status(file_id, 'failed')
            except Exception:
                pass
            return {'status': 'failed', 'reason': str(e)}

    def _index_asset(self, file_path: str) -> Dict[str, Any]:
        """Index a binary asset file: extract metadata, create one FTS5-indexed chunk.

        Args:
            file_path: Absolute path to asset file

        Returns:
            Status dict with 'status' and optional metrics
        """
        rel_path = os.path.relpath(file_path, self.project_path)

        try:
            metadata = get_asset_metadata(file_path)

            path_components = rel_path.replace('\\', '/').split('/')
            synthetic_content = build_asset_synthetic_content(
                filename=os.path.basename(rel_path),
                path_components=path_components,
                mime_type=metadata['mime_type'],
                dimensions=metadata['dimensions'],
                file_size=metadata['file_size'],
            )

            file_hash = self._file_hash(file_path)

            file_id = self.project_db.upsert_file(
                project_id=self.project_id,
                path=rel_path,
                language='asset',
                file_hash=file_hash,
            )

            old_hash = self.project_db.get_old_hash() if hasattr(self.project_db, 'get_old_hash') else None
            existing = self.project_db.get_file(file_id=file_id)

            if (
                existing
                and existing.get('index_status') == 'indexed'
                and old_hash is not None
                and old_hash == file_hash
            ):
                return {'status': 'skipped', 'reason': 'unchanged'}

            dimensions = metadata['dimensions']
            key_path = f"{dimensions['width']}x{dimensions['height']}" if dimensions else None

            with self.project_db.conn:
                self.project_db.clear_file_data(file_id)
                self.project_db.update_file_status(file_id, 'pending')

                chunk_dict = {
                    'project_id': self.project_id,
                    'file_id': file_id,
                    'start_line': 0,
                    'end_line': 0,
                    'symbol_ids': [],
                    'ast_type': metadata['category'],
                    'chunk_type': 'asset',
                    'content': synthetic_content,
                    'source_type': 'asset',
                    'length': metadata['file_size'],
                    'key_path': key_path,
                    'section_heading': None,
                    'page_number': None,
                    'parent_section': None,
                    'file_path': rel_path,
                }

                self.project_db.insert_chunks([chunk_dict])

            self.project_db.update_file_status(file_id, 'indexed')

            return {
                'status': 'indexed',
                'chunks': 1,
                'embedded': 0,
                'mime_type': metadata['mime_type'],
                'dimensions': dimensions,
                'file_size': metadata['file_size'],
            }

        except Exception as e:
            logger.error("Failed to index asset %s: %s", file_path, e)
            try:
                file_id = self.project_db.upsert_file(
                    project_id=self.project_id,
                    path=rel_path,
                    language='asset',
                    file_hash=self._file_hash(file_path),
                )
                self.project_db.update_file_status(file_id, 'failed')
            except Exception:
                pass
            return {'status': 'failed', 'reason': str(e)}

    def _append_asset_chunk(self, file_path: str) -> None:
        """Append an asset metadata chunk to an already-indexed file (for SVG dual-indexing).

        Unlike _index_asset(), this does not upsert the file record or clear existing data.
        It only inserts an additional chunk with source_type='asset'.
        """
        rel_path = os.path.relpath(file_path, self.project_path)
        try:
            metadata = get_asset_metadata(file_path)
            path_components = rel_path.replace('\\', '/').split('/')
            synthetic_content = build_asset_synthetic_content(
                filename=os.path.basename(rel_path),
                path_components=path_components,
                mime_type=metadata['mime_type'],
                dimensions=metadata['dimensions'],
                file_size=metadata['file_size'],
            )

            # Look up existing file_id (already created by document indexer)
            existing = self.project_db.get_file(path=rel_path)
            if not existing:
                return
            file_id = existing['id']

            dimensions = metadata['dimensions']
            key_path = f"{dimensions['width']}x{dimensions['height']}" if dimensions else None

            chunk_dict = {
                'project_id': self.project_id,
                'file_id': file_id,
                'start_line': 0,
                'end_line': 0,
                'symbol_ids': [],
                'ast_type': metadata['category'],
                'chunk_type': 'asset',
                'content': synthetic_content,
                'source_type': 'asset',
                'length': metadata['file_size'],
                'key_path': key_path,
                'section_heading': None,
                'page_number': None,
                'parent_section': None,
                'file_path': rel_path,
            }
            self.project_db.insert_chunks([chunk_dict])
        except Exception as e:
            logger.warning("Failed to append asset chunk for SVG %s: %s", file_path, e)

    def index_file(self, file_path: str) -> Dict[str, Any]:
        """
        Index a single file: parse, chunk, embed, store.

        Args:
            file_path: Absolute path to file

        Returns:
            Status dict with 'status' and optional 'reason' or metrics
        """
        # Route asset files BEFORE document check (some assets like .svg are also documents)
        is_svg = file_path.lower().endswith('.svg')
        if is_asset_file(file_path) and not is_svg:
            return self._index_asset(file_path)

        # Route document files to document indexer
        if self._is_document_file(file_path):
            doc_result = self._index_document_file(file_path)
            # SVG dual-indexing: append asset chunk after document indexing
            if is_svg and doc_result.get('status') == 'indexed':
                self._append_asset_chunk(file_path)
            return doc_result

        rel_path = os.path.relpath(file_path, self.project_path)
        language = detect_language(file_path)

        if not language:
            logger.debug("Skipping unrecognized binary file: %s", file_path)
            return {'status': 'skipped', 'reason': 'unsupported language'}

        # Read file
        try:
            source_code = Path(file_path).read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            return {'status': 'failed', 'reason': str(e)}

        # Compute file hash
        file_hash = self._file_hash(file_path)

        # Upsert file record (returns file_id, may be existing file)
        file_id = self.project_db.upsert_file(
            project_id=self.project_id,
            path=rel_path,
            language=language,
            file_hash=file_hash
        )

        # Check if file changed - get the old hash from before upsert
        old_hash = self.project_db.get_old_hash() if hasattr(self.project_db, 'get_old_hash') else None
        existing = self.project_db.get_file(file_id=file_id)

        # If file was already indexed with the same hash, skip it
        # old_hash is not None means this was an existing file
        if (
            existing
            and existing.get('index_status') == 'indexed'
            and old_hash is not None  # This was an existing file
            and old_hash == file_hash  # And the hash is the same
        ):
            return {'status': 'skipped', 'reason': 'unchanged'}

        # Parse FIRST (before clearing old data — if parsing fails, old data preserved)
        try:
            symbols, references, edges = parse_and_extract(file_path, source_code)
        except Exception as e:
            return {'status': 'failed', 'reason': f'parse error: {e}'}

        # Wrap clear + insert in a transaction for atomicity
        with self.project_db.conn:
            # Clear old data only after successful parse
            self.project_db.clear_file_data(file_id)
            self.project_db.update_file_status(file_id, 'pending')

            # Store symbols
            symbol_dicts = [
                {
                    'project_id': self.project_id,
                    'file_id': file_id,
                    'name': s.name,
                    'kind': s.kind,
                    'line': s.line,
                    'col': s.col,
                    'scope': s.scope,
                    'signature': s.signature,
                }
                for s in symbols
            ]
            symbol_ids = self.project_db.insert_symbols(symbol_dicts) if symbol_dicts else []

            # Insert a module symbol with the file's relative path as name
            pkg_name = _detect_package_name(
                os.path.dirname(file_path), self.project_path, self._package_cache
            )
            module_sym = {
                'project_id': self.project_id,
                'file_id': file_id,
                'name': rel_path,
                'kind': 'module',
                'line': 0,
                'col': 0,
                'scope': '',
                'signature': pkg_name,
            }
            module_ids = self.project_db.insert_symbols([module_sym])
            module_id = module_ids[0] if module_ids else None

            # Store references
            # Build name→id map with both qualified and short names for lookup.
            # PHP symbols may be namespace-qualified (App\Analytics\Tracker) but
            # refs use short names (Tracker). Also map scope-qualified method names.
            name_to_id = {}
            for s, sid in zip(symbols, symbol_ids):
                name_to_id[s.name] = sid
                # For namespaced symbols, also index the short name (last segment)
                if '\\' in s.name:
                    short = s.name.rsplit('\\', 1)[-1]
                    # Only set if not already mapped (avoid overwriting)
                    if short not in name_to_id:
                        name_to_id[short] = sid
                # For scoped methods, also index "scope.method" short form
                if s.scope and s.kind == 'method':
                    # Map short scope name → method for ref resolution
                    short_scope = s.scope.rsplit('\\', 1)[-1] if '\\' in s.scope else s.scope
                    name_to_id.setdefault(f"{short_scope}.{s.name}", sid)

            # Map module symbol — parser refs use '<module>' as from_symbol
            if module_id:
                name_to_id['<module>'] = module_id
                name_to_id[rel_path] = module_id

            ref_dicts = []
            for r in references:
                from_id = name_to_id.get(r.from_symbol)
                to_id = name_to_id.get(r.to_symbol)
                if from_id:  # from must exist; to can be None (external)
                    ref_dicts.append({
                        'project_id': self.project_id,
                        'from_symbol_id': from_id,
                        'to_symbol_id': to_id,
                        'to_symbol_name': r.to_symbol,
                        'kind': r.kind,
                        'context': r.context,
                        'line': r.line
                    })

            if ref_dicts:
                self.project_db.insert_refs(ref_dicts)

            # Store edges
            edge_dicts = []
            for e in edges:
                from_id = name_to_id.get(e.from_name)
                to_id = name_to_id.get(e.to_name)
                if from_id and to_id:
                    edge_dicts.append({
                        'project_id': self.project_id,
                        'from_id': from_id,
                        'to_id': to_id,
                        'type': e.type,
                        'weight': e.weight
                    })

            if edge_dicts:
                self.project_db.insert_edges(edge_dicts)

        # Chunk
        chunks = chunk_with_cast(source_code, language)

        # Embed chunks (if client available)
        embeddings = None
        if self.embedding_client:
            try:
                texts = [c.content for c in chunks]
                embeddings = self.embedding_client.embed(texts) if texts else []
            except EmbeddingUnavailableError:
                logger.warning("Embedding endpoint unavailable, storing chunks without embeddings")
                embeddings = None

        # Store chunks
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            # Find which symbol IDs overlap with this chunk's line range
            overlapping_ids = [
                sid
                for s, sid in zip(symbols, symbol_ids)
                if s.line >= chunk.start_line and s.line <= chunk.end_line
            ]

            d = {
                'project_id': self.project_id,
                'file_id': file_id,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'symbol_ids': overlapping_ids,
                'ast_type': chunk.ast_type,
                'chunk_type': 'code',
                'content': chunk.content,
                'file_path': rel_path,
            }

            if embeddings and i < len(embeddings):
                d['embedding'] = embeddings[i]

            chunk_dicts.append(d)

        if chunk_dicts:
            self.project_db.insert_chunks(chunk_dicts)

        # Mark file indexed
        self.project_db.update_file_status(file_id, 'indexed')

        return {
            'status': 'indexed',
            'symbols': len(symbols),
            'refs': len(references),
            'edges': len(edges),
            'chunks': len(chunks),
            'embedded': len(embeddings) if embeddings else 0
        }

    def index_project(self) -> IndexStats:
        """
        Index entire project. Creates a job record in GlobalDB for tracking.

        Returns:
            IndexStats with aggregate metrics
        """
        start = time.perf_counter()

        if not self.project_id:
            self.register()

        # Create job record for tracking
        job_id = None
        if self.global_db:
            job_id = self.global_db.create_job(self.project_id)
            self.global_db.start_job(job_id)

        try:
            files = self._discover_files()
            stats = IndexStats()

            for file_path in files:
                result = self.index_file(file_path)

                if result['status'] == 'indexed':
                    stats.files_processed += 1
                    stats.symbols_extracted += result.get('symbols', 0)
                    stats.chunks_created += result.get('chunks', 0)
                    stats.chunks_embedded += result.get('embedded', 0)
                elif result['status'] == 'skipped':
                    stats.files_skipped += 1
                elif result['status'] == 'failed':
                    stats.files_failed += 1
                    logger.error(f"Failed to index {file_path}: {result.get('reason')}")

            # Resolve cross-file edges
            cross_file_edges = self._resolve_cross_file_edges()
            logger.info("Cross-file edge resolution: %d edges created", cross_file_edges)

            stats.time_elapsed = time.perf_counter() - start

            if job_id and self.global_db:
                self.global_db.complete_job(job_id)

            # Store last indexed commit for future incremental reindexing
            head = self._get_git_head()
            if head and self.global_db:
                self.global_db.update_last_indexed_commit(self.project_id, head)

            return stats
        except Exception as e:
            if job_id and self.global_db:
                self.global_db.fail_job(job_id, str(e))
            raise

    def _resolve_cross_file_edges(self) -> int:
        """Resolve cross-file references into edges.

        Queries unresolved refs (to_symbol_id IS NULL) and matches
        to_symbol_name against all symbols in the project. Creates
        edges for resolved matches and updates refs.to_symbol_id.

        Returns:
            Number of cross-file edges created.
        """
        if not self.project_id:
            return 0

        # Query all unresolved refs for this project
        cursor = self.project_db.conn.execute(
            """
            SELECT id, from_symbol_id, to_symbol_name, kind
            FROM refs
            WHERE project_id = ? AND to_symbol_id IS NULL AND to_symbol_name IS NOT NULL
            """,
            (self.project_id,)
        )
        unresolved_refs = [dict(row) for row in cursor.fetchall()]

        if not unresolved_refs:
            return 0

        # Build a name-to-symbol lookup for all symbols in the project
        cursor = self.project_db.conn.execute(
            "SELECT id, name, kind FROM symbols WHERE project_id = ?",
            (self.project_id,)
        )
        all_symbols = [dict(row) for row in cursor.fetchall()]

        # Build lookup: name -> symbol_id (with preference logic)
        # Prefer: functions > classes > methods
        # Define preference order
        kind_priority = {'function': 0, 'class': 1, 'method': 2, 'other': 3}

        # Group symbols by name with their priorities
        name_to_candidates: Dict[str, List[tuple]] = {}  # name -> [(symbol_id, priority), ...]
        for symbol in all_symbols:
            name = symbol['name']
            symbol_id = symbol['id']
            kind = symbol['kind']
            priority = kind_priority.get(kind, 3)

            if name not in name_to_candidates:
                name_to_candidates[name] = []
            name_to_candidates[name].append((symbol_id, priority))

        # Resolve each name to a single symbol (if unambiguous)
        name_to_symbol: Dict[str, int] = {}
        for name, candidates in name_to_candidates.items():
            # Sort by priority (lowest first)
            candidates.sort(key=lambda x: x[1])
            best_priority = candidates[0][1]

            # Check if ambiguous (multiple symbols with same best priority)
            best_candidates = [c for c in candidates if c[1] == best_priority]
            if len(best_candidates) == 1:
                # Unambiguous: use the single best candidate
                name_to_symbol[name] = best_candidates[0][0]
            # else: ambiguous (multiple symbols with same best priority), skip

        # Also handle short names for namespaced symbols (PHP)
        # Try to resolve short names, but don't override existing mappings
        for symbol in all_symbols:
            full_name = symbol['name']
            if '\\' in full_name:
                short_name = full_name.rsplit('\\', 1)[-1]
                # Only set if not already mapped (avoid overwriting unambiguous full names)
                if short_name not in name_to_symbol:
                    # Check if short_name is unambiguous among namespaced symbols
                    short_candidates = [s for s in all_symbols if s['name'].endswith('\\' + short_name) or s['name'] == short_name]
                    if len(short_candidates) == 1:
                        name_to_symbol[short_name] = symbol['id']

        # Check for existing edges to avoid duplicates
        cursor = self.project_db.conn.execute(
            """
            SELECT from_id, to_id FROM edges
            WHERE project_id = ?
            """,
            (self.project_id,)
        )
        existing_edges = set(
            (row[0], row[1]) for row in cursor.fetchall()
        )

        # Resolve refs and prepare edges to insert
        edges_to_insert = []
        refs_to_update = []

        for ref in unresolved_refs:
            to_symbol_name = ref['to_symbol_name']
            from_symbol_id = ref['from_symbol_id']
            ref_id = ref['id']
            kind = ref['kind']

            # Try to resolve the symbol
            if to_symbol_name in name_to_symbol:
                to_symbol_id = name_to_symbol[to_symbol_name]

                # Check if edge already exists
                if (from_symbol_id, to_symbol_id) not in existing_edges:
                    edges_to_insert.append({
                        'project_id': self.project_id,
                        'from_id': from_symbol_id,
                        'to_id': to_symbol_id,
                        'type': kind,
                        'weight': 1.0,
                    })
                    existing_edges.add((from_symbol_id, to_symbol_id))

                # Update the ref with resolved symbol_id
                refs_to_update.append((to_symbol_id, ref_id))

        # Insert edges in batch
        if edges_to_insert:
            self.project_db.insert_edges(edges_to_insert)

        # Update refs to mark them as resolved
        with self.project_db.conn:
            for to_symbol_id, ref_id in refs_to_update:
                self.project_db.conn.execute(
                    "UPDATE refs SET to_symbol_id = ? WHERE id = ?",
                    (to_symbol_id, ref_id)
                )

        logger.info(
            "Resolved %d cross-file edges from %d unresolved refs",
            len(edges_to_insert),
            len(unresolved_refs)
        )

        return len(edges_to_insert)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid search across indexed project.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of search results
        """
        query_embedding = None

        if self.embedding_client:
            try:
                import numpy as np
                embedding = self.embedding_client.embed_single(query)
                query_embedding = np.array(embedding, dtype=np.float32)
            except EmbeddingUnavailableError:
                pass

        return hybrid_search(query, query_embedding, self.project_db, limit)
