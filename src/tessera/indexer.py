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

logger = logging.getLogger(__name__)


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
                    files.append(os.path.join(root, f))

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

    def index_file(self, file_path: str) -> Dict[str, Any]:
        """
        Index a single file: parse, chunk, embed, store.

        Args:
            file_path: Absolute path to file

        Returns:
            Status dict with 'status' and optional 'reason' or metrics
        """
        rel_path = os.path.relpath(file_path, self.project_path)
        language = detect_language(file_path)

        if not language:
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
