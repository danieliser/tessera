# Phase 4.5 Specification: Media & Binary File Metadata Catalog

**Version:** 1.0
**Date:** 2026-02-28
**Tier:** Quick
**Estimated Effort:** ~200-300 LOC production, ~100 LOC tests

---

## Executive Summary

Phase 4.5 extends Tessera's indexing pipeline to catalog binary asset files (images, video, audio, fonts, archives) as searchable records without embedding computation or content extraction. A new `src/tessera/assets.py` module provides image dimension extraction via stdlib `struct`, MIME type detection, and synthetic content generation. Asset discovery is integrated into the existing indexing pipeline with zero schema changes — metadata is stored in the existing `chunk_meta` table using `source_type='asset'` for filtering. Existing `.tesseraignore` patterns are automatically respected.

---

## Problem Statement

During Phase 4 live testing, agents searching for assets (e.g., "logo", "hero image") receive zero results even when those files exist in the project. Binary files are currently invisible to Tessera's search, limiting its usefulness for image-heavy projects, documentation sites, and multimedia applications. This is a gap in capability, not a performance issue.

---

## Proposed Solution

1. **Asset module** (`src/tessera/assets.py`): Helpers for dimension extraction (PNG, JPEG, GIF, BMP), MIME type detection, asset category mapping, and synthetic content generation.
2. **Indexer integration**: Route asset files to a new `index_asset()` method in `IndexerPipeline` that extracts metadata and stores one FTS5-indexed chunk per asset.
3. **Search-time filtering**: Existing `search()` tool gets optional `source_type` parameter to filter results to assets only.
4. **Ignore support**: Asset discovery respects existing `.tesseraignore` patterns (no additional logic needed — applies at file discovery layer).

Result: `search("logo")` returns `{"source_type": "asset", "path": "assets/images/logo.png", "content": "logo.png assets images image/png 1200x800"}` and similar matches without breaking existing search contracts.

---

## Implementation

### 1. New Module: `src/tessera/assets.py`

**Purpose:** Dimension extraction, MIME detection, category mapping, synthetic content building.

**Key Functions:**

- `extract_image_dimensions(file_path: str) -> Optional[Dict[str, int]]`
  Reads PNG, JPEG, GIF, BMP headers using `struct` module (zero external dependencies). Returns `{'width': int, 'height': int}` or `None` on error.

- `get_mime_type(filename: str) -> str`
  Uses `mimetypes.guess_type()` + small supplemental map for missing types (`.woff`, `.woff2`, `.opus`). Returns MIME type string or `'application/octet-stream'`.

- `build_asset_synthetic_content(filename: str, path_components: List[str], mime_type: str, dimensions: Optional[Dict[str, int]] = None, file_size: int = 0) -> str`
  Builds whitespace-separated FTS5 content: filename + path components + MIME type + dimensions + file size. Example: `"logo.png assets images image/png 1200x800 45KB"`.

- `is_asset_file(file_path: str) -> bool`
  Returns `True` if file extension is in `ASSET_EXTENSIONS`. Note: `.svg` is included — dual-indexed as both XML text (for content) and asset (for filename/path discoverability).

- `get_asset_category(file_path: str) -> str`
  Maps extension to category: `'image'`, `'video'`, `'audio'`, `'font'`, `'archive'`, or `'binary'`.

**Constants:**

```python
ASSET_EXTENSIONS = {
    # Images
    '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.gif': 'image',
    '.bmp': 'image', '.webp': 'image', '.ico': 'image', '.tiff': 'image',
    '.heif': 'image', '.heic': 'image', '.avif': 'image',
    '.svg': 'image',  # Also indexed as XML text — dual-index for filename/path discovery
    # Video
    '.mp4': 'video', '.mkv': 'video', '.webm': 'video', '.avi': 'video', '.mov': 'video', '.flv': 'video',
    # Audio
    '.mp3': 'audio', '.wav': 'audio', '.aac': 'audio', '.flac': 'audio', '.m4a': 'audio', '.opus': 'audio',
    # Fonts
    '.woff': 'font', '.woff2': 'font', '.ttf': 'font', '.otf': 'font',
    # Archives
    '.zip': 'archive', '.tar': 'archive', '.gz': 'archive', '.rar': 'archive', '.7z': 'archive',
}

SUPPLEMENTAL_MIME_TYPES = {
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.opus': 'audio/opus',
    '.heif': 'image/heif',
    '.heic': 'image/heic',
    '.avif': 'image/avif',
}
```

**Code Size:** ~250 LOC

---

### 2. Changes to `src/tessera/indexer.py`

**Add `ASSET_EXTENSIONS` constant:**

```python
ASSET_EXTENSIONS = [
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.ico', '.tiff',
    '.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv',
    '.mp3', '.wav', '.aac', '.flac', '.m4a', '.opus',
    '.woff', '.woff2', '.ttf', '.otf',
    '.zip', '.tar', '.gz', '.rar', '.7z',
]
```

**Update `_discover_files()` to include assets:**

```python
allowed_exts = set(ALL_DOCUMENT_EXTENSIONS)
for lang in self.languages:
    allowed_exts.update(extensions.get(lang, []))
allowed_exts.update(ASSET_EXTENSIONS)  # ← NEW
```

**Update `_get_changed_files()` to include assets:**

```python
allowed_exts = set(ALL_DOCUMENT_EXTENSIONS)
for lang in self.languages:
    allowed_exts.update(extensions.get(lang, []))
allowed_exts.update(ASSET_EXTENSIONS)  # ← NEW
```

**Add `index_asset()` method:**

```python
def index_asset(self, file_path: str) -> Dict[str, Any]:
    """Index a binary asset file: extract metadata, create one FTS5-indexed chunk.

    Args:
        file_path: Absolute path to asset file

    Returns:
        Status dict with 'status' and optional metrics
    """
    from .assets import (
        extract_image_dimensions, get_mime_type, build_asset_synthetic_content,
        get_asset_category
    )

    rel_path = os.path.relpath(file_path, self.project_path)

    try:
        # Extract dimensions (for images)
        dimensions = extract_image_dimensions(file_path) if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) else None

        # Get MIME type
        mime_type = get_mime_type(os.path.basename(file_path))

        # Get file size
        file_size = os.path.getsize(file_path)

        # Get file hash
        file_hash = self._file_hash(file_path)

        # Create synthetic content for FTS5
        synthetic_content = build_asset_synthetic_content(
            filename=os.path.basename(rel_path),
            path_components=rel_path.split(os.sep),
            mime_type=mime_type,
            dimensions=dimensions,
            file_size=file_size
        )

        # Upsert file record (language='asset')
        file_id = self.project_db.upsert_file(
            project_id=self.project_id,
            path=rel_path,
            language='asset',
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

        # Clear old data
        with self.project_db.conn:
            self.project_db.clear_file_data(file_id)
            self.project_db.update_file_status(file_id, 'pending')

            # Create one chunk per asset (no embedding computation)
            chunk_dict = {
                'project_id': self.project_id,
                'file_id': file_id,
                'start_line': 0,
                'end_line': 0,
                'symbol_ids': [],
                'ast_type': get_asset_category(file_path),
                'chunk_type': 'asset',
                'content': synthetic_content,
                'source_type': 'asset',
                'length': file_size,
                'key_path': f"{dimensions['width']}x{dimensions['height']}" if dimensions else None,
                'section_heading': None,
                'page_number': None,
                'parent_section': None,
            }

            # Store chunk (no embedding)
            self.project_db.insert_chunks([chunk_dict])

        # Mark file indexed
        self.project_db.update_file_status(file_id, 'indexed')

        return {
            'status': 'indexed',
            'chunks': 1,
            'embedded': 0,
            'mime_type': mime_type,
            'dimensions': dimensions,
            'file_size': file_size
        }

    except Exception as e:
        logger.error(f"Failed to index asset {file_path}: {e}")
        try:
            file_id = self.project_db.upsert_file(
                project_id=self.project_id,
                path=rel_path,
                language='asset',
                file_hash=self._file_hash(file_path)
            )
            self.project_db.update_file_status(file_id, 'failed')
        except Exception:
            pass
        return {'status': 'failed', 'reason': str(e)}
```

**Update `index_file()` to route assets:**

```python
def index_file(self, file_path: str) -> Dict[str, Any]:
    """Index a single file: route to asset, document, or code handler."""

    from .assets import is_asset_file

    # Check if asset BEFORE document check (documents are broader)
    if is_asset_file(file_path):
        return self.index_asset(file_path)

    # Route document files to document indexer
    if self._is_document_file(file_path):
        return self._index_document_file(file_path)

    # ... rest of existing code (code indexing)
```

**Unrecognized binary file logging:** During file discovery, log files with unrecognized binary extensions at `DEBUG` level. This provides visibility into what's being skipped without adding noise:

```python
if self._is_binary_file(file_path) and not is_asset_file(file_path):
    logger.debug("Skipping unrecognized binary file: %s", file_path)
```

**Code Size:** ~150 LOC

---

### 3. Changes to `src/tessera/db.py`

**No schema changes required.**

Existing `chunk_meta` table already supports asset metadata:

```sql
CREATE TABLE IF NOT EXISTS chunk_meta (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    symbol_ids TEXT,
    ast_type TEXT,                    -- populated with 'image', 'video', etc.
    chunk_type TEXT,                  -- populated with 'asset'
    content TEXT NOT NULL,            -- FTS5-indexed synthetic string
    length INTEGER,                   -- file size in bytes
    source_type TEXT,                 -- populated with 'asset'
    section_heading TEXT,             -- NULL for assets
    key_path TEXT,                    -- populated with "1200x800" for images
    page_number INTEGER,              -- NULL for assets
    parent_section TEXT,              -- NULL for assets
    FOREIGN KEY(file_id) REFERENCES files(id)
)
```

The `insert_chunks()` method already accepts all these fields. No code changes needed.

---

### 4. Changes to `src/tessera/server.py`

**Add `source_type` parameter filtering to search results** (if not already present):

```python
@app.tool()
def search(
    query: str,
    limit: int = 10,
    source_type: Optional[str] = None,  # NEW: optional filter
) -> List[Dict[str, Any]]:
    """Search across code, documents, and assets.

    Args:
        query: Search query
        limit: Maximum results (default 10)
        source_type: Optional filter: 'code', 'document', 'asset' (returns all types if None)

    Returns:
        List of search result dicts with source_type field
    """
    # ... existing search logic ...

    results = hybrid_search(query, query_embedding, self.project_db, limit)

    # Filter by source_type if provided
    if source_type:
        results = [r for r in results if r.get('source_type') == source_type]

    return results
```

**Code Size:** ~10 LOC (minimal)

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Use `struct` for image dimensions | Zero dependencies, proven 10x faster than Pillow for dimension-only extraction. Covers PNG, JPEG, GIF, BMP (~95% of real projects). |
| Use `mimetypes.guess_type()` | Stdlib, simple, sufficient for metadata context. Future phases can add magic byte validation if needed. |
| Single synthetic content string for FTS5 | FTS5's tokenizer handles whitespace separation. Single `content` column avoids unnecessary schema complexity. |
| No embeddings for assets | Asset metadata (e.g., "logo.png image/png 1200x800") has zero semantic value for similarity search. FTS5 is sufficient. |
| Store in existing `chunk_meta` table | Reusing columns (`source_type='asset'`, `ast_type=category`, `key_path=dimensions`) avoids schema migration and keeps data model unified. |
| Dual-index `.svg` files | SVG content is indexed as XML text by the document handler, but an agent searching for "icon" by filename won't find `icons/menu.svg` via XML content search. Dual-indexing provides filename/path discoverability alongside content indexing. |
| Respect `.tesseraignore` at discovery layer | Ignore filter already applied in `_discover_files()` — asset routing happens after filtering, so no additional logic needed. |
| No new MCP tools | Existing `search()` with `source_type` filtering handles all use cases. |

---

## Architecture Diagram

```
File Discovery (_discover_files)
  |
  +-- Filter by extension (code, document, ASSET) ← ASSET_EXTENSIONS added
  +-- Apply .tesseraignore (already exists)
  |
  v
File Routing (index_file)
  |
  +-- is_asset_file? → index_asset()     ← NEW routing
  |    |
  |    +-- Extract dimensions (struct)
  |    +-- Get MIME type (mimetypes)
  |    +-- Build synthetic content
  |    +-- Store one chunk (no embedding)
  |
  +-- is_document? → _index_document_file()
  |
  +-- is_code? → existing parse/chunk/embed flow
```

---

## Acceptance Criteria

1. **Asset Discovery:** `search("logo")` returns results for asset files like `assets/images/logo.png` with `source_type='asset'`.
2. **Metadata Searchability:** Asset results include path, filename, MIME type, and dimensions (for images) in the `content` field.
3. **Image Dimensions:** PNG, JPEG, GIF, BMP files have `key_path='1200x800'` (or actual dimensions).
4. **Filtering:** `search("logo", source_type="asset")` returns only asset results.
5. **SVG Dual-Indexing:** `.svg` files are indexed both as XML documents (content search) and as assets (filename/path discovery).
6. **Ignore Respect:** Files matched by `.tesseraignore` are excluded from asset discovery.
7. **No Embeddings:** Asset chunks are stored without embedding computation (zero embedding calls).
8. **Backward Compatibility:** All existing tests pass. No breaking changes to `search()` contract (source_type is optional, defaults to None = all types).
9. **Error Handling:** Malformed asset files (e.g., truncated PNG) fail gracefully with logged errors, not indexing crashes.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Malformed image headers cause `struct.unpack()` crashes | Low | Index crash | Wrap dimension extraction in try/except; return None on error; log warning |
| Asset discovery is slow for large projects with many binaries | Low | Indexing slowdown | Asset indexing is fast (no parsing, no embedding). Worst case: 1000 images = negligible overhead. Profile if concern arises. |
| MIME type detection misses custom/obscure extensions | Medium | Incomplete metadata | Covered by supplemental map for common font types. Fallback to `'application/octet-stream'`. Phase 5 can add magic byte detection if needed. |
| `.svg` files dual-indexed (XML + asset) | Low | Slight result duplication | Acceptable — XML result has content, asset result has filename/path metadata. Different search intents served. `source_type` filter can disambiguate. |
| Search results polluted with too many asset hits for broad queries | Low | Noise in results | Users can filter with `source_type="code"` or `source_type="document"`. Default behavior (no filter) is acceptable. |
| Dimension extraction fails silently for WebP | Low | Missing dimensions | WebP deferred to Phase 5. Complexity outweighs benefit. Include in Non-Goals. |

---

## Dependencies

- **Build-time:** No new dependencies (uses stdlib `struct`, `mimetypes`, `os`, `hashlib`).
- **Runtime:** Existing Tessera dependencies (`db.py`, `indexer.py`, `search.py`).
- **External services:** None (no API calls, no external dimension service).

---

## Test Strategy

### Unit Tests (`tests/test_assets.py`)

1. **Dimension extraction:**
   - PNG with valid header → returns `{'width': 1200, 'height': 800}`
   - JPEG with SOF0 marker → returns dimensions
   - GIF with valid header → returns dimensions
   - BMP (modern DIB, old DIB) → returns dimensions
   - Truncated file → returns None
   - Invalid magic bytes → returns None

2. **MIME type detection:**
   - `.png` → `'image/png'`
   - `.mp4` → `'video/mp4'`
   - `.woff2` → `'font/woff2'` (supplemental)
   - Unknown extension → `'application/octet-stream'`

3. **Asset category mapping:**
   - `.png` → `'image'`
   - `.mp4` → `'video'`
   - `.mp3` → `'audio'`
   - `.zip` → `'archive'`
   - Unknown → `'binary'`

4. **Synthetic content building:**
   - With dimensions: "logo.png assets images image/png 1200x800 45KB"
   - Without dimensions: "archive.zip assets archives application/zip 1234567B"
   - Path components included: "file.png" + ["assets", "images"] → "file.png assets images ..."

5. **Asset file detection:**
   - `.png` → True
   - `.svg` → True (dual-indexed as asset for filename/path discovery)
   - `.py` → False

### Integration Tests (`tests/test_indexer_assets.py`)

1. **End-to-end indexing:**
   - Create test project with PNG, MP4, ZIP files
   - Run `IndexerPipeline.index_project()`
   - Verify chunks created with `source_type='asset'`
   - Verify no embeddings computed for assets

2. **Search filtering:**
   - Index mixed project (code, documents, assets)
   - `search("logo")` returns asset + any code/doc matches
   - `search("logo", source_type="asset")` returns only asset
   - `search("def", source_type="code")` returns only code

3. **Ignore patterns:**
   - Create `.tesseraignore` with `*.png`
   - Index project with PNG files
   - Verify ignored files are not indexed

4. **File change detection:**
   - Index asset file once
   - Verify `index_asset()` returns `{'status': 'skipped', 'reason': 'unchanged'}` on re-run
   - Modify file (change bytes)
   - Verify file is re-indexed

5. **Error handling:**
   - Truncated PNG file → `index_asset()` returns `{'status': 'indexed', 'dimensions': None}`
   - Unreadable file (permissions) → `index_asset()` returns `{'status': 'failed', 'reason': '...'}`

---

## Non-Goals (Explicit Out-of-Scope)

- **WebP dimension extraction:** Complex VP8/VP8L bitstream parsing. Deferred to Phase 5.
- **EXIF/IPTC metadata extraction:** Phase 5+ (image-specific, low priority).
- **OCR from images:** Phase 5+ (expensive, separate workflow).
- **Content extraction from archives:** Phase 5+ (requires zip/tar parsing + recursion).
- **Thumbnail generation:** Phase 5+ (requires image processing library).
- **Magic byte validation:** Phase 5+ (for security-critical file validation).
- **Separate asset search tool:** Not needed; filter existing `search()` tool with `source_type`.
- **Asset preview/streaming:** Out of scope for indexing phase.

---

## Deferred Items

None. This spec is complete for Phase 4.5 implementation.

---

## Implementation Checklist

- [ ] Create `src/tessera/assets.py` with dimension extraction, MIME detection, synthetic content building
- [ ] Add `ASSET_EXTENSIONS` to indexer
- [ ] Update `_discover_files()` to include asset extensions
- [ ] Update `_get_changed_files()` to include asset extensions
- [ ] Implement `index_asset()` method in `IndexerPipeline`
- [ ] Update `index_file()` routing to call `index_asset()` for binary files
- [ ] Add `source_type` parameter filtering to `search()` in server (if not present)
- [ ] Write unit tests for `assets.py`
- [ ] Write integration tests for asset indexing and search
- [ ] Verify all existing tests still pass
- [ ] Test with real image/video/audio files in sample project
- [ ] Document asset indexing in user guide (Phase 5 task)

---

## Follow-Up Questions for Clarification

1. **Search result format:** Should asset results include `file_path` in the returned dict? (Recommend: yes, same as code/document chunks.)
2. **Asset chunk limit:** Any memory concern with indexing projects with 10K+ images? (Likely no, but profile before deployment.)
3. **Browser/IDE integration:** Should assets be displayable/browseable in future UI? (Out of scope for Phase 4.5; Phase 6+ consideration.)

---

## References & Sources

- Research findings: `/Users/danieliser/Toolkit/codemem/docs/plans/phase4.5-media-metadata/research.md`
- Image format specs: PNG, JPEG, GIF, BMP standards
- Python stdlib: `struct`, `mimetypes`, `os`, `hashlib`
- Performance data: `scardine/image_size` benchmarks (struct 10x faster than Pillow for dimensions)
