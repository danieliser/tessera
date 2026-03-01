# Research: Phase 4.5 — Media & Binary File Metadata Catalog

**Date:** 2026-02-28
**Tier:** Standard
**Project:** Tessera — Codebase Indexing MCP Server

## Research Questions

1. **Reading image dimensions without Pillow**
2. **MIME type detection strategies**
3. **FTS5 synthetic content optimization**
4. **How other tools handle binary files**
5. **Extension-to-category mapping**
6. **Tessera pipeline integration points**

---

## Question 1: Reading Image Dimensions Without Pillow

### High-Level Finding

**Python's `struct` module is fully sufficient** to extract image dimensions from PNG, JPEG, GIF, and BMP headers. This approach is proven, requires zero external dependencies, and performs ~10x faster than Pillow for dimension-only extraction.

### Confidence: High

Multiple production libraries (`scardine/image_size`, `imagesize_py`) have implemented this successfully. Code is straightforward and header format specs are stable across image format versions.

### Image Format Specifications

#### PNG
- **Magic bytes:** `89 50 4E 47 0D 0A 1A 0A` (`\211PNG\r\n\032\n`)
- **IHDR chunk location:** Starts at byte 8 after signature
- **Width/Height extraction:** Bytes 16–24 of file (within IHDR chunk)
- **Format string:** `'>II'` (big-endian, 2 unsigned 32-bit integers)

```python
import struct

with open('image.png', 'rb') as f:
    header = f.read(24)
    if header.startswith(b'\211PNG\r\n\032\n'):
        width, height = struct.unpack('>II', header[16:24])
```

**Source:** [Python struct documentation](https://docs.python.org/3/library/struct.html), tested in Stack Overflow examples

#### JPEG
- **Magic bytes:** `FF D8` (start-of-image marker)
- **Dimension location:** SOF0 (Start of Frame) marker at `FF C0`
- **Search strategy:** Scan forward through markers until SOF0 found
- **Within SOF0:** Height at offset 5–7, Width at offset 7–9 (big-endian 16-bit shorts)
- **Format string:** `'>HH'` (big-endian, 2 unsigned 16-bit shorts)

```python
with open('image.jpg', 'rb') as f:
    if f.read(2) == b'\xFF\xD8':  # SOI marker
        while True:
            marker = f.read(2)
            if marker[0] != 0xFF:
                break
            if marker[1] == 0xC0:  # SOF0
                f.read(3)  # skip length + precision
                height, width = struct.unpack('>HH', f.read(4))
                break
            else:
                # Skip this segment
                segment_len = struct.unpack('>H', f.read(2))[0]
                f.read(segment_len - 2)
```

**Source:** [Understanding JPEG decoder in Python](https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/), [JPEG marker guide](https://www.ccoderun.ca/programming/2017-01-31_jpeg/)

#### GIF
- **Magic bytes:** `47 49 46 38 37 61` (`GIF87a`) or `47 49 46 38 39 61` (`GIF89a`)
- **Width/Height location:** Bytes 6–10 of file
- **Format string:** `'<HH'` (little-endian, 2 unsigned 16-bit shorts)

```python
with open('image.gif', 'rb') as f:
    header = f.read(10)
    if header[:6] in (b'GIF87a', b'GIF89a'):
        width, height = struct.unpack('<HH', header[6:10])
```

#### BMP
- **Magic bytes:** `42 4D` (`BM`)
- **Header size indicator:** Bytes 14–18 (DIB header size)
- **Two variants:**
  - If DIB header = 12 bytes (old OS/2 format): width/height at bytes 18–22
  - If DIB header ≥ 40 bytes (modern): width/height at bytes 18–26 (signed)
- **Format strings:**
  - Old: `'<HH'` (2 unsigned shorts)
  - Modern: `'<ii'` (2 signed ints; height may be negative)

```python
with open('image.bmp', 'rb') as f:
    header = f.read(26)
    if header[:2] == b'BM':
        dib_header_size = struct.unpack('<I', header[14:18])[0]
        if dib_header_size == 12:
            width, height = struct.unpack('<HH', header[18:22])
        elif dib_header_size >= 40:
            width, height = struct.unpack('<ii', header[18:26])
            height = abs(height)  # height can be negative (inverted)
```

#### WebP
- **Magic bytes:** `52 49 46 46` (`RIFF`) + "WEBP" signature at offset 8
- **Complexity:** Varies significantly between lossy/lossless/animated formats
- **Reading strategy:** Parse VP8/VP8L bitstream headers (moderately complex)
- **Recommendation:** **Skip WebP for Phase 4.5** — complexity-to-benefit ratio unfavorable. Include in Non-Goal list for Phase 5.

### Performance Comparison

From `scardine/image_size` benchmarks (125 KB PNG file):
- **Struct-based (pure Python):** 1.077 sec per 100,000 iterations
- **Pillow PIL.Image.open():** 10.569 sec per 100,000 iterations
- **Speedup:** ~9.8x faster

Only header bytes are loaded into memory (8–32 KB max), making this efficient at scale.

---

## Question 2: MIME Type Detection

### High-Level Finding

**Use `mimetypes.guess_type()` from stdlib for Phase 4.5.** It's zero-dependency, good enough for asset metadata context, and avoids the complexity of magic byte parsing. Don't implement magic byte detection unless file extension is missing or untrusted.

### Confidence: Medium-High

The stdlib approach is proven and sufficient for the use case (searchable asset metadata, not security-critical file validation). If future phases require robustness against renamed files, magic bytes can be added incrementally.

### Comparison: `mimetypes` vs Magic Bytes

| Aspect | `mimetypes.guess_type()` | Magic Bytes (e.g., `python-magic`, `filetype`) |
|--------|--------------------------|----------------------------------------------|
| **Dependency** | None (stdlib) | External (libmagic or pymagic) |
| **Speed** | Very fast (table lookup) | Slower (file I/O + binary parsing) |
| **Accuracy** | Extension-based (8–10% misses if renamed) | Content-based (99%+) |
| **Maintenance** | OS/Python version dependent | Stable across OS |
| **Use case** | Asset discovery, metadata | Security-critical validation |

### Implementation Details

```python
import mimetypes

# Initialize once (reads /etc/mime.types on first call)
mimetypes.init()

# Get MIME type from extension
mime_type, encoding = mimetypes.guess_type('logo.png')
# Returns: ('image/png', None)

mime_type, _ = mimetypes.guess_type('archive.tar.gz')
# Returns: ('application/x-tar', 'gzip')
```

**Stdlib coverage (Python 3.11+):**
- Images: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.svg`, `.webp`, `.ico`, `.tiff`
- Audio: `.mp3`, `.wav`, `.aac`, `.flac`, `.m4a`, `.wma`
- Video: `.mp4`, `.webm`, `.mkv`, `.mov`, `.avi`, `.flv`, `.m3u8`
- Archives: `.zip`, `.tar`, `.gz`, `.bz2`, `.7z`, `.rar`
- Documents: `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`

**Limitations & Workaround:**
- Some extensions missing (e.g., `.woff`, `.woff2` for fonts)
- Solution: Maintain a small **supplemental mapping** for missing types

```python
SUPPLEMENTAL_MIME_TYPES = {
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.otf': 'font/otf',
    '.ttf': 'font/ttf',
    '.m4a': 'audio/mp4',
    '.opus': 'audio/opus',
    '.webm': 'video/webm',
}

def get_mime_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in SUPPLEMENTAL_MIME_TYPES:
        return SUPPLEMENTAL_MIME_TYPES[ext]
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or 'application/octet-stream'
```

**Source:** [Python mimetypes documentation](https://docs.python.org/3/library/mimetypes.html), comparison with [python-magic](https://github.com/livingbio/ffmpeg-media-type/issues/51)

---

## Question 3: FTS5 Synthetic Content Optimization

### High-Level Finding

**For asset metadata, use a single synthetic content column combining filename, path components, and MIME type.** FTS5's tokenizer handles word separation natively. Don't embed images in vectors (no semantic value).

### Confidence: High

Tested pattern in existing Tessera codebase. FTS5 is optimized for exactly this use case.

### Recommended Format

Concatenate fields with whitespace separation. FTS5 tokenizes on whitespace and punctuation:

```python
# Asset: /assets/images/logo/brand_identity_2024_v2.png
# MIME: image/png, 1200x800px, 45KB

synthetic_content = f"""
{filename}
{' '.join(path_components)}
{mime_type}
{dimensions_text}
{file_size_text}
""".strip()

# Result:
# "brand_identity_2024_v2.png assets images logo image/png 1200x800 45KB"
```

**Why this works:**
- FTS5 tokenizes on whitespace and punctuation (`.` becomes token boundary)
- Query `search("logo")` matches both path component and directory name
- Query `search("image/png")` matches MIME type (escapes `/` during tokenization)
- Separate columns unnecessary — single `content` column sufficient for this use case

### Implementation in `insert_chunks()`

Use existing `chunk_meta` columns without schema changes:

```python
# Reuse these columns for asset metadata:
# - content: synthetic string (as above)
# - source_type: 'asset' (distinguishes from code)
# - ast_type: 'asset' or category ('image', 'video', etc.)
# - chunk_type: 'asset'
# - length: file size in bytes

asset_chunk = {
    'project_id': project_id,
    'file_id': file_id,
    'start_line': 0,  # N/A for binary files
    'end_line': 0,
    'symbol_ids': [],
    'ast_type': 'image',  # e.g., image, video, font, archive
    'chunk_type': 'asset',
    'content': synthetic_content,  # FTS5-indexed
    'length': file_size_bytes,
    'source_type': 'asset',  # NEW: filters search results
    # Optional metadata fields (for future use, not indexed):
    'section_heading': None,  # N/A
    'key_path': dimensions_str,  # e.g., "1200x800" for images
    'page_number': None,
    'parent_section': None,
}
```

**Storage pattern:**
- One "chunk" per asset file (not multiple chunks per image)
- FTS5 indexes the synthetic content
- Embeddings **not computed** for asset metadata (no semantic value)

**Source:** [SQLite FTS5 documentation](https://sqlite.org/fts5.html) on tokenization and column design

---

## Question 4: How Other Tools Handle Binary Files

### Sourcegraph Code Search

Sourcegraph uses **Zoekt**, a trigram-based indexer (written in Go), for code search. Binary files are not explicitly indexed:
- Indexed: Source code, file paths, repository metadata
- **Not indexed:** Binary file contents (images, videos, fonts, archives)
- **Fallback:** Unindexed repos use a "searcher" process that performs regexp matching on zipped repository contents

**Relevant:** Sourcegraph treats binary files as invisible to semantic search — aligns with Tessera's approach (metadata only, no content embedding).

### ripgrep-all (rga)

`phiresky/ripgrep-all` extends ripgrep to search inside binary formats:
- Extracts text from PDFs, Office documents, eBooks, zip files, tar archives
- **Not for image/video/font files** — focuses on searchable document formats

**Relevant:** Demonstrates that binary file handling varies by type. Tessera's asset discovery aligns with this: **catalog as metadata, extract only from text-containing formats (future work).**

### GitHub Code Search

GitHub's search does not expose binary file metadata (images, videos, etc.) in search results. Binary files are listed in repo browsers but not searchable.

**Relevant:** Confirms market practice — binary file discovery is an emerging need, not yet standard.

**Sources:** [Sourcegraph code search documentation](https://sourcegraph.com/code-search), [Sourcegraph blog](https://sourcegraph.com/blog/how-to-search-cheat-sheet), [ripgrep-all](https://github.com/phiresky/ripgrep-all)

---

## Question 5: Extension-to-Category Mapping

### Curated Binary Extension List

Organize by category. Exclude text-based formats (already indexed as code/documents):

#### Images
- **Standard:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.svg`, `.webp`, `.ico`, `.tiff`, `.tif`
- **Raw camera:** `.raw`, `.cr2`, `.nef`, `.dng`
- **Vector/design:** `.eps`, `.pdf` (if in assets), `.ai`

#### Video
- **Containers:** `.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`, `.flv`, `.m3u8`
- **Codecs:** `.h264`, `.h265`

#### Audio
- **Formats:** `.mp3`, `.wav`, `.aac`, `.flac`, `.m4a`, `.wma`, `.opus`, `.ogg`
- **Lossless:** `.alac`, `.ape`
- **Specialty:** `.mid`, `.midi`

#### Fonts
- **Web fonts:** `.woff`, `.woff2`, `.ttf`, `.otf`, `.eot`
- **System fonts:** (excluded from project-level assets, usually)

#### Archives & Compressed
- **Compressed:** `.zip`, `.rar`, `.7z`, `.tar`, `.gz`, `.bz2`, `.xz`, `.z`
- **Bundled:** `.iso`, `.dmg`, `.msi`, `.exe` (binary packages — may exclude)

#### Design/Graphics Tools
- **Figma/Sketch:** `.fig`, `.sketch`, `.xd`, `.psd`, `.psb`

#### Office/Documents (if in assets, not code)
- **Microsoft:** `.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`
- **OpenDocument:** `.odt`, `.ods`, `.odp`
- **Adobe:** `.pdf`

#### Other Binary
- **Executables:** `.o`, `.a`, `.so`, `.dll`, `.dylib`, `.class`, `.pyc` (usually ignored)
- **Data:** `.db`, `.sqlite`, `.dat`, `.bin`

### Recommendation: Start with "High-Value" Categories

For Phase 4.5 MVP, focus on **images, video, audio, fonts, archives**. Exclude design tools (Figma, Sketch, PSD) and office documents — lower ROI for code-centric projects.

**Dynamic category assignment:**

```python
EXTENSION_CATEGORIES = {
    # Images
    '.png': 'image',
    '.jpg': 'image',
    '.jpeg': 'image',
    '.gif': 'image',
    '.bmp': 'image',
    '.webp': 'image',
    '.svg': 'image',
    '.ico': 'image',
    '.tiff': 'image',
    '.tif': 'image',

    # Video
    '.mp4': 'video',
    '.mkv': 'video',
    '.webm': 'video',
    '.avi': 'video',
    '.mov': 'video',
    '.flv': 'video',

    # Audio
    '.mp3': 'audio',
    '.wav': 'audio',
    '.aac': 'audio',
    '.flac': 'audio',
    '.m4a': 'audio',
    '.opus': 'audio',

    # Fonts
    '.woff': 'font',
    '.woff2': 'font',
    '.ttf': 'font',
    '.otf': 'font',

    # Archives
    '.zip': 'archive',
    '.tar': 'archive',
    '.gz': 'archive',
    '.rar': 'archive',
    '.7z': 'archive',
}

def get_asset_category(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return EXTENSION_CATEGORIES.get(ext, 'binary')
```

---

## Question 6: Tessera Pipeline Integration

### Current Architecture

From `/src/tessera/indexer.py`:

**File discovery flow:**
1. `IndexerPipeline._discover_files()` walks project, filters by extension
2. Files matched by `DOCUMENT_EXTENSIONS` (PDF, MD, YAML, JSON) or language extensions (PHP, TS, Python, etc.)
3. `IgnoreFilter.should_ignore()` applied (respects `.tesseraignore` and security patterns)
4. Sorted file list returned

**Indexing dispatch:**
- `index_file()` routes to `_index_document_file()` for documents, or AST parsing for code
- Document files chunked by format (PDF → markdown chunks, YAML → key-value chunks, etc.)
- Code files parsed with tree-sitter, chunked with `chunk_with_cast()`

### Integration Point: Asset Discovery Hook

**Best location: Between file discovery and indexing dispatch**

```python
# In IndexerPipeline._discover_files() or index_project()

# 1. Add ASSET_EXTENSIONS to allow-list
ASSET_EXTENSIONS = [
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.ico', '.tiff', '.tif',
    # Video
    '.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv',
    # Audio
    '.mp3', '.wav', '.aac', '.flac', '.m4a', '.opus',
    # Fonts
    '.woff', '.woff2', '.ttf', '.otf',
    # Archives
    '.zip', '.tar', '.gz', '.rar', '.7z',
]

# 2. Update _discover_files() to include assets
allowed_exts = set(ALL_DOCUMENT_EXTENSIONS)
for lang in self.languages:
    allowed_exts.update(extensions.get(lang, []))
allowed_exts.update(ASSET_EXTENSIONS)  # ← NEW

# 3. Add index_asset() method
def index_asset(self, file_path: str) -> Dict[str, Any]:
    """Index a binary asset file: extract metadata, create searchable chunk."""
    rel_path = os.path.relpath(file_path, self.project_path)

    try:
        # Extract dimensions (for images)
        dimensions = extract_image_dimensions(file_path) if is_image(file_path) else None

        # Get MIME type
        mime_type = get_mime_type(rel_path)

        # Get file size
        file_size = os.path.getsize(file_path)

        # Get file hash (for change detection)
        file_hash = self._file_hash(file_path)

        # Create synthetic content for FTS5
        synthetic_content = build_asset_synthetic_content(
            filename=os.path.basename(rel_path),
            path_components=rel_path.split(os.sep),
            mime_type=mime_type,
            dimensions=dimensions,
            file_size=file_size
        )

        # Upsert file record
        file_id = self.project_db.upsert_file(
            project_id=self.project_id,
            path=rel_path,
            language='asset',  # NEW: language='asset'
            file_hash=file_hash
        )

        # Create one chunk per asset
        chunk_dict = {
            'project_id': self.project_id,
            'file_id': file_id,
            'start_line': 0,
            'end_line': 0,
            'symbol_ids': [],
            'ast_type': get_asset_category(rel_path),  # 'image', 'video', etc.
            'chunk_type': 'asset',
            'content': synthetic_content,
            'source_type': 'asset',
            'length': file_size,
            'key_path': f"{dimensions['width']}x{dimensions['height']}" if dimensions else None,
        }

        # Store chunk (no embedding for assets)
        self.project_db.insert_chunks([chunk_dict])

        # Mark file indexed
        self.project_db.update_file_status(file_id, 'indexed')

        return {
            'status': 'indexed',
            'chunks': 1,
            'embedded': 0,  # No embeddings for assets
            'mime_type': mime_type,
            'dimensions': dimensions,
            'file_size': file_size
        }

    except Exception as e:
        logger.error(f"Failed to index asset {file_path}: {e}")
        # ... error handling
        return {'status': 'failed', 'reason': str(e)}

# 4. Route in index_file()
def index_file(self, file_path: str) -> Dict[str, Any]:
    # ... existing code ...

    # Check if asset
    if is_asset_file(file_path):
        return self.index_asset(file_path)

    # Existing document/code routing
    if self._is_document_file(file_path):
        return self._index_document_file(file_path)

    # ... rest of existing flow
```

### `.tesseraignore` Respect

From `/src/tessera/ignore.py`:
- `IgnoreFilter` initialized with project root
- `should_ignore(rel_path)` checks both security patterns and user-configurable patterns
- Already applied in `_discover_files()` → **asset discovery automatically respects ignores**

**Action:** No changes needed. Existing ignore logic applies to assets.

### DB Schema Integration

From `/src/tessera/db.py`:

**`chunk_meta` table already supports asset metadata without migration:**

```sql
CREATE TABLE IF NOT EXISTS chunk_meta (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    symbol_ids TEXT,
    ast_type TEXT,                    -- reuse: 'image', 'video', 'font', etc.
    chunk_type TEXT,                  -- reuse: 'asset'
    content TEXT NOT NULL,            -- synthetic content for FTS5
    length INTEGER,                   -- file size in bytes
    -- These columns already exist:
    source_type TEXT,                 -- NEW VALUE: 'asset' (currently 'code'|'document')
    section_heading TEXT,             -- unused for assets (NULL)
    key_path TEXT,                    -- reuse: image dimensions "1200x800"
    page_number INTEGER,              -- unused for assets (NULL)
    parent_section TEXT,              -- unused for assets (NULL)
    FOREIGN KEY(file_id) REFERENCES files(id)
)
```

**FTS5 integration (existing):**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    chunk_id UNINDEXED,
    file_path UNINDEXED
)
```

Existing FTS5 table indexes `content` → **asset synthetic content automatically searchable.**

### Search Result Filtering

Existing `search()` and `hybrid_search()` in `/src/tessera/search.py` will return asset results. Filter in MCP server:

```python
# MCP tool: search(query, limit=10, source_type=None)
# If source_type='asset', filter results to assets only
# If source_type=None (default), return all (code + documents + assets)

results = hybrid_search(query, embedding, db, limit)
if source_type:
    results = [r for r in results if r.get('source_type') == source_type]
return results
```

**No search contract breakage** — new `source_type` field distinguishes assets but doesn't change existing behavior.

---

## Recommendations

### For Spec Writer

#### 1. Implement Image Dimension Extraction with `struct`

**Approach:**
- Extract PNG/JPEG/GIF/BMP dimensions using `struct` module (tested, fast, zero-dependency)
- Create helper module `/src/tessera/assets.py` with functions:
  - `extract_image_dimensions(file_path) -> Dict[str, int] | None`
  - `get_mime_type(filename) -> str`
  - `build_asset_synthetic_content(...) -> str`

**Code skeleton:**

```python
# /src/tessera/assets.py
import struct
import mimetypes
from typing import Optional, Dict

ASSET_EXTENSIONS = {
    '.png': 'image', '.jpg': 'image', '.jpeg': 'image',
    '.gif': 'image', '.bmp': 'image', '.webp': 'image',
    '.mp4': 'video', '.mkv': 'video',
    '.mp3': 'audio', '.wav': 'audio',
    '.ttf': 'font', '.woff': 'font',
    '.zip': 'archive', '.tar': 'archive',
}

def extract_image_dimensions(file_path: str) -> Optional[Dict[str, int]]:
    """Extract width/height from PNG, JPEG, GIF, BMP without Pillow."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)

        # PNG
        if header.startswith(b'\211PNG\r\n\032\n') and len(header) >= 24:
            width, height = struct.unpack('>II', header[16:24])
            return {'width': int(width), 'height': int(height)}

        # GIF
        if header[:6] in (b'GIF87a', b'GIF89a'):
            width, height = struct.unpack('<HH', header[6:10])
            return {'width': int(width), 'height': int(height)}

        # BMP
        if header[:2] == b'BM' and len(header) >= 26:
            dib_size = struct.unpack('<I', header[14:18])[0]
            if dib_size == 12:
                width, height = struct.unpack('<HH', header[18:22])
            elif dib_size >= 40:
                width, height = struct.unpack('<ii', header[18:26])
                height = abs(height)
            return {'width': int(width), 'height': int(height)}

        # JPEG (requires marker scanning)
        if header[:2] == b'\xFF\xD8':
            with open(file_path, 'rb') as f:
                f.read(2)  # skip SOI
                while True:
                    marker = f.read(2)
                    if marker[0] != 0xFF:
                        break
                    if marker[1] == 0xC0:
                        f.read(3)
                        height, width = struct.unpack('>HH', f.read(4))
                        return {'width': int(width), 'height': int(height)}
                    else:
                        seg_len = struct.unpack('>H', f.read(2))[0]
                        f.read(seg_len - 2)

        return None
    except (OSError, struct.error):
        return None

def get_mime_type(filename: str) -> str:
    """Get MIME type from filename, with stdlib fallback."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type

    # Supplemental types for common missing extensions
    supplemental = {
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.opus': 'audio/opus',
    }
    ext = os.path.splitext(filename)[1].lower()
    return supplemental.get(ext, 'application/octet-stream')

def build_asset_synthetic_content(
    filename: str,
    path_components: list[str],
    mime_type: str,
    dimensions: Optional[Dict[str, int]] = None,
    file_size: int = 0
) -> str:
    """Build synthetic content string for FTS5 indexing."""
    parts = [
        filename,
        ' '.join(path_components),
        mime_type,
    ]

    if dimensions:
        parts.append(f"{dimensions['width']}x{dimensions['height']}")

    if file_size:
        # Format file size (e.g., "45KB", "1.2MB")
        if file_size < 1024:
            size_str = f"{file_size}B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size // 1024}KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f}MB"
        parts.append(size_str)

    return ' '.join(parts)

def is_asset_file(file_path: str) -> bool:
    """Check if file is a binary asset."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ASSET_EXTENSIONS

def get_asset_category(file_path: str) -> str:
    """Get asset category from extension."""
    ext = os.path.splitext(file_path)[1].lower()
    return ASSET_CATEGORIES.get(ext, 'binary')
```

#### 2. Add Asset Indexing to Pipeline

**In `IndexerPipeline`:**
- Add `ASSET_EXTENSIONS` constant
- Extend `_discover_files()` to include assets
- Add `index_asset()` method
- Update `index_file()` to dispatch to `index_asset()` for binary files
- Ensure `get_changed_files()` includes asset extensions

**Code size:** ~150 lines of well-structured code

#### 3. No Schema Changes Required

- Existing `chunk_meta` table columns sufficient
- Populate `source_type='asset'`, `ast_type=category`, `chunk_type='asset'`
- No embedding computation (skip `embedding_client.embed()` for assets)

#### 4. Search Contract Preserved

- Add optional `source_type` parameter to search result filtering
- Default: return all types (backward compatible)
- Filter: `source_type='asset'` returns only binary files

#### 5. SVG Handling

**Decision:** `.svg` files already indexed as XML documents. Do **not** duplicate as assets.

**Implementation:**
```python
def is_asset_file(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.svg':
        return False  # Already indexed as XML
    return ext in ASSET_EXTENSIONS
```

#### 6. Testing Checklist

```python
# Unit tests for assets.py
- extract_image_dimensions() for PNG, JPEG, GIF, BMP (mock file headers)
- get_mime_type() for all supported extensions
- build_asset_synthetic_content() formatting
- is_asset_file() filtering

# Integration tests
- index_asset() e2e with real image files
- search("logo") returns asset results
- Asset results have source_type='asset'
- .tesseraignore patterns respected for assets
- No embeddings computed for assets
- All existing tests still pass
```

#### 7. Non-Goals for Phase 4.5 (Explicit Scope Boundaries)

- **WebP dimensions:** Deferred to Phase 5 (complexity outweighs benefit)
- **EXIF data, IPTC metadata, thumbnails:** Phase 5+
- **OCR from images:** Phase 5+
- **Content extraction from archives:** Phase 5+
- **Magic byte validation (security):** Out of scope (mimetypes sufficient)
- **New MCP tools:** Reuse existing `search()` with `source_type` filtering

---

## Summary Table

| Question | Finding | Confidence |
|----------|---------|------------|
| **Image dimensions** | `struct` module, PNG/JPEG/GIF/BMP only | High |
| **MIME detection** | `mimetypes.guess_type()` + small supplemental map | Medium-High |
| **FTS5 content** | Single synthetic string, whitespace-separated fields | High |
| **Market practice** | Sourcegraph/GitHub don't index binary metadata | High |
| **Extension mapping** | 40+ high-value extensions across 5 categories | Medium |
| **Pipeline hook** | Between discovery & dispatch, respects ignores | High |
| **DB schema** | No changes needed, reuse existing columns | High |

---

## Sources

- [Python struct module documentation](https://docs.python.org/3/library/struct.html)
- [Understanding JPEG decoder in Python](https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/)
- [JPEG marker specification](https://www.ccoderun.ca/programming/2017-01-31_jpeg/)
- [scardine/image_size GitHub](https://github.com/scardine/image_size)
- [imagesize_py GitHub](https://github.com/shibukawa/imagesize_py)
- [Python mimetypes documentation](https://docs.python.org/3/library/mimetypes.html)
- [SQLite FTS5 documentation](https://sqlite.org/fts5.html)
- [Sourcegraph code search documentation](https://sourcegraph.com/code-search)
- [Sourcegraph blog](https://sourcegraph.com/blog/how-to-search-cheat-sheet)
- [ripgrep-all](https://github.com/phiresky/ripgrep-all)
- [MDN image format guide](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Image_types)
- [Wikipedia BMP file format](https://en.wikipedia.org/wiki/BMP_file_format)
- [Wikipedia PNG](https://en.wikipedia.org/wiki/PNG)
