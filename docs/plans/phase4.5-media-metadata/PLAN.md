# Plan: Phase 4.5 — Media & Binary File Metadata Catalog
**Date:** 2026-02-28
**Tier:** Quick
**Status:** Approved

## Executive Summary

Phase 4.5 makes binary files (images, video, audio, fonts, archives) discoverable via Tessera's search by indexing their metadata as FTS5 entries. A new `src/tessera/assets.py` module extracts image dimensions via stdlib `struct`, detects MIME types, and builds synthetic content strings. Asset discovery integrates into the existing indexer pipeline with zero schema changes, zero new dependencies, and zero new MCP tools. Existing `.tesseraignore` patterns are automatically respected.

## Specification

### Files

| File | Change | LOC |
|------|--------|-----|
| `src/tessera/assets.py` | **New** — dimension extraction (PNG/JPEG/GIF/BMP via `struct`), MIME detection, category mapping, synthetic content builder | ~250 |
| `src/tessera/indexer.py` | Asset extension discovery, `index_asset()` method, routing in `index_file()`, unrecognized binary logging | ~150 |
| `src/tessera/server.py` | `source_type` parameter filtering on search results | ~10 |
| `src/tessera/db.py` | No changes — reuses existing `chunk_meta` columns | 0 |

### Asset Extensions

```python
ASSET_EXTENSIONS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.ico', '.tiff',
    '.heif', '.heic', '.avif', '.svg',
    # Video
    '.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv',
    # Audio
    '.mp3', '.wav', '.aac', '.flac', '.m4a', '.opus',
    # Fonts
    '.woff', '.woff2', '.ttf', '.otf',
    # Archives
    '.zip', '.tar', '.gz', '.rar', '.7z',
}
```

### How It Works

1. File discovery adds `ASSET_EXTENSIONS` to allowed extensions
2. `index_file()` routes asset files to `index_asset()` before document/code checks
3. `index_asset()` extracts metadata (dimensions, MIME type, file size) and creates one FTS5-indexed chunk per asset with synthetic content like `"logo.png assets images image/png 1200x800 45KB"`
4. Stored in existing `chunk_meta` with `source_type='asset'`, `ast_type=category`, `key_path=dimensions`
5. No embedding computation — FTS5 keyword search only
6. Unrecognized binary files logged at DEBUG level for visibility

### Search Integration

Existing `search()` and `doc_search()` tools get optional `source_type` parameter:
- `search("logo")` — returns code, document, AND asset matches
- `search("logo", source_type="asset")` — returns only asset matches
- `search("function", source_type="code")` — excludes assets

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| `struct` over Pillow for dimensions | Zero dependencies, 10x faster, covers PNG/JPEG/GIF/BMP (~95% of real projects) |
| `mimetypes.guess_type()` for MIME | Stdlib, sufficient for metadata context |
| No embeddings for assets | "logo.png image/png 1200x800" has zero semantic value. FTS5 is sufficient. |
| Dual-index `.svg` files | SVG content is indexed as XML text, but filename/path searches (e.g., "icon") need asset metadata too |
| Reuse `chunk_meta` table | Avoids schema migration, keeps data model unified |
| Log unrecognized binaries at DEBUG | Visibility into skipped files without noise |

## Risk Register

| Risk | Mitigation |
|------|-----------|
| Malformed image headers crash `struct.unpack()` | Wrap in try/except, return None, log warning |
| MIME detection misses custom extensions | Supplemental map for common types, fallback to `application/octet-stream` |
| SVG dual-indexing causes result duplication | `source_type` filter disambiguates; different search intents served |
| Search noise from too many asset hits | Users filter with `source_type="code"` or `source_type="document"` |

## Follow-up Items

- WebP dimension extraction (complex VP8 bitstream — Phase 5)
- EXIF/IPTC metadata extraction (Phase 5+)
- OCR from images with text (Phase 5+)
- Content extraction from archives (Phase 5+)
- Magic byte validation for security-critical scenarios (Phase 5+)

## Test Plan

### Unit Tests (`tests/test_assets.py`)
- PNG/JPEG/GIF/BMP dimension extraction (valid + truncated + invalid magic bytes)
- MIME type detection (standard + supplemental + unknown)
- Asset category mapping
- Synthetic content building (with/without dimensions)
- Asset file detection (including `.svg` = True)

### Integration Tests (`tests/test_indexer_assets.py`)
- End-to-end indexing with mixed file types
- Search filtering by `source_type`
- `.tesseraignore` pattern respect
- File change detection (skip unchanged assets)
- Error handling (truncated files, permission errors)
