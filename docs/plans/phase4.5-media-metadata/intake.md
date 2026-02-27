# Phase 4.5 — Media & Binary File Metadata Catalog

**Date:** 2026-02-27
**Status:** Placeholder (not yet specced)
**Origin:** Identified during Phase 4 live testing — binary files (images, videos, fonts, archives) are completely invisible to search.

## Problem

Tessera only indexes files it can extract text from. Binary files like logos, screenshots, fonts, videos, and archives are silently skipped during discovery. An AI agent searching for "logo" or "hero image" gets zero results even when those assets exist in the project.

The goal is instantaneous search across everything the agent has access to — not just code and docs.

## Proposed Approach

Index binary files as **metadata-only entries** — no content extraction, but enough information to make them discoverable via search.

### What gets stored per binary file

- **File path** (relative to project root)
- **Filename** (searchable — `logo.svg`, `hero-banner.png`)
- **Extension / MIME type** (filterable — `image/png`, `video/mp4`)
- **File size** (bytes)
- **Image dimensions** (width x height, for image formats — via PIL or similar)
- **Synthetic content string** for FTS5: e.g., `"logo.png image/png 1200x800 assets/images/logo.png"` so keyword search can find it

### New source_type

`source_type = 'asset'` — distinguishes from code/document chunks. Search results with `source_type='asset'` have no extractable content, just metadata.

## Design Decisions Needed

1. **Search result contract change**: Currently `content` is always real extracted text. Asset entries would have synthetic metadata as `content`. Does this break consumer assumptions? Should there be a separate `asset_search()` tool, or should `search()` and `doc_search()` include assets by default?

2. **`trusted` field semantics**: Code chunks are `trusted=True`, doc chunks are `trusted=False`. Assets have no content to trust/distrust — what's the right value?

3. **Image dimensions**: Requires a dependency (Pillow or similar) for reading image metadata. Worth it? Or just store path/size/mime?

4. **Which extensions to catalog**: Need a curated list. Candidates:
   - Images: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.ico`, `.bmp`, `.tiff`, `.avif`
   - Vector: `.svg` (already indexed as XML text — dual-index as asset too?)
   - Video: `.mp4`, `.webm`, `.mov`, `.avi`
   - Audio: `.mp3`, `.wav`, `.ogg`, `.m4a`
   - Fonts: `.woff`, `.woff2`, `.ttf`, `.otf`, `.eot`
   - Archives: `.zip`, `.tar`, `.gz`, `.7z`
   - Documents: `.docx`, `.pptx`, `.xlsx` (metadata only, no content extraction yet)
   - Design: `.psd`, `.sketch`, `.fig`, `.ai`

5. **Schema impact**: No new columns needed — `chunk_meta` already has `source_type`. But the `content` field would contain synthetic metadata instead of real text. Alternatively, add an `asset_meta` table with structured fields (mime_type, dimensions, file_size) and a single FTS-indexed chunk per file.

6. **Embedding**: Should asset metadata strings get embedded? The synthetic content is short and keyword-heavy — FTS5 might be sufficient. Embedding "logo.png image/png 1200x800" doesn't produce a meaningful semantic vector.

## Estimated Scope

- ~200-300 LOC production
- ~100 LOC tests
- 0-1 new dependencies (Pillow for image dimensions, optional)
- No schema migration needed if using existing chunk_meta
- One new schema migration if adding asset_meta table

## Relationship to Other Phases

- **Phase 4** (just completed): Text format indexing. This extends discovery to non-text files.
- **Phase 6+** (architecture plan): DOCX/Confluence content extraction. Orthogonal — that's about extracting text from complex formats, this is about cataloging files that have no extractable text.
- **Future**: OCR for images with text, video thumbnail extraction, EXIF metadata — all build on having the asset catalog in place.
