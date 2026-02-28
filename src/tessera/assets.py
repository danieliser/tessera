"""Asset metadata extraction for non-code files.

Handles images, video, audio, fonts, and archives.
Provides MIME detection, image dimension extraction, and
FTS5-searchable synthetic content generation.
"""

import logging
import mimetypes
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

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


def is_asset_file(file_path: str) -> bool:
    """Return True if the file extension is a known asset type."""
    ext = Path(file_path).suffix.lower()
    return ext in ASSET_EXTENSIONS


def get_asset_category(file_path: str) -> str:
    """Return the asset category for a file path.

    Returns one of: 'image', 'video', 'audio', 'font', 'archive', or 'binary'.
    """
    ext = Path(file_path).suffix.lower()
    return ASSET_EXTENSIONS.get(ext, 'binary')


def get_mime_type(filename: str) -> str:
    """Return the MIME type for a filename.

    Checks SUPPLEMENTAL_MIME_TYPES first for extensions that mimetypes.guess_type()
    gets wrong or misses, then falls back to mimetypes, then 'application/octet-stream'.
    """
    ext = Path(filename).suffix.lower()
    if ext in SUPPLEMENTAL_MIME_TYPES:
        return SUPPLEMENTAL_MIME_TYPES[ext]
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type if mime_type is not None else 'application/octet-stream'


def _get_image_dimensions(path: str) -> Optional[tuple[int, int]]:
    """Read image dimensions from file headers using struct (internal helper).

    Supports PNG, JPEG, GIF, and BMP formats.
    Returns (width, height) tuple or None if unsupported/malformed.
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(26)
    except OSError:
        return None

    # PNG: 8-byte magic, then IHDR chunk: 4-byte length, 4-byte 'IHDR',
    # then width (4 bytes big-endian), height (4 bytes big-endian)
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        try:
            width, height = struct.unpack('>II', header[16:24])
            return (width, height)
        except struct.error:
            logger.warning("Malformed PNG header in %s", path)
            return None

    # JPEG: scan for SOF markers
    if header[:2] == b'\xff\xd8':
        return _jpeg_dimensions(path)

    # GIF: 6-byte signature ('GIF87a' or 'GIF89a'), then width/height as LE uint16
    if header[:6] in (b'GIF87a', b'GIF89a'):
        try:
            width, height = struct.unpack('<HH', header[6:10])
            return (width, height)
        except struct.error:
            logger.warning("Malformed GIF header in %s", path)
            return None

    # BMP: 2-byte 'BM', 4-byte file size, 4 reserved bytes, 4-byte offset,
    # then DIB header: 4-byte size, then width/height as LE int32
    if header[:2] == b'BM':
        try:
            width, height = struct.unpack('<ii', header[18:26])
            return (width, abs(height))
        except struct.error:
            logger.warning("Malformed BMP header in %s", path)
            return None

    return None


def _jpeg_dimensions(path: str) -> Optional[tuple[int, int]]:
    """Scan JPEG markers to find SOF (Start of Frame) with image dimensions."""
    try:
        with open(path, 'rb') as f:
            # Skip SOI marker (0xFF 0xD8)
            f.read(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    return None
                if marker[0] != 0xFF:
                    return None
                marker_type = marker[1]
                # SOF markers: 0xC0–0xCF, excluding 0xC4 (DHT) and 0xCC (DAC)
                if 0xC0 <= marker_type <= 0xCF and marker_type not in (0xC4, 0xCC):
                    # Skip 2-byte length and 1-byte precision
                    f.read(3)
                    dims = f.read(4)
                    if len(dims) < 4:
                        return None
                    height, width = struct.unpack('>HH', dims)
                    return (width, height)
                else:
                    # Skip segment: read 2-byte length (includes the 2 length bytes)
                    length_bytes = f.read(2)
                    if len(length_bytes) < 2:
                        return None
                    length = struct.unpack('>H', length_bytes)[0]
                    f.seek(length - 2, 1)
    except (OSError, struct.error):
        logger.warning("Malformed JPEG in %s", path)
        return None


def extract_image_dimensions(file_path: str) -> Optional[Dict[str, int]]:
    """Extract image dimensions from PNG, JPEG, GIF, or BMP file headers.

    Uses stdlib struct module — zero external dependencies.

    Returns:
        {'width': int, 'height': int} or None on error or unsupported format.
    """
    result = _get_image_dimensions(file_path)
    if result is None:
        return None
    return {'width': result[0], 'height': result[1]}


def get_asset_metadata(path: str) -> dict:
    """Return metadata dict for an asset file.

    Returns:
        dict with keys: mime_type, category, file_size, dimensions
    """
    category = get_asset_category(path)
    mime_type = get_mime_type(os.path.basename(path))

    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = 0

    dimensions = extract_image_dimensions(path) if category == 'image' else None

    return {
        'mime_type': mime_type,
        'category': category,
        'file_size': file_size,
        'dimensions': dimensions,
    }


def _format_size(size_bytes: int) -> str:
    """Format file size as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    kb = size_bytes / 1024
    if kb < 1024:
        return f"{kb:.0f}KB"
    mb = kb / 1024
    return f"{mb:.0f}MB"


def build_asset_synthetic_content(
    filename: str,
    path_components: List[str],
    mime_type: str,
    dimensions: Optional[Dict[str, int]] = None,
    file_size: int = 0,
) -> str:
    """Build an FTS5-searchable string for an asset file.

    Combines filename, path components, MIME type, dimensions, and size
    into a whitespace-separated string suitable for FTS5 indexing.

    Example: "logo.png assets images image/png 1200x800 45KB"
    """
    parts = [filename] + list(path_components)
    parts.append(mime_type)

    if dimensions:
        parts.append(f"{dimensions['width']}x{dimensions['height']}")

    parts.append(_format_size(file_size))

    return ' '.join(parts)
