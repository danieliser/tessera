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
from typing import Optional

logger = logging.getLogger(__name__)

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

CATEGORY_MAP = {
    '.png': 'images', '.jpg': 'images', '.jpeg': 'images', '.gif': 'images',
    '.bmp': 'images', '.webp': 'images', '.ico': 'images', '.tiff': 'images',
    '.heif': 'images', '.heic': 'images', '.avif': 'images', '.svg': 'images',
    '.mp4': 'video', '.mkv': 'video', '.webm': 'video', '.avi': 'video',
    '.mov': 'video', '.flv': 'video',
    '.mp3': 'audio', '.wav': 'audio', '.aac': 'audio', '.flac': 'audio',
    '.m4a': 'audio', '.opus': 'audio',
    '.woff': 'fonts', '.woff2': 'fonts', '.ttf': 'fonts', '.otf': 'fonts',
    '.zip': 'archives', '.tar': 'archives', '.gz': 'archives',
    '.rar': 'archives', '.7z': 'archives',
}

SUPPLEMENTAL_MIME_MAP = {
    '.webp': 'image/webp',
    '.avif': 'image/avif',
    '.heif': 'image/heif',
    '.heic': 'image/heic',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.flac': 'audio/flac',
    '.opus': 'audio/opus',
    '.mkv': 'video/x-matroska',
    '.webm': 'video/webm',
    '.m4a': 'audio/mp4',
    '.7z': 'application/x-7z-compressed',
}


def is_asset_file(path: str) -> bool:
    """Return True if the file extension is a known asset type."""
    ext = Path(path).suffix.lower()
    return ext in ASSET_EXTENSIONS


def get_image_dimensions(path: str) -> Optional[tuple[int, int]]:
    """Read image dimensions from file headers without loading the full image.

    Supports PNG, JPEG, GIF, and BMP formats.
    Returns (width, height) or None if unsupported/malformed.
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


def get_asset_metadata(path: str) -> dict:
    """Return metadata dict for an asset file.

    Returns:
        dict with keys: mime_type, category, file_size, dimensions
    """
    ext = Path(path).suffix.lower()

    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = SUPPLEMENTAL_MIME_MAP.get(ext, 'application/octet-stream')

    category = CATEGORY_MAP.get(ext, 'other')

    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = 0

    dimensions = get_image_dimensions(path) if category == 'images' else None

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


def build_synthetic_content(rel_path: str, metadata: dict) -> str:
    """Build an FTS5-searchable string for an asset file.

    Format: "filename.ext assets category mime/type WIDTHxHEIGHT SIZE"
    The parent directory name is included for context.

    Example: "logo.png assets images image/png 1200x800 45KB"
    """
    p = Path(rel_path)
    filename = p.name
    parent = p.parent.name if p.parent.name and p.parent.name != '.' else ''

    category = metadata.get('category', 'other')
    mime_type = metadata.get('mime_type', 'application/octet-stream')
    file_size = metadata.get('file_size', 0)
    dimensions = metadata.get('dimensions')

    parts = [filename, 'assets', category, mime_type]

    if dimensions:
        parts.append(f"{dimensions[0]}x{dimensions[1]}")

    parts.append(_format_size(file_size))

    if parent:
        parts.append(parent)

    return ' '.join(parts)
