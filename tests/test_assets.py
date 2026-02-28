"""Tests for src/tessera/assets.py"""

import struct
from pathlib import Path

import pytest

from tessera.assets import (
    ASSET_EXTENSIONS,
    CATEGORY_MAP,
    SUPPLEMENTAL_MIME_MAP,
    build_synthetic_content,
    get_asset_metadata,
    get_image_dimensions,
    is_asset_file,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal valid binary headers
# ---------------------------------------------------------------------------

def _make_png(width: int, height: int) -> bytes:
    """Minimal PNG: 8-byte magic + IHDR chunk (width/height only)."""
    magic = b'\x89PNG\r\n\x1a\n'
    # IHDR chunk: 4-byte length=13, 4-byte 'IHDR', 13 bytes data, 4-byte CRC (zeroed)
    ihdr_data = struct.pack('>II', width, height) + b'\x08\x02\x00\x00\x00'  # 13 bytes
    chunk = struct.pack('>I', 13) + b'IHDR' + ihdr_data + b'\x00\x00\x00\x00'
    return magic + chunk


def _make_jpeg(width: int, height: int) -> bytes:
    """Minimal JPEG with SOI + APP0 stub + SOF0 marker."""
    soi = b'\xff\xd8'
    # APP0 marker (0xFFE0), length=16 (includes 2 length bytes)
    app0 = b'\xff\xe0' + struct.pack('>H', 16) + b'JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    # SOF0 marker: 0xFFC0, length=11, precision=8, height, width, components=1
    sof0 = b'\xff\xc0' + struct.pack('>H', 11) + b'\x08' + struct.pack('>HH', height, width) + b'\x01'
    return soi + app0 + sof0


def _make_gif(width: int, height: int, version: bytes = b'GIF89a') -> bytes:
    """Minimal GIF header."""
    return version + struct.pack('<HH', width, height)


def _make_bmp(width: int, height: int) -> bytes:
    """Minimal BMP header (14-byte file header + partial DIB header)."""
    # File header: 'BM', file size (arbitrary), 2 reserved, 2 reserved, offset
    file_header = b'BM' + struct.pack('<I', 54) + b'\x00\x00\x00\x00' + struct.pack('<I', 54)
    # DIB header starts with 4-byte size, then width/height as LE int32
    dib_header = struct.pack('<I', 40) + struct.pack('<ii', width, height)
    return file_header + dib_header


# ---------------------------------------------------------------------------
# PNG dimension tests
# ---------------------------------------------------------------------------

class TestPNGDimensions:
    def test_valid_png(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(_make_png(1200, 800))
        assert get_image_dimensions(str(f)) == (1200, 800)

    def test_truncated_png(self, tmp_path):
        f = tmp_path / "truncated.png"
        # Write magic but truncate before IHDR width/height
        f.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 4)
        assert get_image_dimensions(str(f)) is None

    def test_invalid_magic(self, tmp_path):
        f = tmp_path / "fake.png"
        f.write_bytes(b'\x00' * 26)
        assert get_image_dimensions(str(f)) is None


# ---------------------------------------------------------------------------
# JPEG dimension tests
# ---------------------------------------------------------------------------

class TestJPEGDimensions:
    def test_valid_jpeg(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(_make_jpeg(640, 480))
        assert get_image_dimensions(str(f)) == (640, 480)

    def test_truncated_jpeg(self, tmp_path):
        f = tmp_path / "truncated.jpg"
        # SOI only
        f.write_bytes(b'\xff\xd8')
        assert get_image_dimensions(str(f)) is None

    def test_non_jpeg(self, tmp_path):
        f = tmp_path / "not.jpg"
        f.write_bytes(b'\x00' * 26)
        assert get_image_dimensions(str(f)) is None


# ---------------------------------------------------------------------------
# GIF dimension tests
# ---------------------------------------------------------------------------

class TestGIFDimensions:
    def test_valid_gif89a(self, tmp_path):
        f = tmp_path / "test.gif"
        f.write_bytes(_make_gif(320, 240, b'GIF89a'))
        assert get_image_dimensions(str(f)) == (320, 240)

    def test_valid_gif87a(self, tmp_path):
        f = tmp_path / "old.gif"
        f.write_bytes(_make_gif(100, 100, b'GIF87a'))
        assert get_image_dimensions(str(f)) == (100, 100)

    def test_truncated_gif(self, tmp_path):
        f = tmp_path / "truncated.gif"
        f.write_bytes(b'GIF89a' + b'\x00')
        assert get_image_dimensions(str(f)) is None


# ---------------------------------------------------------------------------
# BMP dimension tests
# ---------------------------------------------------------------------------

class TestBMPDimensions:
    def test_valid_bmp(self, tmp_path):
        f = tmp_path / "test.bmp"
        f.write_bytes(_make_bmp(800, 600))
        assert get_image_dimensions(str(f)) == (800, 600)

    def test_negative_height_bmp(self, tmp_path):
        """BMP height may be negative (top-down); abs() should be returned."""
        f = tmp_path / "topdown.bmp"
        f.write_bytes(_make_bmp(800, -600))
        assert get_image_dimensions(str(f)) == (800, 600)

    def test_truncated_bmp(self, tmp_path):
        f = tmp_path / "truncated.bmp"
        f.write_bytes(b'BM' + b'\x00' * 4)
        assert get_image_dimensions(str(f)) is None


# ---------------------------------------------------------------------------
# MIME detection tests
# ---------------------------------------------------------------------------

class TestMIMEDetection:
    def test_standard_png(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b'\x00')
        meta = get_asset_metadata(str(f))
        assert meta['mime_type'] == 'image/png'

    def test_supplemental_webp(self, tmp_path):
        f = tmp_path / "img.webp"
        f.write_bytes(b'\x00')
        meta = get_asset_metadata(str(f))
        assert meta['mime_type'] == 'image/webp'

    def test_unknown_extension(self, tmp_path):
        f = tmp_path / "file.zzztestunknown"
        f.write_bytes(b'\x00')
        meta = get_asset_metadata(str(f))
        assert meta['mime_type'] == 'application/octet-stream'

    def test_supplemental_avif(self, tmp_path):
        f = tmp_path / "img.avif"
        f.write_bytes(b'\x00')
        meta = get_asset_metadata(str(f))
        assert meta['mime_type'] == 'image/avif'

    def test_supplemental_woff2(self, tmp_path):
        f = tmp_path / "font.woff2"
        f.write_bytes(b'\x00')
        meta = get_asset_metadata(str(f))
        assert meta['mime_type'] == 'font/woff2'


# ---------------------------------------------------------------------------
# Category mapping tests
# ---------------------------------------------------------------------------

class TestCategoryMapping:
    def test_image_category(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'images'

    def test_video_category(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'video'

    def test_audio_category(self, tmp_path):
        f = tmp_path / "song.mp3"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'audio'

    def test_font_category(self, tmp_path):
        f = tmp_path / "typeface.woff2"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'fonts'

    def test_archive_category(self, tmp_path):
        f = tmp_path / "bundle.zip"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'archives'


# ---------------------------------------------------------------------------
# Synthetic content tests
# ---------------------------------------------------------------------------

class TestSyntheticContent:
    def test_with_dimensions(self):
        meta = {
            'mime_type': 'image/png',
            'category': 'images',
            'file_size': 46080,
            'dimensions': (1200, 800),
        }
        content = build_synthetic_content('assets/images/logo.png', meta)
        assert 'logo.png' in content
        assert 'assets' in content
        assert 'images' in content
        assert 'image/png' in content
        assert '1200x800' in content

    def test_without_dimensions(self):
        meta = {
            'mime_type': 'audio/flac',
            'category': 'audio',
            'file_size': 5000,
            'dimensions': None,
        }
        content = build_synthetic_content('media/song.flac', meta)
        assert 'song.flac' in content
        assert 'audio' in content
        assert 'audio/flac' in content
        assert 'x' not in content.split()  # no dimension token

    def test_size_formatting_bytes(self):
        meta = {'mime_type': 'image/png', 'category': 'images', 'file_size': 512, 'dimensions': None}
        content = build_synthetic_content('img.png', meta)
        assert '512B' in content

    def test_size_formatting_kb(self):
        meta = {'mime_type': 'image/png', 'category': 'images', 'file_size': 45 * 1024, 'dimensions': None}
        content = build_synthetic_content('img.png', meta)
        assert 'KB' in content

    def test_size_formatting_mb(self):
        meta = {'mime_type': 'video/mp4', 'category': 'video', 'file_size': 2 * 1024 * 1024, 'dimensions': None}
        content = build_synthetic_content('video.mp4', meta)
        assert 'MB' in content

    def test_parent_directory_included(self):
        meta = {'mime_type': 'image/png', 'category': 'images', 'file_size': 100, 'dimensions': None}
        content = build_synthetic_content('icons/logo.png', meta)
        assert 'icons' in content


# ---------------------------------------------------------------------------
# is_asset_file tests
# ---------------------------------------------------------------------------

class TestIsAssetFile:
    def test_png_is_asset(self):
        assert is_asset_file('path/to/image.png') is True

    def test_mp4_is_asset(self):
        assert is_asset_file('video.mp4') is True

    def test_svg_is_asset(self):
        assert is_asset_file('icon.svg') is True

    def test_py_is_not_asset(self):
        assert is_asset_file('module.py') is False

    def test_js_is_not_asset(self):
        assert is_asset_file('script.js') is False

    def test_unknown_is_not_asset(self):
        assert is_asset_file('file.xyz') is False

    def test_no_extension(self):
        assert is_asset_file('Makefile') is False

    def test_case_insensitive(self):
        assert is_asset_file('IMAGE.PNG') is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_byte_file(self, tmp_path):
        f = tmp_path / "empty.png"
        f.write_bytes(b'')
        meta = get_asset_metadata(str(f))
        assert meta['file_size'] == 0
        assert meta['dimensions'] is None

    def test_nonexistent_file_dimensions(self):
        result = get_image_dimensions('/nonexistent/path/image.png')
        assert result is None

    def test_nonexistent_file_metadata(self):
        meta = get_asset_metadata('/nonexistent/path/image.png')
        assert meta['file_size'] == 0
