"""Tests for src/tessera/assets.py"""

import struct
from pathlib import Path

import pytest

from tessera.assets import (
    ASSET_EXTENSIONS,
    SUPPLEMENTAL_MIME_TYPES,
    build_asset_synthetic_content,
    extract_image_dimensions,
    get_asset_category,
    get_asset_metadata,
    get_mime_type,
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
    file_header = b'BM' + struct.pack('<I', 54) + b'\x00\x00\x00\x00' + struct.pack('<I', 54)
    dib_header = struct.pack('<I', 40) + struct.pack('<ii', width, height)
    return file_header + dib_header


# ---------------------------------------------------------------------------
# PNG dimension tests
# ---------------------------------------------------------------------------

class TestPNGDimensions:
    def test_valid_png(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(_make_png(1200, 800))
        assert extract_image_dimensions(str(f)) == {'width': 1200, 'height': 800}

    def test_truncated_png(self, tmp_path):
        f = tmp_path / "truncated.png"
        f.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 4)
        assert extract_image_dimensions(str(f)) is None

    def test_invalid_magic(self, tmp_path):
        f = tmp_path / "fake.png"
        f.write_bytes(b'\x00' * 26)
        assert extract_image_dimensions(str(f)) is None


# ---------------------------------------------------------------------------
# JPEG dimension tests
# ---------------------------------------------------------------------------

class TestJPEGDimensions:
    def test_valid_jpeg(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(_make_jpeg(640, 480))
        assert extract_image_dimensions(str(f)) == {'width': 640, 'height': 480}

    def test_truncated_jpeg(self, tmp_path):
        f = tmp_path / "truncated.jpg"
        f.write_bytes(b'\xff\xd8')
        assert extract_image_dimensions(str(f)) is None

    def test_non_jpeg(self, tmp_path):
        f = tmp_path / "not.jpg"
        f.write_bytes(b'\x00' * 26)
        assert extract_image_dimensions(str(f)) is None


# ---------------------------------------------------------------------------
# GIF dimension tests
# ---------------------------------------------------------------------------

class TestGIFDimensions:
    def test_valid_gif89a(self, tmp_path):
        f = tmp_path / "test.gif"
        f.write_bytes(_make_gif(320, 240, b'GIF89a'))
        assert extract_image_dimensions(str(f)) == {'width': 320, 'height': 240}

    def test_valid_gif87a(self, tmp_path):
        f = tmp_path / "old.gif"
        f.write_bytes(_make_gif(100, 100, b'GIF87a'))
        assert extract_image_dimensions(str(f)) == {'width': 100, 'height': 100}

    def test_truncated_gif(self, tmp_path):
        f = tmp_path / "truncated.gif"
        f.write_bytes(b'GIF89a' + b'\x00')
        assert extract_image_dimensions(str(f)) is None


# ---------------------------------------------------------------------------
# BMP dimension tests
# ---------------------------------------------------------------------------

class TestBMPDimensions:
    def test_valid_bmp(self, tmp_path):
        f = tmp_path / "test.bmp"
        f.write_bytes(_make_bmp(800, 600))
        assert extract_image_dimensions(str(f)) == {'width': 800, 'height': 600}

    def test_negative_height_bmp(self, tmp_path):
        """BMP height may be negative (top-down); abs() should be returned."""
        f = tmp_path / "topdown.bmp"
        f.write_bytes(_make_bmp(800, -600))
        assert extract_image_dimensions(str(f)) == {'width': 800, 'height': 600}

    def test_truncated_bmp(self, tmp_path):
        f = tmp_path / "truncated.bmp"
        f.write_bytes(b'BM' + b'\x00' * 4)
        assert extract_image_dimensions(str(f)) is None


# ---------------------------------------------------------------------------
# get_mime_type tests (standalone function)
# ---------------------------------------------------------------------------

class TestGetMimeType:
    def test_standard_png(self):
        assert get_mime_type('img.png') == 'image/png'

    def test_standard_mp4(self):
        assert get_mime_type('video.mp4') == 'video/mp4'

    def test_supplemental_woff2(self):
        assert get_mime_type('font.woff2') == 'font/woff2'

    def test_supplemental_opus(self):
        assert get_mime_type('audio.opus') == 'audio/opus'

    def test_supplemental_avif(self):
        assert get_mime_type('img.avif') == 'image/avif'

    def test_unknown_extension(self):
        assert get_mime_type('file.zzztestunknown') == 'application/octet-stream'

    def test_filename_with_path(self):
        # get_mime_type accepts just filename but also works with full paths
        assert get_mime_type('path/to/image.png') == 'image/png'


# ---------------------------------------------------------------------------
# get_asset_category tests (standalone function)
# ---------------------------------------------------------------------------

class TestGetAssetCategory:
    def test_png_is_image(self):
        assert get_asset_category('photo.png') == 'image'

    def test_mp4_is_video(self):
        assert get_asset_category('clip.mp4') == 'video'

    def test_mp3_is_audio(self):
        assert get_asset_category('song.mp3') == 'audio'

    def test_woff2_is_font(self):
        assert get_asset_category('type.woff2') == 'font'

    def test_zip_is_archive(self):
        assert get_asset_category('bundle.zip') == 'archive'

    def test_unknown_is_binary(self):
        assert get_asset_category('file.xyz') == 'binary'

    def test_py_is_binary(self):
        assert get_asset_category('module.py') == 'binary'


# ---------------------------------------------------------------------------
# get_asset_metadata tests
# ---------------------------------------------------------------------------

class TestGetAssetMetadata:
    def test_image_category_singular(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'image'

    def test_video_category(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'video'

    def test_audio_category(self, tmp_path):
        f = tmp_path / "song.mp3"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'audio'

    def test_font_category_singular(self, tmp_path):
        f = tmp_path / "typeface.woff2"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'font'

    def test_archive_category_singular(self, tmp_path):
        f = tmp_path / "bundle.zip"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'archive'

    def test_unknown_category_is_binary(self, tmp_path):
        f = tmp_path / "file.zzztestunknown"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['category'] == 'binary'

    def test_dimensions_dict_for_png(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(_make_png(200, 100))
        meta = get_asset_metadata(str(f))
        assert meta['dimensions'] == {'width': 200, 'height': 100}

    def test_no_dimensions_for_video(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b'\x00')
        assert get_asset_metadata(str(f))['dimensions'] is None


# ---------------------------------------------------------------------------
# build_asset_synthetic_content tests
# ---------------------------------------------------------------------------

class TestBuildAssetSyntheticContent:
    def test_with_dimensions(self):
        content = build_asset_synthetic_content(
            filename='logo.png',
            path_components=['assets', 'images'],
            mime_type='image/png',
            dimensions={'width': 1200, 'height': 800},
            file_size=46080,
        )
        assert 'logo.png' in content
        assert 'assets' in content
        assert 'images' in content
        assert 'image/png' in content
        assert '1200x800' in content

    def test_without_dimensions(self):
        content = build_asset_synthetic_content(
            filename='archive.zip',
            path_components=['assets', 'archives'],
            mime_type='application/zip',
            dimensions=None,
            file_size=1234567,
        )
        assert 'archive.zip' in content
        assert 'archives' in content
        assert 'application/zip' in content
        # No dimension string
        assert 'x' not in [p for p in content.split() if p.count('x') == 1 and p[0].isdigit()]

    def test_path_components_included(self):
        content = build_asset_synthetic_content(
            filename='file.png',
            path_components=['assets', 'images'],
            mime_type='image/png',
        )
        assert 'assets' in content
        assert 'images' in content

    def test_size_formatting_bytes(self):
        content = build_asset_synthetic_content('f.png', [], 'image/png', file_size=512)
        assert '512B' in content

    def test_size_formatting_kb(self):
        content = build_asset_synthetic_content('f.png', [], 'image/png', file_size=45 * 1024)
        assert 'KB' in content

    def test_size_formatting_mb(self):
        content = build_asset_synthetic_content('f.mp4', [], 'video/mp4', file_size=2 * 1024 * 1024)
        assert 'MB' in content


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
        assert is_asset_file('file.zzztestunknown') is False

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
        result = extract_image_dimensions('/nonexistent/path/image.png')
        assert result is None

    def test_nonexistent_file_metadata(self):
        meta = get_asset_metadata('/nonexistent/path/image.png')
        assert meta['file_size'] == 0

    def test_asset_extensions_is_dict(self):
        """ASSET_EXTENSIONS must be a dict mapping ext→category."""
        assert isinstance(ASSET_EXTENSIONS, dict)
        assert ASSET_EXTENSIONS['.png'] == 'image'
        assert ASSET_EXTENSIONS['.mp4'] == 'video'

    def test_supplemental_mime_types_name(self):
        """Constant must be named SUPPLEMENTAL_MIME_TYPES (not SUPPLEMENTAL_MIME_MAP)."""
        assert '.woff2' in SUPPLEMENTAL_MIME_TYPES
        assert SUPPLEMENTAL_MIME_TYPES['.woff2'] == 'font/woff2'
