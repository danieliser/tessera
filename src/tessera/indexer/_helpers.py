"""Indexer helpers: constants, package detection, and statistics."""

import json
import os
from dataclasses import dataclass

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
                    with open(path) as f:
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

