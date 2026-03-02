"""Orchestration layer for incremental re-indexing.

Orchestrates parsing, chunking, embedding, and storage into a complete
indexing pipeline with incremental update support.
"""

from tessera.indexer._helpers import (
    ALL_DOCUMENT_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    MARKUP_EXTENSIONS,
    TEXT_EXTENSIONS,
    IndexStats,
)
from tessera.indexer._pipeline import IndexerPipeline

__all__ = [
    "IndexerPipeline",
    "IndexStats",
    "DOCUMENT_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "MARKUP_EXTENSIONS",
    "ALL_DOCUMENT_EXTENSIONS",
]
