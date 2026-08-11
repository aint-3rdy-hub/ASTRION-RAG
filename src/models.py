"""Shared data structures for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExtractedPage:
    """One PDF page after text extraction."""

    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A single searchable text chunk with a stable identifier."""

    chunk_id: str
    source: str
    page: int
    text: str


@dataclass
class DocumentFailure:
    """A PDF that could not be processed."""

    path: str
    reason: str


@dataclass
class IngestionResult:
    """Summary of a full ingestion run."""

    documents_discovered: int = 0
    documents_processed: int = 0
    pages_processed: int = 0
    chunks_created: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    index_path: Path | None = None
    metadata_path: Path | None = None
    elapsed_seconds: float = 0.0
    failures: list[DocumentFailure] = field(default_factory=list)
    unsupported_files: list[str] = field(default_factory=list)
    empty_pages_skipped: int = 0
    skipped_index: bool = False
