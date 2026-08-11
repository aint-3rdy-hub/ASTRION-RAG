"""Deterministic word-equivalent chunking with overlap."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.models import Chunk, ExtractedPage

_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def source_slug(source: str) -> str:
    """Build a stable slug from a relative source path for chunk IDs."""
    stem = Path(source).with_suffix("").as_posix()
    slug = _NON_ALNUM.sub("_", stem).strip("_").lower()
    return slug or "document"


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """Return a deterministic chunk identifier.

    Example: ``network_security_p12_c03``
    ``chunk_index`` is 1-based.
    """
    return f"{source_slug(source)}_p{page}_c{chunk_index:02d}"


def split_into_word_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping word-equivalent chunks.

    Empty (or whitespace-only) text still yields one empty chunk so that
    pages with little or no extractable text are not discarded.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return [""]

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Convert extracted pages into ordered, deterministic chunks."""
    chunks: list[Chunk] = []
    for page in pages:
        texts = split_into_word_chunks(page.text, chunk_size, chunk_overlap)
        for index, text in enumerate(texts, start=1):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(page.source, page.page, index),
                    source=page.source,
                    page=page.page,
                    text=text,
                )
            )
    return chunks
