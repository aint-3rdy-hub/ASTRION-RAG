"""FAISS index creation and metadata persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_UNIT,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from src.models import Chunk

logger = logging.getLogger(__name__)


class IndexWriteError(Exception):
    """Raised when the FAISS index or metadata cannot be written."""


def build_metadata(
    chunks: list[Chunk],
    embedding_model: str = EMBEDDING_MODEL_NAME,
    embedding_dimension: int = 0,
    num_vectors: int | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable metadata mapping FAISS row → chunk.

    ``vectors[i]`` corresponds to FAISS vector position ``i``.
    """
    vector_count = len(chunks) if num_vectors is None else num_vectors
    return {
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "num_vectors": vector_count,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_unit": CHUNK_UNIT,
        "vectors": [
            {
                "vector_id": index,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
            }
            for index, chunk in enumerate(chunks)
        ],
    }


def write_metadata(metadata: dict[str, Any], path: Path = METADATA_PATH) -> None:
    """Write metadata JSON to disk."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise IndexWriteError(f"Failed to write metadata to {path}: {exc}") from exc


def write_faiss_index(
    embeddings: NDArray[np.float32],
    path: Path = FAISS_INDEX_PATH,
) -> None:
    """Create an inner-product FAISS index and save it to disk."""
    if embeddings.ndim != 2:
        raise IndexWriteError(
            f"Embeddings must be 2-D (n, dim); received shape {embeddings.shape}"
        )

    n_vectors, dimension = embeddings.shape
    logger.info(
        "Creating FAISS IndexFlatIP with %s vector(s), dimension %s",
        n_vectors,
        dimension,
    )
    try:
        index = faiss.IndexFlatIP(dimension)
        if n_vectors:
            vectors = np.ascontiguousarray(embeddings.astype(np.float32))
            index.add(vectors)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
    except Exception as exc:  # noqa: BLE001
        raise IndexWriteError(f"Failed to write FAISS index to {path}: {exc}") from exc

    logger.info("Saved FAISS index: %s", path)


def save_index(
    embeddings: NDArray[np.float32],
    chunks: list[Chunk],
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> dict[str, Any]:
    """Persist the FAISS index and the matching chunk metadata."""
    if embeddings.shape[0] != len(chunks):
        raise IndexWriteError(
            "Embedding count does not match chunk count: "
            f"{embeddings.shape[0]} vectors vs {len(chunks)} chunks"
        )

    dimension = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    metadata = build_metadata(
        chunks,
        embedding_model=embedding_model,
        embedding_dimension=dimension,
        num_vectors=embeddings.shape[0],
    )
    write_faiss_index(embeddings, index_path)
    write_metadata(metadata, metadata_path)
    logger.info("Saved metadata: %s", metadata_path)
    return metadata
