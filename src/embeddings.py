"""Local Sentence Transformer embeddings for ASTRION RAG."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, NORMALIZE_EMBEDDINGS

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


class EmbeddingError(Exception):
    """Raised when the local embedding model cannot be used."""


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load (and cache) the configured local embedding model."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    logger.info("Loading local embedding model: %s", model_name)
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(
            f"Failed to load embedding model '{model_name}': {exc}"
        ) from exc

    _model_cache[model_name] = model
    return model


def embed_texts(
    texts: list[str],
    model: SentenceTransformer | None = None,
    model_name: str = EMBEDDING_MODEL_NAME,
    normalize: bool = NORMALIZE_EMBEDDINGS,
) -> NDArray[np.float32]:
    """Encode texts into a float32 matrix of shape (n, dimension).

    Embeddings are L2-normalized when ``normalize`` is true so inner-product
    search in FAISS is equivalent to cosine similarity.
    """
    if not texts:
        dimension = (
            model.get_sentence_embedding_dimension()
            if model is not None
            else 0
        )
        return np.zeros((0, dimension or 0), dtype=np.float32)

    if model is None:
        model = get_embedding_model(model_name)

    logger.info("Generating embeddings for %s chunk(s)", len(texts))
    try:
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if matrix.ndim != 2:
        raise EmbeddingError(
            f"Expected a 2-D embedding matrix, received shape {matrix.shape}"
        )

    logger.info(
        "Generated %s embedding vector(s) with dimension %s",
        matrix.shape[0],
        matrix.shape[1],
    )
    return matrix
