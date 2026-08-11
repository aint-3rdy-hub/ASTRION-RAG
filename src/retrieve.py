"""Semantic retrieval from the FAISS index. No Groq / LLM calls."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_TOP_K,
)
from src.embeddings import EmbeddingError, embed_texts, get_embedding_model

logger = logging.getLogger(__name__)

SAMPLE_QUERY = "Why should firewall rules match a written security policy?"


class RetrievalError(Exception):
    """User-facing retrieval failure (missing index, embed/search errors)."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked hit. ``score`` is cosine similarity (higher is better)."""

    rank: int
    score: float
    source: str
    page: int
    chunk_id: str
    text: str
    vector_id: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 6)
        return payload


@dataclass
class RetrievalResult:
    """Deterministic retrieval payload for evaluation and the generator."""

    query: str
    hits: list[RetrievedChunk]
    empty: bool
    empty_reason: str | None
    top_k: int
    min_score: float
    embedding_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "empty": self.empty,
            "empty_reason": self.empty_reason,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "embedding_model": self.embedding_model,
            "hit_count": len(self.hits),
            "hits": [hit.to_dict() for hit in self.hits],
        }


class Retriever:
    """Load FAISS + metadata once, then search with a local embedding model."""

    def __init__(
        self,
        index: faiss.Index,
        metadata: dict[str, Any],
        model: Any,
        *,
        top_k: int = RETRIEVAL_TOP_K,
        min_score: float = RETRIEVAL_MIN_SCORE,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        if top_k < 1:
            raise RetrievalError("top_k must be >= 1")
        if not 0.0 <= min_score <= 1.0:
            raise RetrievalError("min_score must be between 0.0 and 1.0")

        self.index = index
        self.metadata = metadata
        self.model = model
        self.top_k = top_k
        self.min_score = min_score
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.vectors: list[dict[str, Any]] = list(metadata.get("vectors") or [])
        self.embedding_model = str(
            metadata.get("embedding_model") or EMBEDDING_MODEL_NAME
        )

        if int(metadata.get("num_vectors", len(self.vectors))) != index.ntotal:
            raise RetrievalError(
                "FAISS index size does not match metadata: "
                f"index.ntotal={index.ntotal}, metadata={metadata.get('num_vectors')}"
            )
        if len(self.vectors) != index.ntotal:
            raise RetrievalError(
                "Metadata vector list length does not match FAISS ntotal: "
                f"{len(self.vectors)} vs {index.ntotal}"
            )

    @classmethod
    def load(
        cls,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        *,
        model: Any | None = None,
        top_k: int = RETRIEVAL_TOP_K,
        min_score: float = RETRIEVAL_MIN_SCORE,
    ) -> Retriever:
        """Load index files and the same embedding model used at ingestion."""
        index_path = index_path or FAISS_INDEX_PATH
        metadata_path = metadata_path or METADATA_PATH
        _require_index_files(index_path, metadata_path)

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalError(
                f"Could not read metadata file {metadata_path}: {exc}"
            ) from exc

        if not isinstance(metadata, dict):
            raise RetrievalError(f"Metadata is not a JSON object: {metadata_path}")

        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Could not load FAISS index {index_path}: {exc}") from exc

        model_name = str(metadata.get("embedding_model") or EMBEDDING_MODEL_NAME)
        if model is None:
            try:
                model = get_embedding_model(model_name)
            except EmbeddingError as exc:
                raise RetrievalError(str(exc)) from exc

        logger.info(
            "Loaded retriever: %s vectors, model=%s, top_k=%s, min_score=%s",
            index.ntotal,
            model_name,
            top_k,
            min_score,
        )
        return cls(
            index,
            metadata,
            model,
            top_k=top_k,
            min_score=min_score,
            index_path=index_path,
            metadata_path=metadata_path,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """Embed ``query`` and return top-k chunks at or above ``min_score``."""
        k = self.top_k if top_k is None else top_k
        threshold = self.min_score if min_score is None else min_score
        if k < 1:
            raise RetrievalError("top_k must be >= 1")

        cleaned = query.strip()
        if not cleaned:
            return self._empty(query, k, threshold, "empty_query")

        if self.index.ntotal == 0:
            return self._empty(cleaned, k, threshold, "empty_index")

        try:
            query_vector = embed_texts([cleaned], model=self.model)
        except EmbeddingError as exc:
            raise RetrievalError(f"Query embedding failed: {exc}") from exc

        if query_vector.shape[1] != self.index.d:
            raise RetrievalError(
                "Query embedding dimension does not match the index: "
                f"{query_vector.shape[1]} vs {self.index.d}"
            )

        search_k = min(k, int(self.index.ntotal))
        try:
            scores, ids = self.index.search(query_vector, search_k)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"FAISS search failed: {exc}") from exc

        candidates: list[tuple[float, int]] = []
        for score, vector_id in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            if int(vector_id) < 0:
                continue
            similarity = float(score)
            if similarity >= threshold:
                candidates.append((similarity, int(vector_id)))

        # Score desc, then vector_id asc so ties are stable for evaluation.
        candidates.sort(key=lambda item: (-item[0], item[1]))

        hits: list[RetrievedChunk] = []
        for rank, (similarity, vector_id) in enumerate(candidates, start=1):
            record = self.vectors[vector_id]
            hits.append(
                RetrievedChunk(
                    rank=rank,
                    score=similarity,
                    source=str(record.get("source", "")),
                    page=int(record.get("page") or 0),
                    chunk_id=str(record.get("chunk_id", "")),
                    text=str(record.get("text", "")),
                    vector_id=vector_id,
                )
            )

        if not hits:
            logger.info(
                "Empty retrieval for query %r (threshold=%s, searched=%s)",
                cleaned,
                threshold,
                search_k,
            )
            return self._empty(cleaned, k, threshold, "no_chunks_above_threshold")

        logger.info("Retrieved %s chunk(s) for query %r", len(hits), cleaned)
        return RetrievalResult(
            query=cleaned,
            hits=hits,
            empty=False,
            empty_reason=None,
            top_k=k,
            min_score=threshold,
            embedding_model=self.embedding_model,
        )

    def _empty(
        self,
        query: str,
        top_k: int,
        min_score: float,
        reason: str,
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            hits=[],
            empty=True,
            empty_reason=reason,
            top_k=top_k,
            min_score=min_score,
            embedding_model=self.embedding_model,
        )


def _require_index_files(index_path: Path, metadata_path: Path) -> None:
    missing: list[str] = []
    if not index_path.is_file():
        missing.append(str(index_path))
    if not metadata_path.is_file():
        missing.append(str(metadata_path))
    if missing:
        joined = ", ".join(missing)
        raise RetrievalError(
            "Index files are missing: "
            f"{joined}. Run: .\\.venv\\Scripts\\python.exe -m src.ingest"
        )


def retrieve_sample(
    query: str = SAMPLE_QUERY,
    *,
    top_k: int = RETRIEVAL_TOP_K,
    min_score: float = RETRIEVAL_MIN_SCORE,
) -> RetrievalResult:
    """Retrieve chunks for a sample query (used by tests and the CLI)."""
    retriever = Retriever.load(top_k=top_k, min_score=min_score)
    return retriever.search(query)


def main() -> int:
    import json as json_lib
    import sys

    query = " ".join(sys.argv[1:]).strip() or SAMPLE_QUERY
    try:
        result = retrieve_sample(query)
    except RetrievalError as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1
    print(json_lib.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
