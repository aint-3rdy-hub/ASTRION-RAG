"""End-to-end RAG pipeline: retrieve → filter → Groq → cited answer.

Never log API keys or environment secrets.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.config import DEBUG, LOG_LEVEL
from src.embeddings import EmbeddingError
from src.generate import (
    EMPTY_GROUNDED_ANSWER,
    LOW_RELEVANCE_ANSWER,
    GenerationError,
    Generator,
    redact_secrets,
)
from src.retrieve import RetrievalError, RetrievalResult, Retriever

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """User-facing pipeline failure."""


def _generation_user_message(exc: GenerationError) -> str:
    if exc.kind == "timeout":
        return "The language model timed out. Please try again."
    if exc.kind == "auth":
        return (
            "The language model rejected the API key. "
            "Check GROQ_API_KEY in your local .env file."
        )
    if exc.kind == "rate_limit":
        return "The language model is rate-limited. Please wait and try again."
    if exc.kind == "config":
        return redact_secrets(str(exc))
    return redact_secrets(str(exc)) or "The language model request failed. Please try again."


def configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _latency(
    retrieval_seconds: float = 0.0,
    generation_seconds: float = 0.0,
    total_seconds: float | None = None,
) -> dict[str, float]:
    total = (
        total_seconds
        if total_seconds is not None
        else retrieval_seconds + generation_seconds
    )
    return {
        "retrieval_seconds": round(retrieval_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "total_seconds": round(total, 4),
    }


def _sources(retrieval: RetrievalResult) -> list[dict[str, Any]]:
    return [
        {
            "source": hit.source,
            "page": hit.page,
            "chunk_id": hit.chunk_id,
            "score": round(float(hit.score), 6),
            "rank": hit.rank,
        }
        for hit in retrieval.hits
    ]


def _payload(
    question: str,
    answer: str,
    retrieval: RetrievalResult | None,
    *,
    retrieval_seconds: float = 0.0,
    generation_seconds: float = 0.0,
    total_seconds: float | None = None,
) -> dict[str, Any]:
    chunks = retrieval.hits if retrieval is not None else []
    return {
        "question": question,
        "answer": answer,
        "sources": _sources(retrieval) if retrieval is not None else [],
        "retrieved_chunks": [hit.to_dict() for hit in chunks],
        "retrieval_count": len(chunks),
        "latency": _latency(retrieval_seconds, generation_seconds, total_seconds),
    }


class RAGPipeline:
    """Connect retrieval and Groq generation into ``answer(question)``."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        **retriever_kwargs: Any,
    ) -> None:
        self._retriever = retriever
        self._retriever_kwargs = retriever_kwargs
        self.generator = generator if generator is not None else Generator()
        self._last_retrieval_seconds = 0.0
        self._last_retrieval: RetrievalResult | None = None

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever.load(**self._retriever_kwargs)
        return self._retriever

    @classmethod
    def load(cls, **retriever_kwargs: Any) -> RAGPipeline:
        """Build a pipeline. Index files are loaded on first ``answer()`` call."""
        return cls(**retriever_kwargs)

    def answer(self, question: str) -> dict[str, Any]:
        """Run retrieve → generate and always return a structured result."""
        started = time.perf_counter()
        cleaned = (question or "").strip()
        logger.info("RAG question received (chars=%s)", len(cleaned))

        if not cleaned:
            return _payload(
                question or "",
                EMPTY_GROUNDED_ANSWER,
                None,
                total_seconds=time.perf_counter() - started,
            )

        try:
            return self._answer(cleaned, started)
        except RetrievalError as exc:
            logger.error("Retrieval failure: %s", redact_secrets(str(exc)))
            return self._fail(cleaned, started, 0.0, 0.0, self._retrieval_message(exc))
        except EmbeddingError as exc:
            logger.error("Embedding failure: %s", exc)
            return self._fail(
                cleaned,
                started,
                0.0,
                0.0,
                "Could not embed the question. Please try again.",
            )
        except GenerationError as exc:
            retrieval_s = getattr(self, "_last_retrieval_seconds", 0.0)
            message = _generation_user_message(exc)
            logger.error(
                "Groq generation failure kind=%s message=%s",
                exc.kind,
                redact_secrets(message),
            )
            return self._fail(
                cleaned,
                started,
                retrieval_s,
                0.0,
                message,
                retrieval=self._last_retrieval,
            )
        except Exception as exc:  # noqa: BLE001
            if DEBUG:
                raise
            logger.error(
                "Unexpected RAG error: %s",
                redact_secrets(f"{type(exc).__name__}: {exc}"),
            )
            return self._fail(
                cleaned,
                started,
                0.0,
                0.0,
                "An unexpected error occurred while answering.",
            )

    def _answer(self, question: str, started: float) -> dict[str, Any]:
        retrieval_started = time.perf_counter()
        retrieval = self.retriever.search(question)
        retrieval_seconds = time.perf_counter() - retrieval_started
        self._last_retrieval_seconds = retrieval_seconds
        self._last_retrieval = retrieval
        self._log_retrieval(question, retrieval, retrieval_seconds)

        if retrieval.empty:
            logger.info(
                "Empty retrieval (reason=%s). Skipping Groq; no answer will be invented.",
                retrieval.empty_reason,
            )
            grounded = (
                LOW_RELEVANCE_ANSWER
                if retrieval.empty_reason == "no_chunks_above_threshold"
                else EMPTY_GROUNDED_ANSWER
            )
            return _payload(
                question,
                grounded,
                retrieval,
                retrieval_seconds=retrieval_seconds,
                generation_seconds=0.0,
                total_seconds=time.perf_counter() - started,
            )

        generation_started = time.perf_counter()
        answer = self.generator.generate(question, retrieval.hits)
        generation_seconds = time.perf_counter() - generation_started
        logger.info("Generation complete in %.3fs (chars=%s)", generation_seconds, len(answer))
        return _payload(
            question,
            answer,
            retrieval,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - started,
        )

    def _fail(
        self,
        question: str,
        started: float,
        retrieval_seconds: float,
        generation_seconds: float,
        message: str,
        retrieval: RetrievalResult | None = None,
    ) -> dict[str, Any]:
        return _payload(
            question,
            message,
            retrieval,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _retrieval_message(exc: RetrievalError) -> str:
        text = str(exc)
        lowered = text.lower()
        if "faiss" in lowered and ("missing" in lowered or "could not load" in lowered):
            return (
                "The FAISS index is missing or unreadable. "
                "Run: .\\.venv\\Scripts\\python.exe -m src.ingest"
            )
        if "metadata" in lowered:
            return (
                "Chunk metadata is missing or unreadable. "
                "Run: .\\.venv\\Scripts\\python.exe -m src.ingest"
            )
        if "embed" in lowered:
            return "Could not embed the question. Please try again."
        return text

    @staticmethod
    def _log_retrieval(
        question: str,
        retrieval: RetrievalResult,
        retrieval_seconds: float,
    ) -> None:
        logger.info(
            "Retrieval finished in %.3fs: hits=%s empty=%s reason=%s top_k=%s min_score=%s",
            retrieval_seconds,
            len(retrieval.hits),
            retrieval.empty,
            retrieval.empty_reason,
            retrieval.top_k,
            retrieval.min_score,
        )
        for hit in retrieval.hits:
            logger.info(
                "  rank=%s score=%.4f source=%s page=%s chunk_id=%s chars=%s",
                hit.rank,
                hit.score,
                hit.source,
                hit.page,
                hit.chunk_id,
                len(hit.text),
            )
        logger.debug("Question text (truncated): %s", question[:200])


def main() -> int:
    configure_logging()
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Usage: python -m src.rag \"your question\"", file=sys.stderr)
        return 2
    try:
        pipeline = RAGPipeline.load()
    except RetrievalError as exc:
        print(f"RAG failed: {exc}", file=sys.stderr)
        return 1
    result = pipeline.answer(question)
    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
