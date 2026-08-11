"""Groq LLM generation with a version-controlled prompt.

Never hard-code, print, or log API keys.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from src.config import (
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    GROQ_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    PROMPT_PATH,
)
from src.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

_SECRET_PATTERN = re.compile(r"(gsk_[A-Za-z0-9]+)|(sk-[A-Za-z0-9]+)", re.IGNORECASE)

EMPTY_GROUNDED_ANSWER = (
    "I could not find enough information in the provided documents to answer this."
)
LOW_RELEVANCE_ANSWER = (
    "I could not find sufficiently relevant information in the provided documents "
    "to answer this."
)


class GenerationError(Exception):
    """User-facing generation failure."""

    def __init__(
        self,
        message: str,
        *,
        timeout: bool = False,
        kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind or ("timeout" if timeout else "api")
        self.timeout = self.kind == "timeout"


def redact_secrets(text: str) -> str:
    """Strip credential-like tokens before logging."""
    return _SECRET_PATTERN.sub("[REDACTED]", text)


def load_prompt_template(path: Path | None = None) -> str:
    prompt_path = path or PROMPT_PATH
    try:
        template = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GenerationError(f"Could not read prompt file {prompt_path}: {exc}") from exc
    if "{context}" not in template or "{question}" not in template:
        raise GenerationError(
            f"Prompt template must contain {{context}} and {{question}}: {prompt_path}"
        )
    return template


def build_context(hits: list[RetrievedChunk]) -> str:
    """Format retrieved chunks with stable source identifiers for citations."""
    blocks: list[str] = []
    for hit in hits:
        identifier = f"[{hit.chunk_id}] {hit.source} (page {hit.page})"
        blocks.append(f"{identifier}\n{hit.text}")
    return "\n\n".join(blocks)


def validate_generation_env() -> None:
    """Reject invalid Groq settings before any network call."""
    load_dotenv(PROJECT_ROOT / ".env")
    raw_timeout = os.getenv("GROQ_TIMEOUT_SECONDS")
    if raw_timeout is not None and raw_timeout.strip():
        try:
            if float(raw_timeout) <= 0:
                raise ValueError
        except ValueError as exc:
            raise GenerationError(
                "Invalid environment configuration: GROQ_TIMEOUT_SECONDS must be a positive number.",
                kind="config",
            ) from exc
    model = (os.getenv("GROQ_MODEL") or GROQ_MODEL or "").strip()
    if not model:
        raise GenerationError(
            "Invalid environment configuration: GROQ_MODEL is empty.",
            kind="config",
        )


def _api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key or key == "your_groq_api_key_here":
        raise GenerationError(
            "GROQ_API_KEY is missing. Set it in the local .env file (not in source code).",
            kind="config",
        )
    return key


class Generator:
    """Call Groq with the version-controlled RAG prompt."""

    def __init__(
        self,
        *,
        client: Groq | None = None,
        model: str | None = None,
        timeout_seconds: float = GROQ_TIMEOUT_SECONDS,
        prompt_path: Path | None = None,
    ) -> None:
        self.model = model or GROQ_MODEL
        self.timeout_seconds = timeout_seconds
        self.prompt_template = load_prompt_template(prompt_path)
        self._client = client

    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=_api_key(), timeout=self.timeout_seconds)
        return self._client

    def generate(self, question: str, hits: list[RetrievedChunk]) -> str:
        context = build_context(hits)
        prompt = self.prompt_template.replace("{context}", context).replace(
            "{question}", question
        )
        validate_generation_env()
        logger.info(
            "Calling Groq model=%s timeout=%ss context_chunks=%s prompt_chars=%s",
            self.model,
            self.timeout_seconds,
            len(hits),
            len(prompt),
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
            )
        except APITimeoutError as exc:
            logger.error("Groq timeout")
            raise GenerationError(
                "The language model timed out. Please try again.",
                kind="timeout",
            ) from exc
        except AuthenticationError as exc:
            logger.error("Groq authentication failure: %s", type(exc).__name__)
            raise GenerationError(
                "The language model rejected the API key. "
                "Check GROQ_API_KEY in your local .env file.",
                kind="auth",
            ) from exc
        except RateLimitError as exc:
            logger.error("Groq rate limit: %s", type(exc).__name__)
            raise GenerationError(
                "The language model is rate-limited. Please wait and try again.",
                kind="rate_limit",
            ) from exc
        except (APIConnectionError, APIStatusError) as exc:
            logger.error("Groq API status/connection error: %s", type(exc).__name__)
            raise GenerationError(
                "The language model request failed. Please try again.",
                kind="api",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Groq unexpected error: %s",
                redact_secrets(f"{type(exc).__name__}: {exc}"),
            )
            raise GenerationError(
                "The language model request failed. Please try again.",
                kind="api",
            ) from exc

        try:
            text = (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError, KeyError) as exc:
            raise GenerationError("The language model returned an empty response.") from exc

        if not text:
            raise GenerationError("The language model returned an empty response.")
        return text
