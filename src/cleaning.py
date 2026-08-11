"""Lightweight cleaning of PDF extraction artifacts."""

from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"[ \t\f\v]+")


def clean_text(text: str) -> str:
    """Normalize extracted PDF text without rewriting its meaning.

    Cleaning is intentionally conservative:

    * Windows/Mac newlines are normalized
    * tabs and non-breaking spaces become regular spaces
    * repeated spaces on a line are collapsed
    * single newlines (typical PDF line wrapping) become spaces
    * repeated blank lines collapse to one paragraph break
    """
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\xa0", " ").replace("\u00a0", " ")

    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in normalized.split("\n")]

    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if line:
            buffer.append(line)
            continue
        if buffer:
            paragraphs.append(" ".join(buffer))
            buffer = []

    if buffer:
        paragraphs.append(" ".join(buffer))

    return "\n\n".join(paragraphs)
