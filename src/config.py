"""Central configuration for ASTRION RAG.

Chunking uses whitespace-separated words as a token/word equivalent.
``CHUNK_SIZE = 800`` therefore means 800 words per chunk, and
``CHUNK_OVERLAP = 120`` means 120 overlapping words between adjacent chunks.
This keeps identifiers and configuration consistent without a separate tokenizer.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Chunking (word-equivalent tokens)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNK_UNIT = "words"

# ---------------------------------------------------------------------------
# Local embeddings (Sentence Transformers)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
NORMALIZE_EMBEDDINGS = True

# ---------------------------------------------------------------------------
# Retrieval (cosine similarity on L2-normalized IndexFlatIP vectors)
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 5
RETRIEVAL_MIN_SCORE = 0.25

# ---------------------------------------------------------------------------
# Generation (Groq). API key is read from the environment, never from this file.
# ---------------------------------------------------------------------------
PROMPT_PATH = PROJECT_ROOT / "prompts" / "rag_prompt.txt"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _env_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


GROQ_TIMEOUT_SECONDS = _env_positive_float("GROQ_TIMEOUT_SECONDS", 30.0)
GROQ_MAX_TOKENS = 1024
GROQ_TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEBUG = os.getenv("ASTRION_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = os.getenv("ASTRION_LOG_LEVEL", "INFO").upper()
