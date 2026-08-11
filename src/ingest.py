"""Document ingestion CLI: PDF → chunks → local embeddings → FAISS.

Run from the project root (no Groq / LLM calls):

    python -m src.ingest

Pipeline
--------
1. Discover PDFs under ``data/documents/`` in sorted order (reproducible).
2. Extract page text with pypdf and keep the source filename as metadata.
3. Split text into overlapping word chunks from ``src/config.py``.
4. Embed chunks locally with Sentence Transformers.
5. Build a FAISS inner-product index and write ``data/index/``.

Architectural notes are inline where a decision would be easy to miss.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from src import config
from src.chunking import chunk_pages
from src.generate import redact_secrets
from src.models import Chunk, DocumentFailure, IngestionResult

# Third-party imports live in helper modules. If this interpreter is not the
# project .venv, they fail with ModuleNotFoundError (e.g. no pypdf).
try:
    from src.documents import list_unsupported_files, load_documents
    from src.embeddings import EmbeddingError, embed_texts, get_embedding_model
    from src.indexer import IndexWriteError, save_index
except ModuleNotFoundError as exc:
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"
    print(
        "A project dependency is missing from this Python interpreter.",
        file=sys.stderr,
    )
    print(f"  interpreter: {sys.executable}", file=sys.stderr)
    print(f"  missing: {exc.name or exc}", file=sys.stderr)
    print("Installations live in .venv. Run:", file=sys.stderr)
    print(f"  {venv_python} -m src.ingest", file=sys.stderr)
    raise SystemExit(1) from None

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """User-facing ingestion failure (printed without a stack trace)."""


def configure_logging() -> None:
    """Configure process-wide logging. Never logs secrets or API keys."""
    level_name = config.LOG_LEVEL
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def validate_documents_dir(documents_dir: Path) -> None:
    """Fail clearly when the PDF input directory is missing or not a folder."""
    if documents_dir.is_file():
        raise IngestionError(
            f"Documents path exists but is a file, not a directory: {documents_dir}"
        )
    if not documents_dir.exists():
        raise IngestionError(
            "Documents directory is missing: "
            f"{documents_dir}\n"
            "Create it, add one or more PDF files, then run: python -m src.ingest"
        )


def indexable_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """Drop empty chunks so FAISS is not filled with meaningless vectors.

    Empty pages are still *handled* (counted and logged), not silently ignored.
    They are just not useful for semantic search.
    """
    kept: list[Chunk] = []
    skipped = 0
    for chunk in chunks:
        if chunk.text.strip():
            kept.append(chunk)
        else:
            skipped += 1
            logger.warning(
                "Empty extracted text skipped: %s page %s (%s)",
                chunk.source,
                chunk.page,
                chunk.chunk_id,
            )
    return kept, skipped


def run_ingestion(
    documents_dir: Path | None = None,
    index_dir: Path | None = None,
    embedding_model_name: str | None = None,
) -> IngestionResult:
    """Run the full local ingestion pipeline and return statistics."""
    started = time.perf_counter()
    documents_dir = documents_dir or config.DOCUMENTS_DIR
    index_dir = index_dir or config.INDEX_DIR
    model_name = embedding_model_name or config.EMBEDDING_MODEL_NAME
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    logger.info("Ingestion start (local embeddings only; Groq is not used)")
    validate_documents_dir(documents_dir)

    # Non-PDF files are ignored, but listed so a misplaced .docx/.txt is visible.
    unsupported = list_unsupported_files(documents_dir)
    for relative in unsupported:
        logger.warning("Unsupported file ignored: %s", relative)

    pages, failures, discovered = load_documents(documents_dir)

    # A PDF that opens but yields no text is a handled failure, not a crash.
    nonempty_pages = [page for page in pages if page.text.strip()]
    empty_page_count = len(pages) - len(nonempty_pages)
    sources_with_text = {page.source for page in nonempty_pages}
    empty_sources = sorted(
        {page.source for page in pages if page.source not in sources_with_text}
    )
    for source in empty_sources:
        reason = "Empty extracted text (no searchable content)"
        logger.error("Failed to process %s: %s", source, reason)
        failures.append(DocumentFailure(path=source, reason=reason))

    # Overlapping word windows keep sentences near chunk boundaries retrievable.
    # Chunk IDs are deterministic: {filename_stem}_p{page}_c{index:02d}.
    chunks = chunk_pages(nonempty_pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    searchable, empty_skipped = indexable_chunks(chunks)
    logger.info(
        "Created %s searchable chunk(s) from %s page(s) (skipped %s empty)",
        len(searchable),
        len(nonempty_pages),
        empty_skipped + empty_page_count,
    )

    processed = len({page.source for page in nonempty_pages})
    result = IngestionResult(
        documents_discovered=discovered,
        documents_processed=processed,
        pages_processed=len(pages),
        chunks_created=len(searchable),
        embedding_model=model_name,
        failures=failures,
        unsupported_files=unsupported,
        empty_pages_skipped=empty_page_count + empty_skipped,
    )

    if not searchable:
        logger.warning("No searchable chunks produced; skipping FAISS index")
        result.skipped_index = True
        result.elapsed_seconds = time.perf_counter() - started
        return result

    # Local Sentence Transformer (all-MiniLM-L6-v2 by default). Vectors are
    # L2-normalized so FAISS IndexFlatIP is cosine similarity, not L2 distance.
    try:
        model = get_embedding_model(model_name)
        embeddings = embed_texts(
            texts=[chunk.text for chunk in searchable],
            model=model,
        )
    except EmbeddingError as exc:
        raise IngestionError(f"Embedding failure: {exc}") from exc

    result.embedding_dimension = int(embeddings.shape[1])

    try:
        save_index(
            embeddings,
            searchable,
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_model=model_name,
        )
    except IndexWriteError as exc:
        raise IngestionError(str(exc)) from exc

    result.index_path = index_path
    result.metadata_path = metadata_path
    result.elapsed_seconds = time.perf_counter() - started
    logger.info(
        "Ingestion complete: %s docs, %s pages, %s chunks, %.2fs",
        result.documents_processed,
        result.pages_processed,
        result.chunks_created,
        result.elapsed_seconds,
    )
    return result


def format_report(result: IngestionResult) -> str:
    """Build the user-facing ingestion summary."""
    index_line = str(result.index_path) if result.index_path else "(not created)"
    metadata_line = (
        str(result.metadata_path) if result.metadata_path else "(not created)"
    )
    lines = [
        "============================================================",
        "ASTRION RAG - DOCUMENT INGESTION",
        "============================================================",
        "",
        f"Documents processed:       {result.documents_processed}",
        f"Pages processed:           {result.pages_processed}",
        f"Chunks created:            {result.chunks_created}",
        f"Embedding dimension:       {result.embedding_dimension}",
        f"Ingestion time:            {result.elapsed_seconds:.2f} seconds",
        "",
        f"Documents discovered:      {result.documents_discovered}",
        f"Embedding model:           {result.embedding_model}",
        "",
        "Index:",
        index_line,
        "",
        "Metadata:",
        metadata_line,
    ]

    if result.unsupported_files:
        lines.extend(["", "Unsupported files (ignored):"])
        for path in result.unsupported_files:
            lines.append(f"  - {path}")

    if result.empty_pages_skipped:
        lines.extend(
            [
                "",
                f"Empty extracted pages skipped: {result.empty_pages_skipped}",
            ]
        )

    if result.failures:
        lines.extend(["", "Failed documents:"])
        for failure in result.failures:
            lines.append(f"  - {failure.path}: {failure.reason}")

    if result.skipped_index:
        lines.extend(
            [
                "",
                "No searchable chunks were produced.",
                "Add readable PDF files to data/documents/ and run ingestion again.",
            ]
        )

    lines.extend(
        [
            "",
            "============================================================",
            "INGESTION COMPLETE",
            "============================================================",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entry point for ``python -m src.ingest``."""
    configure_logging()
    try:
        result = run_ingestion()
    except IngestionError as exc:
        logger.error("%s", redact_secrets(str(exc)))
        print(f"Ingestion failed: {redact_secrets(str(exc))}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        if config.DEBUG:
            raise
        logger.error("Unexpected ingestion error: %s", redact_secrets(str(exc)))
        print("Ingestion failed: an unexpected error occurred.", file=sys.stderr)
        return 1

    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
