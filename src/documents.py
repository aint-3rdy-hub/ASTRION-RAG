"""PDF discovery and page-level text extraction."""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from src.cleaning import clean_text
from src.models import DocumentFailure, ExtractedPage

logger = logging.getLogger(__name__)


class DocumentExtractionError(Exception):
    """Raised when a PDF cannot be read or parsed."""


def _sorted_files(documents_dir: Path) -> list[Path]:
    """List files under ``documents_dir`` in a stable, case-insensitive order."""
    files = [path for path in documents_dir.rglob("*") if path.is_file()]
    return sorted(
        files,
        key=lambda path: path.relative_to(documents_dir).as_posix().lower(),
    )


def discover_pdfs(documents_dir: Path) -> list[Path]:
    """Return PDF paths under ``documents_dir``, sorted for reproducibility.

    Unsupported file types are ignored here and reported separately.
    A missing or empty directory yields an empty list rather than an exception.
    """
    if not documents_dir.exists() or not documents_dir.is_dir():
        return []

    return [
        path
        for path in _sorted_files(documents_dir)
        if path.suffix.lower() == ".pdf"
    ]


def list_unsupported_files(documents_dir: Path) -> list[str]:
    """Return relative paths of non-PDF files (ignored during ingestion)."""
    if not documents_dir.exists() or not documents_dir.is_dir():
        return []

    skipped: list[str] = []
    for path in _sorted_files(documents_dir):
        if path.suffix.lower() == ".pdf":
            continue
        if path.name == ".gitkeep":
            continue
        skipped.append(relative_source(path, documents_dir))
    return skipped


def relative_source(pdf_path: Path, documents_dir: Path) -> str:
    """Return a stable POSIX-style source path relative to the documents root."""
    try:
        return pdf_path.resolve().relative_to(documents_dir.resolve()).as_posix()
    except ValueError:
        return pdf_path.name


def extract_pdf(pdf_path: Path, documents_dir: Path) -> list[ExtractedPage]:
    """Extract cleaned text from every page of a PDF.

    Pages with little or no text are kept. Failures raise
    ``DocumentExtractionError`` so the caller can record them.
    """
    source = relative_source(pdf_path, documents_dir)
    logger.info("Processing document: %s", source)

    try:
        reader = PdfReader(str(pdf_path))
    except FileNotFoundError as exc:
        raise DocumentExtractionError(f"PDF not found: {source}") from exc
    except (PyPdfError, OSError, ValueError) as exc:
        raise DocumentExtractionError(f"Unreadable or invalid PDF: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - pypdf decrypt APIs vary
            raise DocumentExtractionError("Encrypted PDF could not be opened") from exc
        if not unlocked:
            raise DocumentExtractionError("Encrypted PDF could not be opened")

    pages: list[ExtractedPage] = []
    try:
        total = len(reader.pages)
    except Exception as exc:  # noqa: BLE001
        raise DocumentExtractionError(f"Corrupted PDF structure: {exc}") from exc

    if total == 0:
        raise DocumentExtractionError("Empty or corrupted PDF (no pages)")

    extractable = 0
    for index in range(total):
        page_number = index + 1
        try:
            raw = reader.pages[index].extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - keep other pages
            logger.warning(
                "Page extraction problem in %s page %s: %s",
                source,
                page_number,
                exc,
            )
            raw = ""

        cleaned = clean_text(raw)
        if cleaned:
            extractable += 1
        else:
            logger.warning(
                "No extractable text on %s page %s; keeping empty page",
                source,
                page_number,
            )

        pages.append(ExtractedPage(source=source, page=page_number, text=cleaned))

    if extractable == 0:
        logger.warning("PDF has no extractable text: %s", source)

    logger.info("Extracted %s page(s) from %s", len(pages), source)
    return pages


def load_documents(
    documents_dir: Path,
) -> tuple[list[ExtractedPage], list[DocumentFailure], int]:
    """Discover and extract PDFs.

    Returns ``(pages, failures, discovered_count)``.
    """
    pdfs = discover_pdfs(documents_dir)
    pages: list[ExtractedPage] = []
    failures: list[DocumentFailure] = []

    for pdf_path in pdfs:
        source = relative_source(pdf_path, documents_dir)
        try:
            pages.extend(extract_pdf(pdf_path, documents_dir))
        except DocumentExtractionError as exc:
            logger.error("Failed to process %s: %s", source, exc)
            failures.append(DocumentFailure(path=source, reason=str(exc)))

    return pages, failures, len(pdfs)
