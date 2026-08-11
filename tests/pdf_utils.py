"""Minimal PDF writer used by tests and sample-document generation."""

from __future__ import annotations

from pathlib import Path


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_line(text: str, width: int = 90) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word]) if current else word
        if len(trial) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _content_stream(text: str, font_size: int = 12) -> bytes:
    raw_lines = text.splitlines() or [text]
    wrapped: list[str] = []
    for raw in raw_lines:
        wrapped.extend(_wrap_line(raw))
    if not wrapped:
        wrapped = [""]

    commands = ["BT", f"/F1 {font_size} Tf", "50 750 Td"]
    for index, line in enumerate(wrapped[:46]):
        escaped = _escape_pdf_text(line)
        if index:
            commands.append("0 -16 Td")
        commands.append(f"({escaped}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def write_simple_pdf(path: Path, pages: list[str]) -> None:
    """Write a small multi-page Helvetica PDF that pypdf can extract."""
    if not pages:
        raise ValueError("At least one page is required")

    path.parent.mkdir(parents=True, exist_ok=True)
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
    }
    kid_refs = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects[2] = f"<< /Type /Pages /Kids [{kid_refs}] /Count {n_pages} >>".encode("ascii")

    for i, text in enumerate(pages):
        page_id = 3 + 2 * i
        content_id = 4 + 2 * i
        stream = _content_stream(text)
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )

    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    assembled: list[bytes] = []
    offsets = {0: 0}
    cursor = len(header)
    max_id = font_id
    for obj_id in range(1, max_id + 1):
        blob = f"{obj_id} 0 obj\n".encode("ascii") + objects[obj_id] + b"\nendobj\n"
        offsets[obj_id] = cursor
        assembled.append(blob)
        cursor += len(blob)

    xref_lines = [f"xref\n0 {max_id + 1}\n", "0000000000 65535 f \n"]
    for obj_id in range(1, max_id + 1):
        xref_lines.append(f"{offsets[obj_id]:010d} 00000 n \n")
    xref = "".join(xref_lines).encode("ascii")
    trailer = (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{cursor}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(header + b"".join(assembled) + xref + trailer)
