"""Text extraction from uploaded files (PDF, DOCX, TXT).

v1 extracts plain text per page. Visual grounding (bounding boxes) is v1.1.
"""

from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass
class Page:
    page_number: int  # 1-indexed
    text: str


class NoTextError(Exception):
    """Raised when a document yields no selectable text (e.g. a scanned PDF)."""


def extract_pages(file_bytes: bytes, file_name: str) -> list[Page]:
    """Extract text pages from a file. Raises NoTextError if nothing usable."""
    lower = file_name.lower()
    if lower.endswith(".pdf"):
        pages = _extract_pdf(file_bytes)
    elif lower.endswith(".docx"):
        pages = _extract_docx(file_bytes)
    elif lower.endswith((".txt", ".md")):
        # A form feed (\f) marks a page break, so a seeded book can cite by chapter
        # ('bhagavad-gita.txt p.2' = chapter 2) instead of every chunk saying p.1.
        text = file_bytes.decode("utf-8", errors="ignore")
        parts = [p for p in text.split("\f") if p.strip()] or [text]
        pages = [Page(i, p) for i, p in enumerate(parts, start=1)]
    else:
        raise NoTextError(f"Unsupported file type: {file_name}")

    if not any(p.text.strip() for p in pages):
        raise NoTextError(
            "This document has no selectable text (a scanned PDF?). "
            "OCR isn't supported in the demo."
        )
    return pages


def _extract_pdf(file_bytes: bytes) -> list[Page]:
    import fitz  # pymupdf

    pages: list[Page] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            pages.append(Page(page_number=i, text=page.get_text("text")))
    return pages


def _extract_docx(file_bytes: bytes) -> list[Page]:
    import docx

    document = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in document.paragraphs)
    return [Page(page_number=1, text=text)]
