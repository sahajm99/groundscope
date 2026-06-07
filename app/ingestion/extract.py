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
        pages = [Page(1, file_bytes.decode("utf-8", errors="ignore"))]
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
