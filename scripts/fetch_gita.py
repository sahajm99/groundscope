"""Build the public-domain demo corpus: the Bhagavad-Gita in Sir Edwin Arnold's 1885 verse
translation ("The Song Celestial"), as ONE text file with a form feed between chapters so a
citation reads 'bhagavad-gita.txt p.N' where N is the chapter number.

Usage:  python -m scripts.fetch_gita            # writes data/demo/bhagavad-gita.txt
        python -m scripts.seed data/demo/bhagavad-gita.txt

Source: Project Gutenberg eBook #2388 (public domain in the US). The Gutenberg header and
license boilerplate are stripped; the translation text is unchanged.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/2388/pg2388.txt"
OUT = Path("data/demo/bhagavad-gita.txt")
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV",
         "XV", "XVI", "XVII", "XVIII"]
ATTRIBUTION = ("The Song Celestial, or Bhagavad-Gita (from the Mahabharata): a discourse between "
               "Arjuna and Krishna, translated from the Sanskrit by Sir Edwin Arnold (1885). "
               "Public domain; text from Project Gutenberg eBook #2388.")


def build(raw: str) -> list[tuple[int, str, str]]:
    """Return (chapter number, chapter name, text) for the 18 chapters."""
    raw = raw.replace("\r\n", "\n")
    body = raw[raw.index("\nCHAPTER I\n"): raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")]
    parts = re.split(r"^\s*CHAPTER ([IVX]+)\s*$", body, flags=re.M)  # '', 'I', text, 'II', text...
    chapters = []
    for numeral, text in zip(parts[1::2], parts[2::2], strict=True):
        n = ROMAN.index(numeral) + 1
        # 'Entitled "Sankhya-Yog," / Or "The Book of Doctrines."': the spacing, the trailing
        # comma inside the quotes and a line break inside the second title vary by chapter.
        m = re.search(r'Entitled\s*"([^"]+?)[.,]?"\s*Or\s*"([^"]+?)[.,]?"', text)
        name = f'{" ".join(m.group(1).split())}, "{" ".join(m.group(2).split())}"' if m else ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
        chapters.append((n, name, text))
    return chapters


def render(chapters: list[tuple[int, str, str]]) -> str:
    pages = []
    for n, name, text in chapters:
        head = f"Bhagavad-Gita, Chapter {n}" + (f": {name}" if name else "")
        if n == 1:
            head = ATTRIBUTION + "\n\n" + head
        pages.append(f"{head}\n\n{text}\n")
    return "\f".join(pages)


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
    chapters = build(raw)
    if len(chapters) != 18:
        raise SystemExit(f"expected 18 chapters, found {len(chapters)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(chapters), encoding="utf-8", newline="\n")
    for n, name, text in chapters:
        print(f"chapter {n:>2}: {len(text.split()):>5} words  {name}")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
