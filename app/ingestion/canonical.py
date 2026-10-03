"""Canonical corpus-pack documents -> verse-level records.

A canonical document is one source edition written as structured Markdown under the contract in
data/packs/README.md: front matter, `## Heading` units each tagged `<!-- ref: text.unit -->`,
and `### Part` blocks (Mantra, Translation, Commentary, Notes, ...). This module parses those
files into records that carry the unit reference and all provenance metadata, so retrieval can
cite "Isha 2 · Commentary · Shankara, tr. Gambhirananda" and filter by commentator or layer.

No third-party dependencies: the manifest is TOML (stdlib tomllib), the output is JSON lines.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MAX_WORDS = 350

# ----------------------------------------------------------------------------- front matter


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split `---` fenced front matter from the body. Values may be bare, quoted or JSON arrays."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict = {}
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1 :])
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, raw = line.partition(":")
        meta[key.strip()] = _parse_value(raw.strip())
    return meta, ""


def _parse_value(raw: str):
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


# ----------------------------------------------------------------------------- units & parts


@dataclass
class Part:
    name: str
    text: str


@dataclass
class Unit:
    ref: str
    heading: str
    meta: dict = field(default_factory=dict)
    parts: list[Part] = field(default_factory=list)

    @property
    def unit_id(self) -> str:
        return self.ref.split(".", 1)[1] if "." in self.ref else self.ref


_REF_RE = re.compile(r"<!--\s*ref:\s*([A-Za-z0-9_.\-]+)\s*(?:\|(.*?))?-->")


def parse_units(body: str) -> list[Unit]:
    units: list[Unit] = []
    unit: Unit | None = None
    part: Part | None = None
    buf: list[str] = []

    def flush_part():
        nonlocal part, buf
        if unit is not None and part is not None:
            part.text = "\n".join(buf).strip("\n")
            unit.parts.append(part)
        part = None
        buf = []

    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            flush_part()
            heading = line[3:].strip()
            # the ref comment must be the next non-blank line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            m = _REF_RE.search(lines[j]) if j < len(lines) else None
            if not m:
                raise ValueError(f"unit '{heading}' has no <!-- ref: ... --> line")
            meta: dict = {}
            for kv in (m.group(2) or "").split("|"):
                if ":" in kv:
                    k, _, v = kv.partition(":")
                    meta[k.strip()] = v.strip()
            unit = Unit(ref=m.group(1), heading=heading, meta=meta)
            units.append(unit)
            i = j + 1
            continue
        if line.startswith("### ") and unit is not None:
            flush_part()
            part = Part(name=line[4:].strip(), text="")
            i += 1
            continue
        if unit is not None and part is not None:
            buf.append(line)
        i += 1
    flush_part()
    return units


def part_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ----------------------------------------------------------------------------- chunking


def split_into_chunks(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split at paragraph boundaries so no chunk exceeds max_words; an oversize paragraph is
    split on sentence boundaries. Chunks never cut a sentence in half."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paras:
        if len(p.split()) <= max_words:
            pieces.append(p)
        else:
            pieces.extend(_split_sentences(p, max_words))
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for piece in pieces:
        w = len(piece.split())
        if cur and cur_words + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, cur_words = [], 0
        cur.append(piece)
        cur_words += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _split_sentences(text: str, max_words: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?।॥])\s+", text)
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


# ----------------------------------------------------------------------------- cross-references

_ABBR = {
    "br": "br", "brh": "br", "ch": "ch", "cha": "ch", "tai": "tai", "ka": "ka", "kath": "ka",
    "ke": "ke", "mu": "mu", "mun": "mu", "ma": "ma", "pr": "pr", "ai": "ai", "sv": "sv",
    "śv": "sv", "is": "isha", "īś": "isha", "isa": "isha",
}
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def _num(tok: str) -> int:
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    total, prev = 0, 0
    for ch in reversed(tok):
        val = _ROMAN[ch]
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


_ABBR_ALT = "|".join(sorted((re.escape(a) for a in _ABBR), key=len, reverse=True))
_THREE = re.compile(rf"\b({_ABBR_ALT})\.\s*([IVXLCivxlc]+|\d+)\.\s*([IVXLCivxlc]+|\d+)\.\s*(\d+)(?:-\d+)?", re.IGNORECASE)
_TWO = re.compile(rf"\b({_ABBR_ALT})\.\s*([IVXLCivxlc]+|\d+)\.\s*(\d+)(?!\.\s*\d)", re.IGNORECASE)
_GITA = re.compile(r"\bG[īi]t[āa],?\s*([IVXLCivxlc]+|\d+)\.\s*(\d+)")
_MBH = re.compile(r"\bMbh\.\s*[ŚS][āa]ntiparva,?\s*(\d+)\.(\d+)")
_INTERNAL = re.compile(r"\((\d{1,2})\)")


def extract_crossrefs(text: str, internal_text: str | None = None, max_internal: int = 18) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def add(r: str):
        if r not in seen:
            seen.add(r)
            refs.append(r)

    consumed = text
    for m in _THREE.finditer(text):
        add(f"{_ABBR[m.group(1).lower()]}.{_num(m.group(2))}.{_num(m.group(3))}.{int(m.group(4))}")
        consumed = consumed.replace(m.group(0), " ")
    for m in _TWO.finditer(consumed):
        add(f"{_ABBR[m.group(1).lower()]}.{_num(m.group(2))}.{int(m.group(3))}")
    for m in _GITA.finditer(text):
        add(f"gita.{_num(m.group(1))}.{int(m.group(2))}")
    for m in _MBH.finditer(text):
        add(f"mbh.santi.{m.group(1)}.{m.group(2)}")
    if internal_text:
        for m in _INTERNAL.finditer(text):
            n = int(m.group(1))
            if 1 <= n <= max_internal:
                add(f"{internal_text}.{n}")
    return refs


# ----------------------------------------------------------------------------- records


def _unit_sort_key(unit_id: str) -> tuple:
    if unit_id.isdigit():
        return (1, float(unit_id))
    if re.fullmatch(r"\d+\.\d+", unit_id):
        a, b = unit_id.split(".")
        return (1, float(a) + float(b) / 1000)
    return {"intro": (1, 0.5), "epilogue": (2, 0), "colophon": (2, 1), "notes": (2, 2)}.get(unit_id, (3, 0))


def _citation_unit(cite_as: str, unit: Unit) -> str:
    uid = unit.unit_id
    if uid.isdigit() and uid != "0" or re.fullmatch(r"\d+\.\d+", uid):
        return f"{cite_as} {uid}"
    return f"{cite_as}, {unit.heading}"


def _source_label(meta: dict) -> str:
    c, t = meta.get("commentator"), meta.get("translator")
    if c and t and c != t:
        return f"{c}, tr. {t}"
    if c:
        return str(c)
    if t:
        return f"tr. {t}"
    return "mula"


def _notes_text(text: str) -> str:
    out = []
    for line in text.splitlines():
        m = re.match(r"\[\^(\w+)\]:\s*(.*)", line.strip())
        out.append(f"[{m.group(1)}] {m.group(2)}" if m else line)
    return "\n".join(out).strip()


def build_records_from_text(text: str, doc_slug: str, cite_as: str | None = None, seq_start: int = 0) -> list[dict]:
    meta, body = parse_front_matter(text)
    units = parse_units(body)
    text_id = str(meta.get("text_id", doc_slug.split(".")[0]))
    cite_as = cite_as or text_id.capitalize()
    title = str(meta.get("title", text_id))
    src = _source_label(meta)
    records: list[dict] = []
    seq = seq_start
    for unit in units:
        for part in unit.parts:
            pkey = part_key(part.name)
            body_text = _notes_text(part.text) if pkey == "notes" else part.text.strip()
            if not body_text:
                continue
            chunks = split_into_chunks(body_text) if pkey != "notes" else [body_text]
            internal = text_id if pkey not in ("notes", "mantra_devanagari", "mantra_iast") else None
            for ci, chunk in enumerate(chunks):
                rid = f"{doc_slug}:{unit.ref}:{pkey}" + (f":{ci}" if len(chunks) > 1 else "")
                header = f"{title} | {unit.heading} | {part.name} | {src}"
                records.append(
                    {
                        "id": rid,
                        "pack": meta.get("pack"),
                        "text_id": text_id,
                        "text_title": title,
                        "ref": unit.ref,
                        "unit": unit.unit_id,
                        "unit_label": unit.heading,
                        "unit_meta": unit.meta,
                        "part": pkey,
                        "part_label": part.name,
                        "layer": meta.get("layer"),
                        "language": meta.get("language"),
                        "school": meta.get("school"),
                        "commentator": meta.get("commentator"),
                        "translator": meta.get("translator"),
                        "doc_slug": doc_slug,
                        "source_file": meta.get("source_file") or meta.get("source"),
                        "citation": f"{_citation_unit(cite_as, unit)} · {part.name} · {src}",
                        "header": header,
                        "text": chunk,
                        "crossrefs": extract_crossrefs(chunk, internal_text=internal),
                        "chunk_index": ci,
                        "chunk_count": len(chunks),
                        "seq": seq,
                    }
                )
                seq += 1
    return records


def load_manifest(pack_dir: Path) -> dict:
    with open(pack_dir / "pack.toml", "rb") as f:
        return tomllib.load(f)


def build_pack(pack_dir: str | Path) -> list[dict]:
    pack_dir = Path(pack_dir)
    manifest = load_manifest(pack_dir)
    records: list[dict] = []
    for text in manifest.get("texts", []):
        for rel in text["files"]:
            path = pack_dir / rel
            recs = build_records_from_text(
                path.read_text(encoding="utf-8"), doc_slug=path.stem, cite_as=text.get("cite_as"), seq_start=len(records)
            )
            for r in recs:
                r["pack"] = r["pack"] or manifest["pack"]["id"]
            records.extend(recs)
    return records


def build_toc(records: list[dict]) -> dict:
    texts: dict[str, dict] = {}
    for r in records:
        t = texts.setdefault(r["text_id"], {"title": r["text_title"], "units": {}})
        u = t["units"].setdefault(
            r["ref"], {"ref": r["ref"], "unit": r["unit"], "label": r["unit_label"], "parts": {}, "sources": [], "meta": {}}
        )
        u["parts"].setdefault(r["part"], []).append(r["id"])
        if r["doc_slug"] not in u["sources"]:
            u["sources"].append(r["doc_slug"])
        if r["unit_meta"]:
            u["meta"][r["doc_slug"]] = r["unit_meta"]
        # the mula file (layer shruti) names the text; a commentary file's title is its own
        if r.get("layer") == "shruti":
            t["title"] = r["text_title"]
    for t in texts.values():
        t["units"] = sorted(t["units"].values(), key=lambda u: _unit_sort_key(u["unit"]))
    pack = next((r["pack"] for r in records if r.get("pack")), None)
    return {"pack": pack, "texts": texts, "records": len(records)}
