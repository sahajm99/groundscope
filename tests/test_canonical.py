"""Canonical corpus-pack documents -> verse-level records (data/packs/README.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion import canonical

PACK = Path("data/packs/vedanta")
ISHA = PACK / "texts" / "isha"

# The Vedanta texts are private (data/packs/README.md, "Distribution"): present on the author's
# machine, absent from a clone. Tests that read them skip there; the demo-pack tests here and in
# test_storage.py cover the same build and seed paths everywhere.
needs_pack_texts = pytest.mark.skipif(
    not ISHA.is_dir(), reason="the Vedanta pack's texts are not distributed with the repository"
)

SAMPLE = """---
pack: demo
text_id: demo
title: "A demo text"
layer: bhashya
commentator: Shankara
translator: "Swami X"
language: en
tags: ["a", "b"]
---

# Ignored title

Preface prose that the parser must ignore.

## Introduction
<!-- ref: demo.intro -->

### Commentary

General introduction paragraph.

## Verse 1
<!-- ref: demo.1 | pdf_pages: 3-4 -->

### Translation

1. The verse rendered.

### Commentary

First paragraph citing (Br. I. v. 16) and verse (7).

Second paragraph citing (Tai. II. viii. 1) and the Gita, XVII. 2, and (Śv. IV. 10).

### Notes

[^1]: A footnote with (1) a list and (Ka. I. ii. 4).
[^2]: Another.
"""


def test_front_matter_parses_bare_quoted_and_array_values():
    meta, body = canonical.parse_front_matter(SAMPLE)
    assert meta["pack"] == "demo"
    assert meta["title"] == "A demo text"
    assert meta["translator"] == "Swami X"
    assert meta["tags"] == ["a", "b"]
    assert body.lstrip().startswith("# Ignored title")


def test_units_and_parts_are_recovered_with_ref_and_unit_meta():
    _, body = canonical.parse_front_matter(SAMPLE)
    units = canonical.parse_units(body)
    assert [u.ref for u in units] == ["demo.intro", "demo.1"]
    assert units[1].heading == "Verse 1"
    assert units[1].meta == {"pdf_pages": "3-4"}
    assert [p.name for p in units[1].parts] == ["Translation", "Commentary", "Notes"]
    assert units[1].parts[0].text.strip() == "1. The verse rendered."


def test_part_names_map_to_stable_keys():
    assert canonical.part_key("Mantra (Devanagari)") == "mantra_devanagari"
    assert canonical.part_key("Mantra (IAST)") == "mantra_iast"
    assert canonical.part_key("Commentary") == "commentary"
    assert canonical.part_key("Encoder's notes") == "encoder_s_notes"


def test_records_carry_metadata_citation_header_and_crossrefs():
    recs = canonical.build_records_from_text(SAMPLE, doc_slug="demo.shankara.x", cite_as="Demo")
    by = {(r["ref"], r["part"]): r for r in recs}
    c = by[("demo.1", "commentary")]
    assert c["id"] == "demo.shankara.x:demo.1:commentary"
    assert c["pack"] == "demo" and c["text_id"] == "demo" and c["unit"] == "1"
    assert c["layer"] == "bhashya" and c["commentator"] == "Shankara" and c["translator"] == "Swami X"
    assert c["unit_meta"] == {"pdf_pages": "3-4"}
    assert c["citation"] == "Demo 1 · Commentary · Shankara, tr. Swami X"
    assert c["header"].startswith("A demo text | Verse 1 | Commentary")
    assert set(c["crossrefs"]) == {"br.1.5.16", "demo.7", "tai.2.8.1", "gita.17.2", "sv.4.10"}
    # internal (n) refs are not extracted from footnotes, where "(1)" is a list marker
    n = by[("demo.1", "notes")]
    assert "demo.1" not in n["crossrefs"] and "ka.1.2.4" in n["crossrefs"]
    assert n["text"].startswith("[1] A footnote")
    intro = by[("demo.intro", "commentary")]
    assert intro["citation"] == "Demo, Introduction · Commentary · Shankara, tr. Swami X"


def test_long_parts_split_at_paragraph_boundaries_into_stable_chunks():
    para = " ".join(["word"] * 150)
    text = "\n\n".join([para, para, para])  # 450 words -> [300, 150] at 350: never cut a paragraph
    chunks = canonical.split_into_chunks(text, max_words=350)
    assert [len(c.split()) for c in chunks] == [300, 150]
    assert chunks[0].count("\n\n") == 1 and chunks[1].count("\n\n") == 0
    # a single oversize paragraph is split on sentence boundaries
    big = ". ".join([f"sentence number {i}" for i in range(300)]) + "."
    parts = canonical.split_into_chunks(big, max_words=350)
    assert len(parts) >= 2 and all(len(p.split()) <= 350 for p in parts)


def test_roman_numerals_and_abbreviations_normalise():
    assert canonical.extract_crossrefs("(Br. IV. iv. 22)") == ["br.4.4.22"]
    assert canonical.extract_crossrefs("(Br. 5.1.1.)") == ["br.5.1.1"]
    assert canonical.extract_crossrefs("(Mbh. Śāntiparva, 241.6)") == ["mbh.santi.241.6"]
    assert canonical.extract_crossrefs("(Śv. VI. 21)") == ["sv.6.21"]
    assert canonical.extract_crossrefs("(Mu. III. ii. 9)") == ["mu.3.2.9"]
    assert canonical.extract_crossrefs("see (8) and (11)", internal_text="isha") == ["isha.8", "isha.11"]
    assert canonical.extract_crossrefs("(19)", internal_text="isha") == []  # verses stop at 18


@needs_pack_texts
@pytest.mark.parametrize(
    "name, units, parts",
    [
        ("isha.md", 19, {"mantra_devanagari", "mantra_iast"}),
        ("isha.shankara.gambhirananda.md", 21, {"translation", "commentary"}),
        ("isha.aurobindo.md", 18, {"translation"}),
        ("isha.shankara.sa.md", 22, {"commentary"}),  # intro + 18 verses + epilogue, colophon, notes
    ],
)
def test_real_isha_files_parse_completely(name, units, parts):
    text = (ISHA / name).read_text(encoding="utf-8")
    _, body = canonical.parse_front_matter(text)
    us = canonical.parse_units(body)
    assert len(us) == units, [u.ref for u in us]
    verse_units = [u for u in us if u.unit_id.isdigit() and u.unit_id != "0"]
    assert [int(u.unit_id) for u in verse_units] == list(range(1, 19))
    for u in verse_units:
        keys = {canonical.part_key(p.name) for p in u.parts}
        assert parts <= keys, (name, u.ref, keys)


@needs_pack_texts
def test_gambhirananda_footnotes_are_captured_per_unit():
    text = (ISHA / "isha.shankara.gambhirananda.md").read_text(encoding="utf-8")
    recs = canonical.build_records_from_text(text, doc_slug="isha.shankara.gambhirananda", cite_as="Isha")
    notes = {r["ref"]: r["text"] for r in recs if r["part"] == "notes"}
    assert "isha.9" in notes and "[3]" in notes["isha.9"]
    assert "[1]" in notes["isha.epilogue"] and "A.G." in notes["isha.epilogue"]
    v2 = [r for r in recs if r["ref"] == "isha.2" and r["part"] == "commentary"]
    assert v2 and "irremovable like a mountain" in " ".join(r["text"] for r in v2)
    assert "isha.7" in [x for r in recs if r["ref"] == "isha.9" and r["part"] == "introduction" for x in r["crossrefs"]]


@needs_pack_texts
def test_toc_orders_units_in_reading_order_and_joins_files():
    recs = canonical.build_pack(PACK)
    toc = canonical.build_toc(recs)
    isha = toc["texts"]["isha"]
    order = [u["unit"] for u in isha["units"]]
    assert order[:3] == ["0", "intro", "1"] and order[-4:] == ["18", "epilogue", "colophon", "notes"]
    v2 = next(u for u in isha["units"] if u["unit"] == "2")
    assert {"isha", "isha.shankara.sa", "isha.shankara.gambhirananda", "isha.aurobindo"} <= set(v2["sources"])
    assert "mantra_iast" in v2["parts"] and "commentary" in v2["parts"]


@needs_pack_texts
def test_build_pack_writes_jsonl_and_toc(tmp_path):
    from scripts.build_pack import main

    out = main(str(PACK), out_dir=str(tmp_path))
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(line) for line in lines]
    assert len(recs) == out["records"] and len(recs) > 100
    assert len({r["id"] for r in recs}) == len(recs), "record ids must be unique"
    toc = json.loads((tmp_path / "toc.json").read_text(encoding="utf-8"))
    assert toc["pack"] == "vedanta" and "isha" in toc["texts"]


def test_a_pack_builds_from_its_manifest_into_records_and_a_toc(demo_pack, tmp_path):
    from scripts.build_pack import main

    out = main(str(demo_pack), out_dir=str(tmp_path / "build"))
    lines = (tmp_path / "build" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(line) for line in lines]
    assert len(recs) == out["records"] == 3  # verse 1: translation and commentary; verse 2: translation
    assert len({r["id"] for r in recs}) == len(recs), "record ids must be unique"
    assert {r["pack"] for r in recs} == {"demo"}
    assert recs[0]["citation"].startswith("Demo 1 · Translation")
    toc = json.loads((tmp_path / "build" / "toc.json").read_text(encoding="utf-8"))
    assert toc["pack"] == "demo"
    assert [u["unit"] for u in toc["texts"]["demo"]["units"]] == ["1", "2"]
