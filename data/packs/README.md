# Corpus packs

A corpus pack is a versioned set of **canonical documents** plus a manifest. Groundscope's
code is the golden image; a pack is the knowledge it is pointed at. Swapping the pack swaps
the domain without touching the agent. The first pack is `vedanta/` (Phase 1: the Isha
Upanishad with Shankara's commentary).

```
data/packs/<pack>/
  pack.toml            manifest: pack settings + the list of canonical files per text
  texts/<text>/*.md    canonical documents, one per source edition
  golden.jsonl         pack-specific golden questions (same format as evals/golden.jsonl)
  build/records.jsonl  generated: one record per addressable unit part (or chunk of one)
  build/toc.json       generated: the pack's table of contents (text -> unit -> parts -> records)
```

Build with `python -m scripts.build_pack data/packs/<pack>`; seed the build into the database
with `python -m scripts.seed_pack data/packs/<pack>`.

## Distribution

This repository is public, and a pack's texts are other people's work. `data/packs/*/texts/` and
`data/packs/*/build/` are gitignored: the canonical files and everything built from them stay on
the machine that made them. What is tracked is the contract (this file), each pack's manifest
(`pack.toml`) and its golden set.

For the Vedanta pack that means all four Isha files stay local, together. The mantras and Sri
Aurobindo's translation are in the public domain, but Swami Gambhirananda's translation is
Advaita Ashrama's copyright and the Sanskrit bhashya encoding carries a no-repost notice, and
the pack is kept whole rather than split. Each file records its own `license` in its front
matter. A public deployment must seed only texts it has the right to serve (DECISIONS 33, 35).

Tests that read the real Vedanta texts skip when the folder is absent. A two-verse synthetic
pack in `tests/conftest.py` keeps the build and seed path under test everywhere.

## Why a canonical document, not the PDF

A PDF is a *rendering*. What gets indexed is the *text*, recovered once, cleaned once, and
given its own structure back: verse, chapter, sutra. Every addressable unit carries a stable
reference and metadata (which text, which translator, which commentator, which school), so a
citation can say "Isha 2, Shankara's commentary (tr. Gambhirananda)" instead of "file p.14",
and a query can ask for one commentator, one layer, or one verse across all of them.

One canonical file mirrors **one source edition**. The Sanskrit mantras, a translation, and a
commentary are three files even when one book prints them together, because provenance,
language and license differ per source. They join on the unit `ref`.

## The document contract

```markdown
---
pack: vedanta
text_id: isha
title: "Isha Upanishad with Shankara's commentary (English)"
layer: bhashya                # shruti | smriti | bhashya | modern
commentator: Shankara         # omit for a plain translation
translator: "Swami Gambhirananda"
school: advaita
language: en
structure: verse
source_file: "…/Eight Upanishads with Shankara Bhashya (2 vols) - Swami Gambhirananda.pdf"
license: "…"
---

# Human title (ignored by the parser)

Prose before the first `##` is a preface for readers; the parser ignores it.

## Verse 2                                   <- one addressable unit
<!-- ref: isha.2 | pdf_pages: 19-21 -->      <- required ref; optional key: value pairs

### Introduction                             <- a part of the unit
Shankara's bridge passage into this verse.

### Translation
2. By doing karma, indeed, …

### Commentary
Kurvan eva iha, verily by doing here … with footnote markers like this.[^1]

### Notes
[^1]: The footnote text.
```

Rules:

- **Front matter** is `key: value` per line between `---` fences. Values may be bare, quoted, or
  a JSON array. Keys used by the build: `pack`, `text_id`, `title`, `layer`, `language`,
  `commentator`, `translator`, `school`, `source_file`, `source_pages`, `license`.
- **`## Heading`** opens a unit. The very next non-blank line must be an HTML comment
  `<!-- ref: <text_id>.<unit> [| key: value]* -->`. Unit ids: `0` for an invocation, `intro`
  for a commentator's general introduction, `1`…`n` for verses (or `2.47` for chapter.verse),
  `epilogue`, `colophon`, `notes`. A unit may appear in many files; that is the join key.
- **`### Part`** opens a part. Recognised names: `Mantra (Devanagari)`, `Mantra (IAST)`,
  `Introduction` (the commentator's avataraṇikā for *this* verse), `Translation`, `Commentary`,
  `Notes`, `Analysis`. Anything else is kept under a slug of its name.
- **Footnotes** live in `### Notes` as `[^n]: text`, one per line, and are referenced in the
  body with `[^n]`. Footnote numbering restarts per unit.
- **Bold lines** inside a part (`**आक्षेपः**`) are sub-headings carried over from the source.
- Keep the source's wording. Fix OCR errors and restore diacritics; do not paraphrase, do not
  modernise punctuation inside quoted translations, and never "correct" a scripture from memory.
- Record what you could not verify in the front matter (`coverage`, `conventions`).

## What the build produces

`records.jsonl`: one record per part; a long part is split at paragraph boundaries into chunks
of at most ~350 words, each chunk a record with `chunk_index`/`chunk_count`. Every record has:

| field | example |
|---|---|
| `id` | `isha.shankara.gambhirananda:isha.2:commentary:1` |
| `pack`, `text_id`, `ref`, `unit`, `unit_label` | `vedanta`, `isha`, `isha.2`, `2`, `Verse 2` |
| `part` | `commentary` |
| `layer`, `language`, `school` | `bhashya`, `en`, `advaita` |
| `commentator`, `translator` | `Shankara`, `Swami Gambhirananda` |
| `doc_slug`, `source_file`, `unit_meta` | file stem, the PDF path, `{"pdf_pages": "19-21"}` |
| `citation` | `Isha 2 · Commentary · Shankara, tr. Swami Gambhirananda` |
| `header` | contextual header prepended before embedding |
| `text` | the chunk text |
| `crossrefs` | `["br.1.5.16", "isha.7"]` normalised from `(Br. I. v. 16)`, `(7)` |
| `seq` | global order within the pack |

`toc.json` is the structural index: for each text, its units in reading order, and for each
unit which parts exist in which files. A structural tool (`get_toc`, `get_verse`) navigates
this instead of guessing with similarity search.

## Recovering text from a scan

1. Prefer an existing clean digital text (sanskritdocuments.org, Wikisource, GRETIL) over OCR.
2. If the PDF has a text layer, `pdftotext -layout` page by page keeps page numbers.
3. If it is image-only, take the archive.org `_djvu.txt`, or extract page images with pypdf
   and read them; footnotes are often missing from OCR and have to be read from the images.
4. Note the edition. Re-typeset OCR files can differ in wording from the printed edition.
