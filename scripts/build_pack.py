"""Build a corpus pack: canonical Markdown -> build/records.jsonl + build/toc.json.

Usage:  python -m scripts.build_pack data/packs/vedanta [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.ingestion.canonical import build_pack, build_toc


def main(pack_dir: str, out_dir: str | None = None) -> dict:
    pack = Path(pack_dir)
    out = Path(out_dir) if out_dir else pack / "build"
    out.mkdir(parents=True, exist_ok=True)
    records = build_pack(pack)
    toc = build_toc(records)
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "toc.json").write_text(json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")
    by_doc = Counter(r["doc_slug"] for r in records)
    by_part = Counter(r["part"] for r in records)
    words = sum(len(r["text"].split()) for r in records)
    summary = {"records": len(records), "words": words, "by_doc": dict(by_doc), "by_part": dict(by_part), "out": str(out)}
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pack_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = main(args.pack_dir, args.out)
    print(f"{s['records']} records, {s['words']} words -> {s['out']}")
    for k, v in sorted(s["by_doc"].items()):
        print(f"  {k:40s} {v:4d} records")
    print("  parts:", ", ".join(f"{k}={v}" for k, v in sorted(s["by_part"].items())))
