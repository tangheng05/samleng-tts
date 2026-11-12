"""Join checked teacher batches into one training manifest, English in both scripts.

uv run python scripts/train_manifest.py v1c v2c v3c v4c v5c --out outputs/train/distill_v2_clear/manifest.jsonl

Takes every kept clip of outputs/distill/<batch>/manifest.jsonl. A clip with English words gets
a second row with the same audio and the words in Khmer script (samleng.data.both_scripts).
Rows the model cannot read (a character outside its alphabet) are dropped and counted.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from samleng.data import both_scripts, english_words
from samleng.paths import OUTPUT_DIR, ROOT
from samleng.runners import read_jsonl, write_jsonl
from samleng.text import EN_KM, train_words

# sengtha's alphabet after its cleaner lowercases the text: Khmer block, a-z, its punctuation.
UNREADABLE = re.compile(r"[^ក-៿a-z!,.?:;()\"'«»“”\-–—%/ ]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    split = json.loads(
        (ROOT / "bench" / "khmertts_bench_v1_mixed_split.json").read_text()
    )
    banned = set(split["banned"])
    words = {**EN_KM, **train_words()}
    rows, seconds, dropped = [], 0.0, Counter()
    for batch in args.batches:
        for r in read_jsonl(OUTPUT_DIR / "distill" / batch / "manifest.jsonl"):
            if "reject" in r:
                dropped[r["reject"]] += 1
                continue
            english = english_words(r["text"])
            # Same rule as distill_problems: "poster" would teach the held-out "post".
            if any(b in w for w in english for b in banned if w == b or len(b) > 3):
                raise SystemExit(f"{r['id']}: held-out English word in training text")
            versions = both_scripts(r, words)
            if any(UNREADABLE.search(v["text"].lower()) for v in versions):
                dropped["unreadable"] += 1
                continue
            seconds += r["seconds"]
            rows += [v | {"batch": batch} for v in versions]
    ids = Counter(r["id"] for r in rows)
    if clash := [i for i, n in ids.items() if n > 1]:
        raise SystemExit(f"duplicate ids: {clash[:5]}")
    write_jsonl(args.out, rows)
    khmer = sum(r["id"].endswith("-km") for r in rows)
    print(
        f"{len(rows)} rows ({khmer} Khmer-script copies), {seconds / 3600:.2f} h of audio, "
        f"left out: {dict(dropped)} -> {args.out}"
    )
    print(dict(Counter(r["source"] for r in rows)))


if __name__ == "__main__":
    main()
