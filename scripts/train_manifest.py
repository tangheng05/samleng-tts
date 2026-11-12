"""Join checked teacher batches into one training manifest, English in both scripts.

uv run python scripts/train_manifest.py v1c v2c v3c v4c v5c --out outputs/train/distill_v2_clear/manifest.jsonl

Takes every kept clip of outputs/distill/<batch>/manifest.jsonl. A clip with English words gets
a second row with the same audio and the words in Khmer script (samleng.data.both_scripts).
Rows the model cannot read (a character outside its alphabet) are dropped and counted.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from samleng.data import training_rows
from samleng.paths import OUTPUT_DIR, ROOT
from samleng.runners import read_jsonl, write_jsonl
from samleng.text import EN_KM, train_words


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
    batch_rows = [
        r | {"batch": batch}
        for batch in args.batches
        for r in read_jsonl(OUTPUT_DIR / "distill" / batch / "manifest.jsonl")
    ]
    try:
        rows, dropped = training_rows(batch_rows, words, banned)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    # Each clip counts once, however many rows it has.
    seconds = sum(r["seconds"] for r in rows if not r["id"].endswith("-km"))
    write_jsonl(args.out, rows)
    khmer = sum(r["id"].endswith("-km") for r in rows)
    print(
        f"{len(rows)} rows ({khmer} Khmer-script copies), {seconds / 3600:.2f} h of audio, "
        f"left out: {dict(dropped)} -> {args.out}"
    )
    print(dict(Counter(r["source"] for r in rows)))


if __name__ == "__main__":
    main()
