"""Build the sentences VoxCPM2 reads in sengtha's voice (Phase 2 distillation).

uv run python scripts/distill_text.py           # 500 mixed + 1,000 OpenSLR sentences
Writes outputs/distill/v1/tts_in.jsonl ({"id", "source", "text"}), ready for the VoxCPM2 runner.
"""

import argparse
import json
import random
from collections import Counter

from samleng.bench import load_bench
from samleng.data import LeakCheck, distill_problems
from samleng.paths import OUTPUT_DIR, ROOT
from samleng.runners import read_jsonl, write_jsonl
from samleng.text import verbalize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--openslr", type=int, default=1000, help="OpenSLR sentences to add"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bench = ROOT / "bench" / "khmertts_bench_v1.jsonl"
    leak = LeakCheck([it.text for it in load_bench(bench)])
    split = json.loads(
        (ROOT / "bench" / "khmertts_bench_v1_mixed_split.json").read_text()
    )
    banned = set(split["banned"])

    rows = []
    mixed = (ROOT / "texts" / "distill_mixed_v1.txt").read_text(encoding="utf-8")
    for i, text in enumerate((t.strip() for t in mixed.splitlines() if t.strip()), 1):
        if problems := distill_problems(text, leak, banned):
            raise SystemExit(f"texts/distill_mixed_v1.txt line {i}: {problems}: {text}")
        rows.append(
            {"id": f"mix-{i:03d}", "source": "mixed_v1", "text": verbalize(text)}
        )

    manifest = read_jsonl(OUTPUT_DIR / "data" / "openslr42" / "manifest.jsonl")
    slr = [
        r
        for r in manifest
        if "reject" not in r and not distill_problems(r["raw_text"], leak, banned)
    ]
    random.Random(args.seed).shuffle(slr)
    rows += [
        {"id": f"slr-{r['id']}", "source": "openslr42", "text": r["text"]}
        for r in slr[: args.openslr]
    ]

    out = OUTPUT_DIR / "distill" / "v1" / "tts_in.jsonl"
    write_jsonl(out, rows)
    print(dict(Counter(r["source"] for r in rows)), "->", out)


if __name__ == "__main__":
    main()
