"""Build the sentences VoxCPM2 reads in sengtha's voice (Phase 2 distillation).

uv run python scripts/distill_text.py                # v1: 500 mixed + 1,000 OpenSLR sentences
uv run python scripts/distill_text.py --version v2   # v2: new mixed + numbers + unused OpenSLR
uv run python scripts/distill_text.py --version v3   # v3: Khmer Wikipedia sentences
Writes outputs/distill/<version>/tts_in.jsonl ({"id", "source", "text"}), ready for the VoxCPM2 runner.
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

# version -> (source, file under texts/, id prefix); earlier versions' OpenSLR rows are skipped.
TEXTS = {
    "v1": [("mixed_v1", "distill_mixed_v1.txt", "mix")],
    "v2": [
        ("mixed_v2", "distill_mixed_v2.txt", "mix2"),
        ("numbers_v2", "distill_numbers_v2.txt", "num2"),
    ],
    # scripts/wiki_text.py; OpenSLR is used up after v2.
    "v3": [
        ("wiki_v3", "distill_wiki_v3.txt", "wiki3"),
        ("mixed_acr_v3", "distill_acronyms_v3.txt", "acr3"),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--openslr",
        type=int,
        help="OpenSLR sentences to add (v1: 1000, later: all unused)",
    )
    ap.add_argument("--version", default="v1", choices=TEXTS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bench = ROOT / "bench" / "khmertts_bench_v1.jsonl"
    leak = LeakCheck([it.text for it in load_bench(bench)])
    split = json.loads(
        (ROOT / "bench" / "khmertts_bench_v1_mixed_split.json").read_text()
    )
    banned = set(split["banned"])

    rows = []
    for source, name, prefix in TEXTS[args.version]:
        lines = (ROOT / "texts" / name).read_text(encoding="utf-8").splitlines()
        for i, text in enumerate((t.strip() for t in lines if t.strip()), 1):
            if problems := distill_problems(text, leak, banned):
                raise SystemExit(f"texts/{name} line {i}: {problems}: {text}")
            rows.append(
                {"id": f"{prefix}-{i:03d}", "source": source, "text": verbalize(text)}
            )

    used = set()
    for version in list(TEXTS)[: list(TEXTS).index(args.version)]:
        used |= {
            r["id"]
            for r in read_jsonl(OUTPUT_DIR / "distill" / version / "tts_in.jsonl")
        }

    manifest = read_jsonl(OUTPUT_DIR / "data" / "openslr42" / "manifest.jsonl")
    slr = [
        r
        for r in manifest
        if "reject" not in r and not distill_problems(r["raw_text"], leak, banned)
    ]
    random.Random(args.seed).shuffle(slr)
    slr = [r for r in slr if f"slr-{r['id']}" not in used]
    n = args.openslr or (1000 if args.version == "v1" else len(slr))
    rows += [
        {"id": f"slr-{r['id']}", "source": "openslr42", "text": r["text"]}
        for r in slr[:n]
    ]

    out = OUTPUT_DIR / "distill" / args.version / "tts_in.jsonl"
    if out.exists():
        # Teacher audio is generated from this file; the OpenSLR pool may have changed since.
        raise SystemExit(
            f"{out} exists: it is frozen once generated, delete it by hand"
        )
    write_jsonl(out, rows)
    print(dict(Counter(r["source"] for r in rows)), "->", out)


if __name__ == "__main__":
    main()
