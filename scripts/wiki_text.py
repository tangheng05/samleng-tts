"""Pick clean Khmer sentences from Khmer Wikipedia as distillation text (CC-BY-SA 3.0).

uv run python scripts/wiki_text.py --n 3000 --out texts/distill_wiki_v3.txt

Keeps sentences that are almost all Khmer, a natural length, free of brackets/links/tables,
fully readable after number verbalization, and with no benchmark overlap. Text is written
as it appears (digits kept); distill_text.py verbalizes it like every other source.
"""

import argparse
import random
import re
import unicodedata

import pandas as pd
from huggingface_hub import hf_hub_download

from samleng.bench import load_bench
from samleng.data import LeakCheck
from samleng.eval.cer import normalize_khmer
from samleng.paths import ROOT
from samleng.text import verbalize

REPO, FILE = "wikimedia/wikipedia", "20231101.km/train-00000-of-00001.parquet"
REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SPLIT = re.compile(r"(?<=[។៕?!])\s*|\n+")
BAD = re.compile(r"[A-Za-z\[\]{}()<>|=_*#@/\\~^`\"“”«»:;]|http|www")
KHMER = re.compile(r"[ក-៿]")


def clean(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub("[\u200b\u200c\u200d\ufeff]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def keep(s: str, min_len: int, max_len: int) -> bool:
    if BAD.search(s) or not s.endswith(("។", "៕", "?", "!")):
        return False
    core = normalize_khmer(s, spell_numbers=False)
    if not min_len <= len(core) <= max_len:
        return False
    if len(KHMER.findall(core)) / max(len(core), 1) < 0.95:
        return False
    # Every digit must turn into words; otherwise the teacher would read raw digits.
    return not re.search(r"[0-9០-៩]", verbalize(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "texts" / "distill_wiki_v3.txt"))
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset", revision=REV))
    leak = LeakCheck(
        [it.text for it in load_bench(ROOT / "bench" / "khmertts_bench_v1.jsonl")]
    )
    seen, picked = set(), []
    for text in df["text"]:
        for s in SPLIT.split(text):
            s = clean(s)
            key = normalize_khmer(s, spell_numbers=False)
            if key in seen or not keep(s, args.min_len, args.max_len) or leak(s):
                continue
            seen.add(key)
            picked.append(s)
    random.Random(args.seed).shuffle(picked)
    out = picked[: args.n]
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(s + "\n" for s in out)
    print(
        f"{len(df)} articles, {len(picked)} clean sentences, wrote {len(out)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
