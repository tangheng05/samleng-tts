"""Build clean training clips from a public corpus (only OpenSLR 42 so far).

uv run python scripts/prepare_data.py prepare   # extract, clean audio, drop bench overlap
uv run python scripts/prepare_data.py asr       # ASR round trip: CER per clip + stats.md
uv run python scripts/prepare_data.py stats     # rescore stored transcripts (after a CER change)
"""

import argparse
import zipfile

import pandas as pd
import soundfile as sf

from samleng.bench import load_bench
from samleng.data import (
    TARGET_SR,
    LeakCheck,
    parse_line_index,
    prepare_audio,
    unsegment,
)
from samleng.eval.cer import khmer_cer
from samleng.paths import DATA_DIR, OUTPUT_DIR, ROOT
from samleng.runners import read_jsonl, run, write_jsonl
from samleng.text import verbalize

SOURCE = "openslr42"
RAW = DATA_DIR / SOURCE / "raw" / "km_kh_male.zip"
EXTRACTED = DATA_DIR / SOURCE / "extracted"
CLIPS = DATA_DIR / SOURCE / "clips"
OUT = OUTPUT_DIR / "data" / SOURCE
MIN_S, MAX_S = 0.5, 20.0


def prepare() -> None:
    if not (EXTRACTED / "km_kh_male").exists():
        with zipfile.ZipFile(RAW) as z:
            z.extractall(EXTRACTED)
    corpus = EXTRACTED / "km_kh_male"
    index = (corpus / "line_index.tsv").read_text(encoding="utf-8").splitlines()
    leaks = LeakCheck(
        [it.text for p in (ROOT / "bench").rglob("*.jsonl") for it in load_bench(p)]
    )
    manifest = OUT / "manifest.jsonl"
    # Keep transcripts from an earlier ASR run: that step takes an hour.
    asr_done = (
        {r["id"]: r["asr"] for r in read_jsonl(manifest) if "asr" in r}
        if manifest.exists()
        else {}
    )
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows = []
    for utt, speaker, text in parse_line_index(index):
        spoken = verbalize(unsegment(text))
        row = {"id": utt, "speaker": speaker, "raw_text": text, "text": spoken}
        if leaks(text):
            row["reject"] = "bench_overlap"
        else:
            y, sr = sf.read(corpus / "wavs" / f"{utt}.wav")
            y = prepare_audio(y, sr)
            row["seconds"] = round(len(y) / TARGET_SR, 3)
            if not MIN_S <= row["seconds"] <= MAX_S:
                row["reject"] = "duration"
            else:
                row["audio"] = str(CLIPS / f"{utt}.wav")
                sf.write(row["audio"], y, TARGET_SR, subtype="PCM_16")
                if utt in asr_done:
                    row["asr"] = asr_done[utt]
        rows.append(row)
    write_jsonl(manifest, rows)
    print(pd.Series([r.get("reject", "kept") for r in rows]).value_counts().to_string())
    if asr_done:
        stats()


def asr() -> None:
    rows = read_jsonl(OUT / "manifest.jsonl")
    kept = [r for r in rows if "reject" not in r]
    asr_in, asr_out = OUT / "asr_in.jsonl", OUT / "asr_out.jsonl"
    write_jsonl(asr_in, [{"id": r["id"], "audio": r["audio"]} for r in kept])
    run("asr_qwen", "transcribe.py", asr_in, asr_out)
    hyp = {r["id"]: r["text"] for r in read_jsonl(asr_out)}
    for r in kept:
        r["asr"] = hyp[r["id"]]
    write_jsonl(OUT / "manifest.jsonl", rows)
    stats()


def stats() -> None:
    rows = read_jsonl(OUT / "manifest.jsonl")
    for r in rows:
        if "asr" in r:
            r["cer"] = round(khmer_cer(r["raw_text"], r["asr"]), 4)
    write_jsonl(OUT / "manifest.jsonl", rows)
    write_stats(pd.DataFrame(rows))


def write_stats(df: pd.DataFrame) -> None:
    kept = df[df["reject"].isna()] if "reject" in df else df
    cer = kept["cer"]
    per_speaker = (
        kept.groupby("speaker")["seconds"]
        .agg(clips="count", hours=lambda s: s.sum() / 3600)
        .round(3)
    )
    lines = [
        f"# {SOURCE}",
        f"- clips: {len(df)}, kept: {len(kept)}, hours kept: {kept['seconds'].sum() / 3600:.2f}",
        "## Rejected",
        df.get("reject", pd.Series(dtype=str)).value_counts().to_markdown(),
        "## ASR round-trip CER (pick the threshold from this)",
        cer.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(3).to_markdown(),
        *(f"- CER > {t}: {(cer > t).sum()} clips" for t in (0.05, 0.1, 0.2, 0.3, 0.5)),
        "## Per speaker",
        per_speaker.to_markdown(),
    ]
    (OUT / "stats.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "stats.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    steps = {"prepare": prepare, "asr": asr, "stats": stats}
    ap.add_argument("step", choices=steps)
    steps[ap.parse_args().step]()


if __name__ == "__main__":
    main()
