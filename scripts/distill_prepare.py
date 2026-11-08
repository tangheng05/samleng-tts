"""Turn the teacher's (VoxCPM2) clips into a checked training set for the student (sengtha).

uv run python scripts/distill_prepare.py prepare   # 22.05 kHz clips + voice similarity (CPU)
uv run python scripts/distill_prepare.py asr       # ASR round trip, then filter + stats.md
uv run python scripts/distill_prepare.py stats     # re-filter without re-running ASR
"""

import argparse
import glob

import numpy as np
import pandas as pd
import soundfile as sf

from samleng.data import TARGET_SR, distill_reject, prepare_audio
from samleng.eval.cer import khmer_cer, normalize_khmer
from samleng.eval.spk import SpeakerEncoder
from samleng.paths import DATA_DIR, OUTPUT_DIR, ROOT
from samleng.runners import read_jsonl, run, write_jsonl

RUN = OUTPUT_DIR / "distill" / "v1"
TEACHER = RUN / "voxcpm2"
CLIPS = DATA_DIR / "distill" / "v1" / "clips"
MANIFEST = RUN / "manifest.jsonl"
# The runner was started from its own folder, so its audio paths are relative to it.
RUNNER_DIR = ROOT / "runners" / "tts_voxcpm"


def voice_profile(enc: SpeakerEncoder) -> np.ndarray:
    clips = sorted(
        glob.glob(str(OUTPUT_DIR / "baseline/khmertts_bench_v1/sengtha/pure-*.wav"))
    )[:10]
    embs = [enc.embed(p) for p in [*clips, RUN / "reference.wav"]]
    unit = np.mean([e / np.linalg.norm(e) for e in embs], axis=0)
    return unit / np.linalg.norm(unit)


def prepare() -> None:
    texts = {r["id"]: r for r in read_jsonl(RUN / "tts_in.jsonl")}
    enc = SpeakerEncoder()
    profile = voice_profile(enc)
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in read_jsonl(TEACHER / "results.jsonl"):
        y, sr = sf.read((RUNNER_DIR / r["audio"]).resolve())
        y = prepare_audio(y, sr)
        audio = CLIPS / f"{r['id']}.wav"
        sf.write(audio, y, TARGET_SR, subtype="PCM_16")
        e = enc.embed(audio)
        src = texts[r["id"]]
        rows.append(
            {
                "id": r["id"],
                "source": src["source"],
                "text": src["text"],
                "audio": str(audio),
                "seconds": round(len(y) / TARGET_SR, 3),
                "spk": round(float(e @ profile / np.linalg.norm(e)), 4),
            }
        )
    write_jsonl(MANIFEST, rows)
    print(
        f"{len(rows)} clips prepared; voice similarity p5={np.percentile([r['spk'] for r in rows], 5):.3f}"
    )


def asr() -> None:
    rows = read_jsonl(MANIFEST)
    asr_in, asr_out = RUN / "asr_in.jsonl", RUN / "asr_out.jsonl"
    write_jsonl(asr_in, [{"id": r["id"], "audio": r["audio"]} for r in rows])
    run("asr_qwen", "transcribe.py", asr_in, asr_out)
    hyp = {r["id"]: r["text"] for r in read_jsonl(asr_out)}
    for r in rows:
        r["asr"] = hyp[r["id"]]
    write_jsonl(MANIFEST, rows)
    stats()


def stats() -> None:
    rows = read_jsonl(MANIFEST)
    for r in rows:
        r["cer"] = round(khmer_cer(r["text"], r["asr"]), 4)
        r["cps"] = round(len(normalize_khmer(r["text"])) / r["seconds"], 2)
    cps = [r["cps"] for r in rows]
    rate = (float(np.percentile(cps, 1)), float(np.percentile(cps, 99)))
    for r in rows:
        r.pop("reject", None)
        if reason := distill_reject(r, rate):
            r["reject"] = reason
    write_jsonl(MANIFEST, rows)
    df = pd.DataFrame(rows)
    kept = df[df["reject"].isna()] if "reject" in df else df
    lines = [
        "# Distillation v1 (VoxCPM2 → sengtha's voice)",
        f"- clips: {len(df)}, kept: {len(kept)}, hours kept: {kept['seconds'].sum() / 3600:.2f}",
        f"- speaking-rate window (chars/s, p1–p99): {rate[0]:.1f}–{rate[1]:.1f}",
        "## Rejected",
        df.get("reject", pd.Series(dtype=str)).value_counts().to_markdown(),
        "## Per source (median CER, median voice similarity)",
        df.groupby("source")
        .agg(clips=("id", "count"), cer=("cer", "median"), spk=("spk", "median"))
        .round(3)
        .to_markdown(),
    ]
    (RUN / "stats.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print((RUN / "stats.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    steps = {"prepare": prepare, "asr": asr, "stats": stats}
    ap.add_argument("step", choices=steps)
    steps[ap.parse_args().step]()


if __name__ == "__main__":
    main()
