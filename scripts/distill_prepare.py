"""Turn the teacher's (VoxCPM2) clips into a checked training set for the student (sengtha).

uv run python scripts/distill_prepare.py prepare   # 22.05 kHz clips + voice similarity (CPU)
uv run python scripts/distill_prepare.py asr       # ASR round trip, then filter + stats.md
uv run python scripts/distill_prepare.py stats     # re-filter without re-running ASR
uv run python scripts/distill_prepare.py retry --seed 1   # re-generate rejected clips, keep passing takes
Add --version v2 for later batches (default v1).
"""

import argparse
import glob
import json
from pathlib import Path

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
# Default voice target (sengtha). A batch can name another voice in <batch>/voice.json:
# {"name": ..., "reference": <wav>, "profile": <glob of clips in that voice>}, paths under outputs/.
SENGTHA = {
    "name": "sengtha",
    "reference": "distill/v1/reference.wav",
    "profile": "baseline/khmertts_bench_v1/sengtha/pure-*.wav",
}
# The runner was started from its own folder, so its audio paths are relative to it.
RUNNER_DIR = ROOT / "runners" / "tts_voxcpm"


def voice() -> dict:
    f = RUN / "voice.json"
    return json.loads(f.read_text()) if f.exists() else SENGTHA


def voice_profile(enc: SpeakerEncoder) -> np.ndarray:
    v = voice()
    clips = sorted(glob.glob(str(OUTPUT_DIR / v["profile"])))[:10]
    embs = [enc.embed(p) for p in [*clips, OUTPUT_DIR / v["reference"]]]
    unit = np.mean([e / np.linalg.norm(e) for e in embs], axis=0)
    return unit / np.linalg.norm(unit)


def clip(teacher_audio: str, out: Path, enc: SpeakerEncoder, profile) -> dict:
    """Resample/trim/level one teacher clip and measure its voice similarity."""
    y, sr = sf.read((RUNNER_DIR / teacher_audio).resolve())
    y = prepare_audio(y, sr)
    sf.write(out, y, TARGET_SR, subtype="PCM_16")
    e = enc.embed(out)
    return {
        "audio": str(out),
        "seconds": round(len(y) / TARGET_SR, 3),
        "spk": round(float(e @ profile / np.linalg.norm(e)), 4),
    }


def transcribe(rows: list[dict], name: str) -> dict[str, str]:
    asr_in, asr_out = RUN / f"{name}_in.jsonl", RUN / f"{name}_out.jsonl"
    write_jsonl(asr_in, [{"id": r["id"], "audio": r["audio"]} for r in rows])
    run("asr_qwen", "transcribe.py", asr_in, asr_out)
    return {r["id"]: r["text"] for r in read_jsonl(asr_out)}


def score(r: dict) -> None:
    r["cer"] = round(khmer_cer(r["text"], r["asr"]), 4)
    r["cps"] = round(len(normalize_khmer(r["text"])) / r["seconds"], 2)


def rate_window(rows: list[dict]) -> tuple[float, float]:
    cps = [r["cps"] for r in rows]
    return float(np.percentile(cps, 1)), float(np.percentile(cps, 99))


def prepare() -> None:
    texts = {r["id"]: r for r in read_jsonl(RUN / "tts_in.jsonl")}
    enc = SpeakerEncoder()
    profile = voice_profile(enc)
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in read_jsonl(TEACHER / "results.jsonl"):
        src = texts[r["id"]]
        rows.append(
            {"id": r["id"], "source": src["source"], "text": src["text"]}
            | clip(r["audio"], CLIPS / f"{r['id']}.wav", enc, profile)
        )
    write_jsonl(MANIFEST, rows)
    print(
        f"{len(rows)} clips prepared; voice similarity p5={np.percentile([r['spk'] for r in rows], 5):.3f}"
    )


def asr() -> None:
    rows = read_jsonl(MANIFEST)
    hyp = transcribe(rows, "asr")
    for r in rows:
        r["asr"] = hyp[r["id"]]
    write_jsonl(MANIFEST, rows)
    stats()


def stats() -> None:
    rows = read_jsonl(MANIFEST)
    for r in rows:
        score(r)
    rate = rate_window(rows)
    for r in rows:
        r.pop("reject", None)
        if reason := distill_reject(r, rate):
            r["reject"] = reason
    write_jsonl(MANIFEST, rows)
    df = pd.DataFrame(rows)
    kept = df[df["reject"].isna()] if "reject" in df else df
    lines = [
        f"# Distillation {RUN.name} (VoxCPM2 → {voice()['name']}'s voice)",
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


def retry(seed: int) -> None:
    """Re-generate rejected clips with another seed; a take that passes replaces the reject.

    Hard sentences fail most often, so dropping them would bias the data toward easy ones.
    """
    rows = read_jsonl(MANIFEST)
    bad = [r for r in rows if "reject" in r]
    if not bad:
        print("nothing to retry")
        return
    tag = f"seed{seed}"
    texts = {r["id"]: r["text"] for r in read_jsonl(RUN / "tts_in.jsonl")}
    take_in, take_dir = RUN / f"retry_{tag}.jsonl", RUN / f"voxcpm2_{tag}"
    write_jsonl(take_in, [{"id": r["id"], "text": texts[r["id"]]} for r in bad])
    run(
        "tts_voxcpm",
        "synthesize.py",
        take_in,
        take_dir,
        "--reference-wav",
        OUTPUT_DIR / voice()["reference"],
        "--seed",
        str(seed),
        "--hints",
        ROOT / "texts/teacher_hints.json",
        "--resume",
    )
    enc = SpeakerEncoder()
    profile = voice_profile(enc)
    CLIPS.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in rows}
    takes = [
        {
            **by_id[t["id"]],
            **clip(t["audio"], CLIPS / f"{t['id']}.{tag}.wav", enc, profile),
        }
        for t in read_jsonl(take_dir / "results.jsonl")
    ]
    hyp = transcribe(takes, f"retry_{tag}_asr")
    rate = rate_window(rows)
    fixed = 0
    for t in takes:
        t["asr"] = hyp[t["id"]]
        score(t)
        t.pop("reject", None)
        if distill_reject(t, rate) is None:
            by_id[t["id"]].clear()
            by_id[t["id"]].update(t | {"take": tag})
            fixed += 1
    write_jsonl(MANIFEST, rows)
    print(f"{tag}: {fixed} of {len(takes)} rejected clips replaced by a passing take")
    stats()


def main() -> None:
    ap = argparse.ArgumentParser()
    steps = {"prepare": prepare, "asr": asr, "stats": stats, "retry": retry}
    ap.add_argument("step", choices=steps)
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--seed", type=int, default=1, help="retry: teacher seed (first run used 0)"
    )
    args = ap.parse_args()
    global RUN, TEACHER, CLIPS, MANIFEST
    RUN = OUTPUT_DIR / "distill" / args.version
    TEACHER = RUN / "voxcpm2"
    CLIPS = DATA_DIR / "distill" / args.version / "clips"
    MANIFEST = RUN / "manifest.jsonl"
    steps[args.step](args.seed) if args.step == "retry" else steps[args.step]()


if __name__ == "__main__":
    main()
