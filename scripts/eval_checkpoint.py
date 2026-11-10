"""Score a fine-tuning checkpoint on bench v1 (Khmer categories) against base sengtha.

uv run python scripts/eval_checkpoint.py outputs/train/distill_v1_A/<run folder>
uv run python scripts/eval_checkpoint.py <run folder> --step 2000

Text is verbalized like the training data. Mixed sentences are split into
mixed_unseen / mixed_seen / mixed_partial (bench/khmertts_bench_v1_mixed_split.json).
Clips and transcripts are cached per step; base sengtha is shared across runs.
Writes outputs/train/eval/<run name>/summary.md, with a guardrail table (voice similarity to base,
speaking speed vs base, silence, clipping) so lower CER can't hide a drifting voice.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from samleng.bench import load_bench
from samleng.data import mixed_group
from samleng.eval.cer import mixed_cer
from samleng.eval.guard import guard_flags, signal_stats
from samleng.eval.report import compare_to_base
from samleng.paths import OUTPUT_DIR, ROOT
from samleng.runners import read_jsonl, run, write_jsonl
from samleng.text import verbalize

BENCH = ROOT / "bench" / "khmertts_bench_v1.jsonl"
SPLIT = ROOT / "bench" / "khmertts_bench_v1_mixed_split.json"
EVAL = OUTPUT_DIR / "train" / "eval"


def checkpoints(run_dir: Path) -> dict[int, Path]:
    return {
        int(m[1]): p
        for p in run_dir.glob("checkpoint_*.pth")
        if (m := re.search(r"_(\d+)\.pth$", p.name))
    }


def score(out: Path, items, ckpt: Path | None, config: Path | None) -> pd.DataFrame:
    """Synthesize + transcribe once per folder; later calls read the cached results.csv."""
    if (out / "results.csv").exists():
        return pd.read_csv(out / "results.csv")
    manifest = out / "tts_in.jsonl"
    write_jsonl(manifest, [{"id": it.id, "text": verbalize(it.text)} for it in items])
    extra = ["--checkpoint", ckpt, "--config", config] if ckpt else []
    run("train_vits", "synthesize.py", manifest, out / "wav", *extra)
    clips = read_jsonl(out / "wav" / "results.jsonl")
    asr_in, asr_out = out / "asr_in.jsonl", out / "asr_out.jsonl"
    write_jsonl(asr_in, [{"id": c["id"], "audio": c["audio"]} for c in clips])
    run("asr_qwen", "transcribe.py", asr_in, asr_out)
    hyps = {r["id"]: r["text"] for r in read_jsonl(asr_out)}
    split = json.loads(SPLIT.read_text())
    by_id = {it.id: it for it in items}
    df = pd.DataFrame(
        {
            "id": c["id"],
            "category": mixed_group(c["id"], by_id[c["id"]].category, split),
            "ref": by_id[c["id"]].text,
            "hyp": hyps[c["id"]],
            "seconds": c["seconds"],
            "gen_seconds": c["gen_seconds"],
        }
        for c in clips
    )
    df.to_csv(out / "results.csv", index=False)
    return df


def guard(out: Path, enc) -> pd.DataFrame | None:
    """Per-clip speaker embedding and signal stats, cached in guard.csv (needs the clips)."""
    if (out / "guard.csv").exists():
        return pd.read_csv(out / "guard.csv")
    if not (out / "wav" / "results.jsonl").exists():
        return None
    rows = []
    for c in read_jsonl(out / "wav" / "results.jsonl"):
        y, sr = sf.read(c["audio"])
        e = enc.embed(c["audio"])
        rows.append(
            {"id": c["id"], "seconds": c["seconds"], **signal_stats(y, sr)}
            | {f"e{i}": v for i, v in enumerate(e / np.linalg.norm(e))}
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "guard.csv", index=False)
    return df


def guard_table(guards: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = guards["base"].set_index("id")
    cols = [c for c in base.columns if c.startswith("e") and c[1:].isdigit()]
    profile = base[cols].to_numpy().mean(axis=0)
    profile /= np.linalg.norm(profile)
    rows = []
    for system, g in guards.items():
        g = g.set_index("id")
        spk = g[cols].to_numpy() @ profile
        ratio = g["seconds"] / base.loc[g.index, "seconds"]
        rows.append(
            {
                "system": system,
                "spk_median": round(float(np.median(spk)), 3),
                "dur_ratio_median": round(float(ratio.median()), 3),
                "lead_silence_s": round(float(g["lead_silence_s"].median()), 2),
                "trail_silence_s": round(float(g["trail_silence_s"].median()), 2),
                "clip_frac_p95": round(float(g["clip_frac"].quantile(0.95)), 5),
            }
        )
    out = pd.DataFrame(rows)
    base_row = out[out["system"] == "base"].iloc[0].to_dict()
    out["flags"] = [
        ", ".join(guard_flags(r, base_row)) or "ok" for r in out.to_dict("records")
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, help="default: newest checkpoint")
    args = ap.parse_args()

    ckpts = checkpoints(args.run_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoint_*.pth in {args.run_dir}")
    step = args.step or max(ckpts)
    items = [it for it in load_bench(BENCH) if it.category != "english"]
    name = args.run_dir.parent.name
    config = args.run_dir / "config.json"

    frames = [score(EVAL / "base", items, None, None).assign(system="base")]
    score(EVAL / name / str(step), items, ckpts[step], config)
    for d in sorted((EVAL / name).iterdir()):
        if d.name.isdigit() and (d / "results.csv").exists():
            frames.append(pd.read_csv(d / "results.csv").assign(system=f"step{d.name}"))

    # Same as plain CER without English words; on mixed it accepts either script.
    table = compare_to_base(pd.concat(frames, ignore_index=True), "base", mixed_cer)
    table["order"] = table["system"].str.removeprefix("step").astype(int)
    table = table.sort_values(["category", "order"]).drop(columns="order")
    table["p"] = table["p"].map("{:.2g}".format)
    text = (
        f"# {name}: checkpoints vs base sengtha (bench v1, verbalized)\n\n"
        "Median per-sentence CER with English words in Khmer script on both sides "
        "(mixed CER, same as CER without English); better/worse/same count "
        "sentences vs base; p = paired Wilcoxon.\n\n"
        + table.to_markdown(index=False)
        + "\n"
    )
    # Lazy import: WavLM is only needed once per new step (results are cached).
    from samleng.eval.spk import SpeakerEncoder

    enc = SpeakerEncoder()
    guards = {"base": guard(EVAL / "base", enc)}
    for d in sorted(
        (EVAL / name).iterdir(), key=lambda d: int(d.name) if d.name.isdigit() else -1
    ):
        if d.name.isdigit() and (g := guard(d, enc)) is not None:
            guards[f"step{d.name}"] = g
    if guards["base"] is not None:
        text += (
            "\n## Guardrails (vs base; flags: voice similarity drop > 0.03, speed outside "
            "0.9–1.1×, clipping)\n\n"
            + guard_table(guards).to_markdown(index=False)
            + "\n"
        )
    (EVAL / name / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
