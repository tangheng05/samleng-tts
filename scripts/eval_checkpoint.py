"""Score a fine-tuning checkpoint on bench v1 (Khmer categories) against base sengtha.

uv run python scripts/eval_checkpoint.py outputs/train/distill_v1_A/<run folder>
uv run python scripts/eval_checkpoint.py <run folder> --step 2000
uv run python scripts/eval_checkpoint.py <run B folder> --translit            # English in Khmer script
uv run python scripts/eval_checkpoint.py <run folder> --tag seed1 --synth="--noise-scale-dp 0 --seed 1"
uv run python scripts/eval_checkpoint.py <run folder> --voice teacher_clear/0 --pace 0.867   # new voice

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
from samleng.data import khmer_script_leaks, mixed_group, strict_unseen
from samleng.eval.cer import mixed_cer
from samleng.eval.guard import guard_table, signal_stats
from samleng.eval.report import compare_to_base
from samleng.paths import OUTPUT_DIR, ROOT
from samleng.runners import read_jsonl, run, same_settings, write_jsonl
from samleng.text import EN_KM, khmer_script, verbalize

BENCH = ROOT / "bench" / "khmertts_bench_v1.jsonl"
SPLIT = ROOT / "bench" / "khmertts_bench_v1_mixed_split.json"
EVAL = OUTPUT_DIR / "train" / "eval"
# Owner-rated and scored better than the model's shipped 0.6 (2026-11-12); base keeps 0.6.
STEADY = "--noise-scale-dp 0"


def checkpoints(run_dir: Path) -> dict[int, Path]:
    return {
        int(m[1]): p
        for p in run_dir.glob("checkpoint_*.pth")
        if (m := re.search(r"_(\d+)\.pth$", p.name))
    }


def score(
    out: Path, items, ckpt: Path | None, config: Path | None, spoken=verbalize, synth=()
) -> pd.DataFrame:
    """Synthesize + transcribe once per folder; later calls read the cached results.csv."""
    if ckpt:
        same_settings(out, {"synth": [str(x) for x in synth], "input": spoken.__name__})
    if (out / "results.csv").exists():
        return pd.read_csv(out / "results.csv")
    manifest = out / "tts_in.jsonl"
    write_jsonl(manifest, [{"id": it.id, "text": spoken(it.text)} for it in items])
    extra = ["--checkpoint", ckpt, "--config", config, *synth] if ckpt else []
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


def strict_ids(manifest: Path, items) -> set[str]:
    """Never-seen mixed sentences whose English words are also absent in Khmer script."""
    if not manifest.exists():
        return set()
    split = json.loads(SPLIT.read_text())
    spellings = {w: EN_KM.get(w, "") for w in split["banned"]}
    leaked = khmer_script_leaks(spellings, (r["text"] for r in read_jsonl(manifest)))
    return strict_unseen(items, split, leaked)


def pace_hint(gt: pd.DataFrame, step: int, synth: list[str], pace: float) -> str:
    """Which --length-scale would bring this step to the wanted pace."""
    row = gt[gt["system"] == f"step{step}"]
    if row.empty:
        return ""
    ratio = float(row["dur_ratio_median"].iloc[0])
    used = (
        float(synth[synth.index("--length-scale") + 1])
        if "--length-scale" in synth
        else 1.15
    )
    return (
        f"\nPace: step {step} speaks at {ratio:g}× base with --length-scale {used:g}; "
        f"for {pace:g}× use --length-scale {used * pace / ratio:.2f}.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, help="default: newest checkpoint")
    ap.add_argument(
        "--notify", action="store_true", help="send a phone progress update"
    )
    ap.add_argument(
        "--translit",
        action="store_true",
        help="feed English in Khmer script, to the checkpoint and to base",
    )
    ap.add_argument("--tag", help="score into <run>_<tag> (use with --synth)")
    ap.add_argument(
        "--synth",
        default=STEADY,
        help="synthesize.py options for the checkpoint (replaces the default)",
    )
    ap.add_argument(
        "--voice",
        default="base",
        help="scored folder under outputs/train/eval with the target voice's clips",
    )
    ap.add_argument(
        "--train-manifest",
        type=Path,
        help="the run's training manifest (default: manifest.jsonl next to the run folder); "
        "adds a mixed_unseen_strict row",
    )
    ap.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="wanted speech duration relative to base (the owner-approved pace is 0.867)",
    )
    args = ap.parse_args()

    ckpts = checkpoints(args.run_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoint_*.pth in {args.run_dir}")
    step = args.step or max(ckpts)
    items = [it for it in load_bench(BENCH) if it.category != "english"]
    parts = [args.run_dir.parent.name, "translit" if args.translit else None, args.tag]
    name = "_".join(p for p in parts if p)
    base = EVAL / ("base_translit" if args.translit else "base")
    spoken = khmer_script if args.translit else verbalize
    config = args.run_dir / "config.json"

    frames = [score(base, items, None, None, spoken).assign(system="base")]
    score(
        EVAL / name / str(step), items, ckpts[step], config, spoken, args.synth.split()
    )
    for d in sorted((EVAL / name).iterdir()):
        if d.name.isdigit() and (d / "results.csv").exists():
            frames.append(pd.read_csv(d / "results.csv").assign(system=f"step{d.name}"))

    # Same as plain CER without English words; on mixed it accepts either script.
    table = compare_to_base(pd.concat(frames, ignore_index=True), "base", mixed_cer)
    strict = strict_ids(
        args.train_manifest or args.run_dir.parent / "manifest.jsonl", items
    )
    if strict:
        both = pd.concat(frames, ignore_index=True)
        sub = compare_to_base(both[both["id"].isin(strict)], "base", mixed_cer)
        sub = sub[sub["category"] == "ALL"].assign(category="mixed_unseen_strict")
        table = pd.concat([table, sub], ignore_index=True)
    table["order"] = table["system"].str.removeprefix("step").astype(int)
    table = table.sort_values(["category", "order"]).drop(columns="order")
    rows = table.to_dict("records")
    table["p"] = table["p"].map("{:.2g}".format)
    text = (
        f"# {name}: checkpoints vs base sengtha (bench v1, verbalized"
        + (", English fed in Khmer script" if args.translit else "")
        + ")\n\n"
        "Median per-sentence CER with English words in Khmer script on both sides "
        "(mixed CER, same as CER without English); better/worse/same count "
        "sentences vs base; p = paired Wilcoxon.\n\n"
        + table.to_markdown(index=False)
        + "\n"
    )
    if strict:
        text += (
            f"\nmixed_unseen_strict: the {len(strict)} never-seen sentences whose English words "
            "also do not occur in the training text in Khmer script.\n"
        )
    # Lazy import: WavLM is only needed once per new step (results are cached).
    from samleng.eval.spk import SpeakerEncoder

    enc = SpeakerEncoder()
    guards = {"base": guard(base, enc)}
    if args.voice != "base":
        guards[args.voice] = guard(EVAL / args.voice, enc)
    for d in sorted(
        (EVAL / name).iterdir(), key=lambda d: int(d.name) if d.name.isdigit() else -1
    ):
        if d.name.isdigit() and (g := guard(d, enc)) is not None:
            guards[f"step{d.name}"] = g
    ready = all(g is not None for g in guards.values())
    gt = guard_table(guards, args.voice, args.pace) if ready else None
    if gt is not None:
        text += (
            f"\n## Guardrails (voice similarity to {args.voice}: flag if more than 0.03 below "
            f"its own clips; speech duration vs base: flag outside {args.pace:g}× ±10%; clipping)\n\n"
            + gt.to_markdown(index=False)
            + "\n"
        )
        text += pace_hint(gt, step, args.synth.split(), args.pace)
    (EVAL / name / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    if args.notify:
        from samleng.notify import checkpoint_message, send

        run = name.removeprefix("distill_").replace("_", "-")
        g = {} if gt is None else {r["system"]: r for r in gt.to_dict("records")}
        title, body = checkpoint_message(run, step, rows, g)
        send(title, body, tags="chart_with_upwards_trend")


if __name__ == "__main__":
    main()
