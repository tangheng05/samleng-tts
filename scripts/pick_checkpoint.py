"""Compare candidate checkpoints of one run on several seeds, to pick the final model.

uv run python scripts/pick_checkpoint.py <run folder> --steps 18000 20000 \\
    --synth="--noise-scale-dp 0 --length-scale 1.3" [--seeds 3] [--translit] [--voice teacher_clear/0 --pace 0.867]

One seed moves the all-Khmer median by about a point, so a single score cannot rank two
checkpoints. For each step this scores seeds 0..N-1 with eval_checkpoint.py (seed 0 is the
watcher's own folder; seed k goes to <run>[_translit]_seedK; cached results are reused),
averages the per-sentence error over seeds, and tests each pair of steps on the same
sentences. Writes outputs/train/eval/<run>[_translit]/pick.md.
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

from samleng.eval.cer import mixed_cer
from samleng.eval.report import category_medians, paired, seed_average
from samleng.paths import OUTPUT_DIR, ROOT

EVAL = OUTPUT_DIR / "train" / "eval"
SCORER = ROOT / "scripts" / "eval_checkpoint.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--synth",
        required=True,
        help="synthesize.py options without --seed, exactly as the watcher used them",
    )
    ap.add_argument("--translit", action="store_true")
    ap.add_argument("--voice", default="base")
    ap.add_argument("--pace", type=float, default=1.0)
    args = ap.parse_args()

    run = args.run_dir.parent.name + ("_translit" if args.translit else "")
    avg, per_seed = {}, {}
    for step in args.steps:
        frames = []
        for seed in range(args.seeds):
            name = run if seed == 0 else f"{run}_seed{seed}"
            synth = args.synth if seed == 0 else f"{args.synth} --seed {seed}"
            cmd = [sys.executable, str(SCORER), str(args.run_dir), "--step", str(step)]
            cmd += [f"--synth={synth}", "--voice", args.voice, "--pace", str(args.pace)]
            cmd += ["--translit"] * args.translit
            cmd += ["--tag", f"seed{seed}"] if seed else []
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            frames.append(pd.read_csv(EVAL / name / str(step) / "results.csv"))
        avg[step] = seed_average(frames, mixed_cer)
        per_seed[step] = [
            category_medians(seed_average([f], mixed_cer))["ALL"] for f in frames
        ]

    medians = pd.DataFrame({f"step{s}": category_medians(a) for s, a in avg.items()}).T
    medians.insert(0, "ALL per seed", [" / ".join(map(str, per_seed[s])) for s in avg])
    pairs = [
        {"a": f"step{a}", "b": f"step{b}", "category": cat or "ALL"}
        | paired(avg[a], avg[b], cat)
        for a, b in itertools.combinations(args.steps, 2)
        for cat in (None, "mixed_unseen", "pure", "numbers", "names")
    ]
    text = (
        f"# {run}: candidate checkpoints over {args.seeds} seeds\n\n"
        f"Settings: `{args.synth}`. Median per-sentence error (%) of the {args.seeds}-seed "
        "average; lower is better.\n\n" + medians.to_markdown() + "\n"
    )
    if pairs:
        table = pd.DataFrame(pairs).assign(p=lambda d: d["p"].map("{:.2g}".format))
        text += (
            "\n## Pairs (better/worse = sentences where a has the lower/higher error; "
            "p = paired Wilcoxon)\n\n" + table.to_markdown(index=False) + "\n"
            "\nPick the step that wins with p < 0.05; if none does, they are equal: take the "
            "later step and let a native listener confirm.\n"
        )
    (EVAL / run / "pick.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
