"""Check automatic metrics against human MOS on rated mixed sentences.

uv run python scripts/validate_metrics.py mixed_v2 mms=pilot_v1:mms voxcpm2=pilot_v1:voxcpm2

Each SYSTEM=RUN:LABEL points a rated system at outputs/baseline/RUN/results.csv rows of LABEL.
Writes outputs/mos/<study>/metrics.md: Spearman (95% bootstrap CI) between per-sentence MOS
and each metric, overall and within each system. Negative is good for error rates.
"""

import argparse

import pandas as pd

from samleng.eval.cer import english_hits, khmer_cer, mixed_cer
from samleng.eval.report import spearman_ci
from samleng.mos import load_ratings
from samleng.paths import OUTPUT_DIR

METRICS = {
    "cer": lambda r, h: khmer_cer(r, h),
    "mixed_cer": lambda r, h: mixed_cer(r, h),
    "english_miss": lambda r, h: (
        1 - english_hits(r, h)[0] / max(english_hits(r, h)[1], 1)
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("systems", nargs="+", metavar="SYSTEM=RUN:LABEL")
    args = ap.parse_args()

    study = OUTPUT_DIR / "mos" / args.study
    ratings = pd.DataFrame(load_ratings(study / "ratings.jsonl"))
    mos = ratings.groupby(["system", "item_id"])["score"].mean().rename("mos")

    frames = []
    for spec in args.systems:
        system, _, src = spec.partition("=")
        run, _, label = src.partition(":")
        df = pd.read_csv(OUTPUT_DIR / "baseline" / run / "results.csv")
        df = df[df["system"] == label].assign(system=system)
        frames.append(df[["system", "id", "ref", "hyp"]])
    df = pd.concat(frames).rename(columns={"id": "item_id"})
    df = df.join(mos, on=["system", "item_id"], how="inner")
    df["hyp"] = df["hyp"].fillna("")
    for name, fn in METRICS.items():
        df[name] = [fn(r, h) for r, h in zip(df["ref"], df["hyp"], strict=True)]

    rows = []
    groups = [("all", df), *df.groupby("system")]
    for group, g in groups:
        for name in METRICS:
            rho, lo, hi = spearman_ci(g[name], g["mos"])
            rows.append(
                {
                    "group": group,
                    "metric": name,
                    "n": len(g),
                    "rho": rho,
                    "ci95": f"[{lo}, {hi}]",
                }
            )
    means = df.groupby("system")[["mos", *METRICS]].mean().round(3)
    text = (
        f"# Metrics vs MOS: {args.study}\n\n"
        f"Systems: {', '.join(args.systems)}\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n\n## Means per system\n\n"
        + means.to_markdown()
        + "\n"
    )
    (study / "metrics.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
