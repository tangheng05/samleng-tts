"""Live training status for the project page: where the run is, when it ends, how it scores."""

import re
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from samleng.eval.cer import mixed_cer

ANSI = re.compile(r"\x1b\[[0-9;]*m")
EPOCH = re.compile(r"EPOCH: (\d+)/(\d+)")
STEP = re.compile(r"TIME: (\S+ \S+) -- STEP: (\d+)/(\d+) -- GLOBAL_STEP: (\d+)")
STEP_TIME = re.compile(r"\| > step_time: ([\d.]+)")
CATEGORIES = ("pure", "numbers", "names", "long", "mixed_unseen")


def parse_train_log(text: str) -> dict | None:
    """Latest epoch and step of a Coqui training log, with the recent median seconds per step."""
    text = ANSI.sub("", text)
    epochs, steps = EPOCH.findall(text), STEP.findall(text)
    if not epochs or not steps:
        return None
    epoch, last = epochs[-1]
    logged_at, in_epoch, per_epoch, step = steps[-1]
    times = [float(t) for t in STEP_TIME.findall(text)[-20:]]
    return {
        "epoch": int(epoch),
        "epochs": int(last) + 1,
        "step_in_epoch": int(in_epoch),
        "steps_per_epoch": int(per_epoch),
        "step": int(step),
        "step_seconds": round(statistics.median(times), 2) if times else None,
        "logged_at": logged_at,
    }


def _medians(df: pd.DataFrame) -> dict:
    df = df[df["category"] != "english"]
    cer = pd.Series(
        [mixed_cer(r, h) for r, h in zip(df["ref"], df["hyp"].fillna(""), strict=True)],
        index=df.index,
    )
    out = {"all": round(100 * float(cer.median()), 1)}
    for c in CATEGORIES:
        if (df["category"] == c).any():
            out[c] = round(100 * float(cer[df["category"] == c].median()), 1)
    return out


def step_scores(eval_dir: Path) -> list[dict]:
    """Median per-sentence CER (%) of every scored step folder under eval_dir, oldest first."""
    rows = []
    for d in eval_dir.iterdir() if eval_dir.exists() else []:
        if d.name.isdigit() and (d / "results.csv").exists():
            rows.append(
                {"step": int(d.name), **_medians(pd.read_csv(d / "results.csv"))}
            )
    return sorted(rows, key=lambda r: r["step"])


def base_scores(results_csv: Path) -> dict:
    return _medians(pd.read_csv(results_csv)) if results_csv.exists() else {}


def build_status(
    pos: dict | None,
    *,
    scores: list[dict],
    base: dict,
    state: str,
    run: str | None,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    steps_left = finish_at = progress = None
    if pos:
        per = pos["steps_per_epoch"]
        steps_left = (per - pos["step_in_epoch"]) + (
            pos["epochs"] - pos["epoch"] - 1
        ) * per
        progress = round((pos["epoch"] + pos["step_in_epoch"] / per) / pos["epochs"], 4)
        if state == "training" and pos["step_seconds"]:
            finish_at = (
                now + timedelta(seconds=steps_left * pos["step_seconds"])
            ).isoformat()
    best = min(scores, key=lambda s: s["all"]) if scores else None
    return {
        "updated_at": now.isoformat(),
        "state": state,
        "run": run,
        **(pos or {}),
        "progress": progress,
        "steps_left": steps_left,
        "finish_at": finish_at,
        "scores": scores,
        "base": base,
        "best": {"step": best["step"], "all": best["all"]} if best else None,
    }
