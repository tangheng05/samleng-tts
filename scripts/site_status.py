"""Write the project page's live status file from the Mac mini's queue, training log and scores.

uv run python scripts/site_status.py --run distill_v2_clear --out outputs/site/status.json
"""

import argparse
import json
import time
from pathlib import Path

from samleng.paths import OUTPUT_DIR, ROOT
from samleng.site_status import (
    base_scores,
    build_status,
    is_run_log,
    job_info,
    parse_train_log,
    step_scores,
)

EVAL = OUTPUT_DIR / "train" / "eval"
# A log untouched this long means training is not running, whatever the queue says.
STALE_SECONDS = 15 * 60


def queue_state(jobs: Path, job: Path | None, info: dict, log: Path | None) -> str:
    paused = jobs / "PAUSED"
    if paused.exists() and not paused.read_text().startswith("HOLD"):
        return "paused"
    if job is None:
        return "idle"
    live = log is not None and log.stem == job.stem
    if live and time.time() - log.stat().st_mtime < STALE_SECONDS:
        return "training"
    return info.get("kind", "testing")


def training_log(logs: Path, run: str) -> tuple[Path, dict] | tuple[None, None]:
    """The newest queue log that trains `run`, with its parsed position."""
    for log in sorted(
        logs.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        text = log.read_text(errors="replace")
        if (
            "GLOBAL_STEP" in text
            and is_run_log(text, run)
            and (pos := parse_train_log(text))
        ):
            return log, pos
    return None, None


def activity(job: Path, info: dict) -> dict:
    out = {"job": job.stem, "en": info.get("en"), "km": info.get("km")}
    if " " in info.get("progress", ""):
        pattern, total = info["progress"].rsplit(" ", 1)
        out |= {"done": len(list(ROOT.glob(pattern))), "total": int(total)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        required=True,
        help="run name under outputs/train/ unless the job names one",
    )
    ap.add_argument("--jobs", type=Path, default=Path.home() / "samleng-jobs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    job = next(iter(sorted((args.jobs / "running").glob("*.sh"))), None)
    info = job_info(job.read_text(errors="replace")) if job else {}
    run = info.get("run", args.run)
    log, pos = training_log(args.jobs / "logs", run)
    status = build_status(
        pos,
        scores=step_scores(EVAL / run),
        base=base_scores(EVAL / "base" / "results.csv"),
        state=queue_state(args.jobs, job, info, log),
        run=run,
    )
    status["activity"] = activity(job, info) if job else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(args.out)


if __name__ == "__main__":
    main()
