"""Write the project page's live status file from the Mac mini's queue, training log and scores.

uv run python scripts/site_status.py --run distill_v2_clear --out outputs/site/status.json
"""

import argparse
import json
import time
from pathlib import Path

from samleng.paths import OUTPUT_DIR
from samleng.site_status import base_scores, build_status, parse_train_log, step_scores

EVAL = OUTPUT_DIR / "train" / "eval"
# A log untouched this long means training is not running, whatever the queue says.
STALE_SECONDS = 15 * 60


def queue_state(jobs: Path, log: Path) -> str:
    paused = jobs / "PAUSED"
    if paused.exists() and not paused.read_text().startswith("HOLD"):
        return "paused"
    fresh = log.exists() and time.time() - log.stat().st_mtime < STALE_SECONDS
    return "training" if any((jobs / "running").glob("*.sh")) and fresh else "idle"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under outputs/train/")
    ap.add_argument(
        "--log", type=Path, help="training log (default: the queue's 060 job log)"
    )
    ap.add_argument("--jobs", type=Path, default=Path.home() / "samleng-jobs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    log = args.log or args.jobs / "logs" / "060-train-clear.log"
    pos = parse_train_log(log.read_text(errors="replace")) if log.exists() else None
    status = build_status(
        pos,
        scores=step_scores(EVAL / args.run),
        base=base_scores(EVAL / "base" / "results.csv"),
        state=queue_state(args.jobs, log),
        run=args.run,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(args.out)


if __name__ == "__main__":
    main()
