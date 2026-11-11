"""Phone alerts through ntfy.sh. The topic name is private and lives outside the repo."""

import os
import subprocess
from pathlib import Path

TOPIC_FILE = Path.home() / "samleng-ops" / "ntfy_topic"
# Median CER differences smaller than this (points) count as "about the same".
SAME_WITHIN = 0.3


def send(title: str, body: str, priority: str = "default", tags: str = "") -> bool:
    """Best effort: never raises, so a dead network can't break a job."""
    topic = os.environ.get("NTFY_TOPIC") or (
        TOPIC_FILE.read_text().strip() if TOPIC_FILE.exists() else ""
    )
    if not topic:
        return False
    # curl, not urllib: curl falls back to IPv4 when IPv6 is broken (urllib just times out).
    cmd = ["curl", "-s", "-m", "20", "-o", "/dev/null", "-w", "%{http_code}"]
    cmd += ["-H", f"Title: {title}", "-H", f"Priority: {priority}"]
    if tags:
        cmd += ["-H", f"Tags: {tags}"]
    cmd += ["--data-binary", "@-", f"https://ntfy.sh/{topic}"]
    try:
        out = subprocess.run(
            cmd, check=False, input=body.encode(), capture_output=True, timeout=30
        )
        return out.stdout.decode().strip() == "200"
    except (OSError, subprocess.SubprocessError):
        return False


def checkpoint_message(
    run: str, step: int, table: list[dict], guards: dict[str, dict]
) -> tuple[str, str]:
    """A plain-language progress update for one scored checkpoint (lower CER is better)."""

    def cer(system: str, cat: str) -> dict | None:
        return next(
            (r for r in table if r["system"] == system and r["category"] == cat), None
        )

    me = f"step{step}"
    now = cer(me, "ALL")
    earlier = [
        (r["median_cer_pct"], int(r["system"][4:]))
        for r in table
        if r["category"] == "ALL" and r["system"] != me and int(r["system"][4:]) < step
    ]
    lines = []
    if earlier:
        best, best_step = min(earlier)
        diff = now["median_cer_pct"] - best
        verdict = (
            "better than the best so far"
            if diff < -SAME_WITHIN
            else "worse than the best so far"
            if diff > SAME_WITHIN
            else "about the same as the best so far"
        )
        lines.append(
            f"Errors: {now['median_cer_pct']}% (start {now['median_cer_base_pct']}%, "
            f"best before {best}% at step {best_step:,})"
        )
    else:
        verdict = "first score"
        lines.append(
            f"Errors: {now['median_cer_pct']}% (start {now['median_cer_base_pct']}%)"
        )
    unseen = cer(me, "mixed_unseen")
    if unseen:
        lines.append(
            f"Never-seen English: {unseen['median_cer_pct']}% "
            f"(start {unseen['median_cer_base_pct']}%)"
        )
    extras = [
        f"{label} {r['median_cer_pct']}%"
        for cat, label in (("numbers", "Numbers"), ("pure", "Pure Khmer"))
        if (r := cer(me, cat))
    ]
    if extras:
        lines.append(" · ".join(extras))
    if g := guards.get(me):
        status = "ok" if g["flags"] == "ok" else f"⚠️ {g['flags']}"
        lines.append(
            f"Voice {g['spk_median']} · speed {g['dur_ratio_median']}× · {status}"
        )
    return f"{run} step {step:,}: {verdict}", "\n".join(lines)
