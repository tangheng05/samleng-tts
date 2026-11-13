"""Export listening samples for the project page: small AAC clips plus web/data/samples.json.

uv run python scripts/site_samples.py --ids pure-34 numbers-49 \
    --system base=outputs/mos_clips/round1/base --system samleng=outputs/mos_clips/round1/student_steady \
    --ratings outputs/mos/student_v1/ratings.jsonl --rated samleng=student_steady
"""

import argparse
import json
import subprocess
from array import array
from pathlib import Path

from samleng.bench import load_bench
from samleng.paths import ROOT

WEB = ROOT / "web"
BENCH = ROOT / "bench" / "khmertts_bench_v1.jsonl"
RATE = 8000
BARS_PER_SECOND = 14


def waveform(wav: Path) -> tuple[float, list[int]]:
    """Clip length in seconds and its loudness outline as 0-100 bars."""
    cmd = ["ffmpeg", "-loglevel", "error", "-i", wav, "-ac", "1", "-ar", str(RATE)]
    raw = subprocess.run([*cmd, "-f", "s16le", "-"], check=True, capture_output=True)
    pcm = array("h", raw.stdout)
    seconds = len(pcm) / RATE
    n = max(1, round(seconds * BARS_PER_SECOND))
    size = max(1, len(pcm) // n)
    peaks = [max(map(abs, pcm[i * size : (i + 1) * size]), default=0) for i in range(n)]
    top = max(peaks) or 1
    return round(seconds, 2), [round(100 * p / top) for p in peaks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument(
        "--system", action="append", required=True, help="label=folder of <id>.wav"
    )
    ap.add_argument("--ratings", type=Path, help="ratings.jsonl of a MOS study")
    ap.add_argument(
        "--rated", action="append", default=[], help="label=system name in the study"
    )
    args = ap.parse_args()
    systems = dict(s.split("=", 1) for s in args.system)
    rated = dict(s.split("=", 1) for s in args.rated)
    scores: dict[tuple[str, str], list[int]] = {}
    if args.ratings:
        for line in args.ratings.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            scores.setdefault((r["item_id"], r["system"]), []).append(r["score"])
    bench = {it.id: it for it in load_bench(BENCH)}
    samples = []
    for i in args.ids:
        clips = {}
        for label, folder in systems.items():
            src, dst = Path(folder) / f"{i}.wav", WEB / "audio" / label / f"{i}.m4a"
            dst.parent.mkdir(parents=True, exist_ok=True)
            aac = ["-ac", "1", "-c:a", "aac", "-b:a", "64k"]
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, *aac, dst]
            subprocess.run(cmd, check=True)
            seconds, peaks = waveform(src)
            got = scores.get((i, rated.get(label, label)))
            clips[label] = {
                "seconds": seconds,
                "peaks": peaks,
                "score": round(sum(got) / len(got), 1) if got else None,
            }
        it = bench[i]
        samples.append(
            {"id": i, "category": it.category, "text": it.text, "clips": clips}
        )
    path = WEB / "data" / "samples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
