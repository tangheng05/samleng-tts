"""Collect human MOS ratings in the browser, then summarize them.

uv run python scripts/mos.py serve
uv run python scripts/mos.py report
uv run python scripts/mos.py report --study mixed_v0 \
    --merge mms=mms_raw,mms_verbalized --merge voxcpm2=voxcpm2_raw,voxcpm2_verbalized
"""

import argparse
import contextlib
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from samleng.bench import load_bench
from samleng.mos import build_clips, compare, load_ratings, merge_systems, summarize
from samleng.paths import OUTPUT_DIR, ROOT
from samleng.rating_server import mos_handler

DEFAULT_SYSTEMS = {
    "mms": "baseline/pilot_v1/mms",
    "voxcpm2": "baseline/pilot_v1/voxcpm2",
}
PROTOCOL = """\
- Scale: 1-5 absolute category rating with ITU-T P.800 labels (Bad, Poor, Fair, Good, Excellent).
- MOS is the mean of per-sentence means; ci95 is a t-interval over sentences.
- p is a paired Wilcoxon signed-rank test over the sentences both systems share.
- Clips are loudness-normalized to -26 dB active speech level; native sample rates kept.
- MOS values are only comparable within this study."""


def parse_merge(pairs):
    groups = {}
    for pair in pairs or []:
        label, _, olds = pair.partition("=")
        if not label or not olds:
            raise SystemExit(f"--merge must look like new=old1,old2, got {pair!r}")
        groups[label] = olds.split(",")
    return groups


def write_report(study, ratings, groups):
    lines = [f"# MOS: {study.name}", "", PROTOCOL]
    lines += [f"- Merged: {k} = {' + '.join(v)}" for k, v in groups.items()]
    lines += [
        "",
        "## Systems",
        "",
        pd.DataFrame(summarize(ratings)).to_markdown(index=False),
    ]
    pairs = compare(ratings)
    if pairs:
        table = pd.DataFrame(pairs).assign(p=lambda d: d["p"].map("{:.2g}".format))
        lines += ["", "## Pairwise", "", table.to_markdown(index=False)]
    text = "\n".join(lines) + "\n"
    print(text)
    (study / "summary.md").write_text(text, encoding="utf-8")


def parse_systems(pairs):
    if not pairs:
        return {k: OUTPUT_DIR / v for k, v in DEFAULT_SYSTEMS.items()}
    systems = {}
    for pair in pairs:
        label, _, folder = pair.partition("=")
        if not label or not folder:
            raise SystemExit(f"--system must look like label=dir, got {pair!r}")
        systems[label] = OUTPUT_DIR / folder
    return systems


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="open the rating page")
    serve.add_argument("--bench", default=str(ROOT / "bench" / "pilot_v1.jsonl"))
    serve.add_argument("--category", default="mixed")
    serve.add_argument(
        "--system",
        action="append",
        metavar="LABEL=DIR",
        help="audio folder (relative to outputs/); repeat per system",
    )
    serve.add_argument(
        "--ids", help="file with one sentence id per line (replaces --category)"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true")
    report = sub.add_parser("report", help="summarize ratings")
    report.add_argument(
        "--merge",
        action="append",
        metavar="NEW=OLD1,OLD2",
        help="pool systems, e.g. two takes of the same voice",
    )
    for p in (serve, report):
        p.add_argument("--study", default="mixed_v2")
    args = ap.parse_args()

    study = OUTPUT_DIR / "mos" / args.study
    ratings_path = study / "ratings.jsonl"

    if args.cmd == "report":
        ratings = load_ratings(ratings_path)
        if not ratings:
            raise SystemExit(f"no ratings yet in {ratings_path}")
        groups = parse_merge(args.merge)
        write_report(study, merge_systems(ratings, groups), groups)
        return

    items = load_bench(args.bench)
    if args.ids:
        keep = set(Path(args.ids).read_text().split())
        items = [it for it in items if it.id in keep]
    else:
        items = [it for it in items if it.category == args.category]
    clips = build_clips(items, parse_systems(args.system))
    server = ThreadingHTTPServer(
        (args.host, args.port), mos_handler(clips, ratings_path)
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"{len(clips)} clips, saving to {ratings_path}")
    print(f"Rating page: {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
