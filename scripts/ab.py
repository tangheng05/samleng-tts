"""Paired A/B listening test (CCR: "second compared to first", -3..+3), then CMOS per pair.

uv run python scripts/ab.py serve --study ab_v1 --bench bench/khmertts_bench_v1.jsonl \\
    --category mixed --a student=baseline/RUN/student --b gemini=baseline/khmertts_bench_v1/gemini_tts
uv run python scripts/ab.py report --study ab_v1

Several --a/--b pairs may be given (one comparison each, all in one session).
Each comparison adds --checks identical-clip pairs; raters who hear a difference there are
reported and left out of the CMOS.
"""

import argparse
import contextlib
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from samleng.bench import load_bench
from samleng.mos import build_pairs, careless_raters, load_ratings, summarize_ccr
from samleng.paths import OUTPUT_DIR, ROOT
from samleng.rating_server import ab_handler

PROTOCOL = """\
- Paired comparison, ITU-T P.800 CCR scale (-3 much worse .. +3 much better), "second vs first";
  sides shuffled per pair, both clips must be played to the end before answering.
- CMOS = mean of per-sentence means of (A minus B); ci95 is a t-interval over sentences;
  p is a Wilcoxon signed-rank test of CMOS = 0.
- Attention checks: pairs of an identical clip; raters with mean |score| > 1 on them are excluded.
- Clips are loudness-normalized to -26 dB active speech level."""


def parse_system(spec: str) -> tuple[str, Path]:
    label, _, folder = spec.partition("=")
    if not label or not folder:
        raise SystemExit(f"systems must look like label=dir, got {spec!r}")
    return label, OUTPUT_DIR / folder


def report(study: Path) -> None:
    ratings = load_ratings(study / "ratings.jsonl", key="pair")
    if not ratings:
        raise SystemExit(f"no ratings yet in {study / 'ratings.jsonl'}")
    careless = careless_raters(ratings)
    kept = [r for r in ratings if r["rater"] not in careless]
    raters = sorted({r["rater"] for r in ratings})
    excluded = ", ".join(sorted(careless)) or "none"
    lines = [
        f"# A/B: {study.name}",
        "",
        PROTOCOL,
        "",
        f"- Raters: {len(raters)}; excluded by attention checks: {excluded}",
        "",
        "## CMOS (positive = A better)",
        "",
        pd.DataFrame(summarize_ccr(kept))
        .assign(p=lambda d: d["p"].map("{:.2g}".format))
        .to_markdown(index=False),
    ]
    text = "\n".join(lines) + "\n"
    (study / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="open the A/B page")
    serve.add_argument(
        "--bench", default=str(ROOT / "bench" / "khmertts_bench_v1.jsonl")
    )
    serve.add_argument("--category", default="mixed")
    serve.add_argument(
        "--ids", help="file with one sentence id per line (replaces --category)"
    )
    serve.add_argument("--a", action="append", required=True, metavar="LABEL=DIR")
    serve.add_argument("--b", action="append", required=True, metavar="LABEL=DIR")
    serve.add_argument("--checks", type=int, default=4)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--no-open", action="store_true")
    rep = sub.add_parser("report", help="summarize ratings")
    for p in (serve, rep):
        p.add_argument("--study", required=True)
    args = ap.parse_args()

    study = OUTPUT_DIR / "mos" / args.study
    if args.cmd == "report":
        return report(study)

    if len(args.a) != len(args.b):
        raise SystemExit("give one --b for each --a")
    items = load_bench(args.bench)
    if args.ids:
        keep = set(Path(args.ids).read_text().split())
        items = [it for it in items if it.id in keep]
    else:
        items = [it for it in items if it.category == args.category]
    pairs = []
    for a, b in zip(args.a, args.b, strict=True):
        pairs += build_pairs(
            items, parse_system(a), parse_system(b), checks=args.checks
        )
    server = ThreadingHTTPServer(
        (args.host, args.port), ab_handler(pairs, study / "ratings.jsonl")
    )
    url = f"http://{args.host}:{args.port}/"
    print(
        f"{len(pairs)} pairs ({len(items)} sentences), saving to {study}/ratings.jsonl"
    )
    print(f"Rating page: {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
