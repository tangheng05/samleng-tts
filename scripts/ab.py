"""Paired A/B listening test (CCR: "second compared to first", -3..+3), then CMOS per pair.

uv run python scripts/ab.py serve --study ab_v1 --bench bench/khmertts_bench_v1.jsonl \\
    --category mixed --a student=baseline/RUN/student --b gemini=baseline/khmertts_bench_v1/gemini_tts
uv run python scripts/ab.py report --study ab_v1

Several --a/--b pairs may be given (one comparison each, all in one session).
Each comparison adds --checks identical-clip pairs; raters who hear a difference there are
reported and left out of the CMOS.
"""

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from samleng.audio import leveled_wav
from samleng.bench import load_bench
from samleng.mos import (
    append_ccr,
    build_pairs,
    careless_raters,
    load_ratings,
    shuffle_for,
    summarize_ccr,
)
from samleng.paths import OUTPUT_DIR, ROOT

PAGE = Path(__file__).resolve().parents[1] / "src" / "samleng" / "mos_page" / "ab.html"
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


def make_handler(pairs, ratings_path):
    by_id = {p.pair: p for p in pairs}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body=b"", ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, status=HTTPStatus.OK):
            self.send(status, json.dumps(obj, ensure_ascii=False).encode())

        def do_GET(self):
            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            if url.path == "/":
                self.send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/pairs":
                rater = parse_qs(url.query).get("rater", [""])[0].strip()
                if not rater:
                    return self.send_json({"error": "rater is required"}, 400)
                rated = {
                    r["pair"]: r["raw"]
                    for r in load_ratings(ratings_path, key="pair")
                    if r["rater"] == rater
                }
                shuffled = shuffle_for(rater, pairs, key=lambda p: p.pair)
                self.send_json(
                    {
                        "pairs": [{"pair": p.pair, "text": p.text} for p in shuffled],
                        "rated": rated,
                    }
                )
            elif len(parts) == 3 and parts[0] == "audio" and parts[1] in by_id:
                p = by_id[parts[1]]
                clip = {"1": p.first, "2": p.second}.get(parts[2])
                if clip is None:
                    return self.send(HTTPStatus.NOT_FOUND)
                self.send(HTTPStatus.OK, leveled_wav(clip.audio), "audio/wav")
            else:
                self.send(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/api/rate":
                return self.send(HTTPStatus.NOT_FOUND)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                rater, pair = body["rater"].strip(), by_id[body["pair"]]
                if not rater:
                    raise ValueError("rater is required")
                with lock:
                    append_ccr(ratings_path, rater, pair, int(body["score"]))
            except KeyError as e:
                return self.send_json({"error": f"unknown or missing {e}"}, 400)
            except (ValueError, TypeError) as e:
                return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            self.send_json({"ok": True})

        def log_message(self, *args):
            pass

    return Handler


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
    serve.add_argument("--ids", help="file with one sentence id per line (subset)")
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
    items = [it for it in load_bench(args.bench) if it.category == args.category]
    if args.ids:
        keep = set(Path(args.ids).read_text().split())
        items = [it for it in items if it.id in keep]
    pairs = []
    for a, b in zip(args.a, args.b, strict=True):
        pairs += build_pairs(
            items, parse_system(a), parse_system(b), checks=args.checks
        )
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(pairs, study / "ratings.jsonl")
    )
    url = f"http://{args.host}:{args.port}/"
    print(
        f"{len(pairs)} pairs ({len(items)} sentences), saving to {study}/ratings.jsonl"
    )
    print(f"Rating page: {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
