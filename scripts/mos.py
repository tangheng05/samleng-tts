"""Collect human MOS ratings in the browser, then summarize them.

uv run python scripts/mos.py serve
uv run python scripts/mos.py report
uv run python scripts/mos.py report --study mixed_v0 \
    --merge mms=mms_raw,mms_verbalized --merge voxcpm2=voxcpm2_raw,voxcpm2_verbalized
"""

import argparse
import functools
import io
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import soundfile as sf

from samleng.audio import normalize_loudness
from samleng.bench import load_bench
from samleng.mos import (
    PAGE,
    append_rating,
    build_clips,
    compare,
    load_ratings,
    merge_systems,
    order_for,
    summarize,
)
from samleng.paths import OUTPUT_DIR, ROOT

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


@functools.cache
def leveled_wav(path):
    y, sr = sf.read(path, always_2d=False)
    buf = io.BytesIO()
    sf.write(
        buf,
        normalize_loudness(y if y.ndim == 1 else y.mean(axis=1), sr),
        sr,
        format="WAV",
        subtype="PCM_16",
    )
    return buf.getvalue()


def make_handler(clips, ratings_path):
    by_id = {c.clip: c for c in clips}
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
            if url.path == "/":
                self.send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/clips":
                rater = parse_qs(url.query).get("rater", [""])[0].strip()
                if not rater:
                    return self.send_json({"error": "rater is required"}, 400)
                rated = {
                    r["clip"]: r["score"]
                    for r in load_ratings(ratings_path)
                    if r["rater"] == rater
                }
                order = [
                    {"clip": c.clip, "text": c.text} for c in order_for(rater, clips)
                ]
                self.send_json({"clips": order, "rated": rated})
            elif url.path.startswith("/audio/") and url.path[7:] in by_id:
                self.send(
                    HTTPStatus.OK, leveled_wav(by_id[url.path[7:]].audio), "audio/wav"
                )
            else:
                self.send(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/api/rate":
                return self.send(HTTPStatus.NOT_FOUND)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                rater, clip = body["rater"].strip(), by_id[body["clip"]]
                if not rater:
                    raise ValueError("rater is required")
                with lock:
                    append_rating(ratings_path, rater, clip, int(body["score"]))
            except KeyError as e:
                return self.send_json({"error": f"unknown or missing {e}"}, 400)
            except (ValueError, TypeError) as e:
                return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            self.send_json({"ok": True})

        def log_message(self, *args):
            pass

    return Handler


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

    items = [it for it in load_bench(args.bench) if it.category == args.category]
    clips = build_clips(items, parse_systems(args.system))
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(clips, ratings_path)
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(clips)} clips, saving to {ratings_path}")
    print(f"Rating page: {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
