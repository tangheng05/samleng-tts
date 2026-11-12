"""HTTP handlers for the blind rating pages (MOS and A/B), shared by scripts/mos.py and scripts/ab.py.

The pages are static files; the handlers serve opaque clip ids, loudness-leveled audio and
append one JSON line per answer. Requests are size-capped and rater names validated, so the
server can sit behind a tunnel for remote raters.
"""

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from samleng.audio import leveled_wav
from samleng.mos import (
    PAGE,
    Clip,
    Pair,
    append_ccr,
    append_rating,
    load_ratings,
    order_for,
    shuffle_for,
)

AB_PAGE = PAGE.with_name("ab.html")
MAX_BODY = 4096
MAX_RATER = 40


def clean_rater(name: object) -> str:
    """A rater name as stored: trimmed, 1-40 printable characters."""
    if not isinstance(name, str):
        raise ValueError("rater is required")
    name = " ".join(name.split())
    if not name:
        raise ValueError("rater is required")
    if len(name) > MAX_RATER or not name.isprintable():
        raise ValueError(f"rater must be at most {MAX_RATER} printable characters")
    return name


class JsonHandler(BaseHTTPRequestHandler):
    def send(self, status, body=b"", ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=HTTPStatus.OK):
        self.send(status, json.dumps(obj, ensure_ascii=False).encode())

    def read_json(self) -> dict:
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_BODY:
            raise ValueError("request body missing or too large")
        body = json.loads(self.rfile.read(size))
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        return body

    def query_rater(self) -> str:
        return clean_rater(parse_qs(urlparse(self.path).query).get("rater", [""])[0])

    def log_message(self, *args):
        pass


def mos_handler(clips: list[Clip], ratings_path: Path) -> type[JsonHandler]:
    """Absolute 1-5 ratings: GET /, /api/clips?rater=, /audio/<clip>; POST /api/rate."""
    by_id = {c.clip: c for c in clips}
    lock = threading.Lock()

    class Handler(JsonHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/clips":
                try:
                    rater = self.query_rater()
                except ValueError as e:
                    return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                rated = {
                    r["clip"]: r["score"]
                    for r in load_ratings(ratings_path)
                    if r["rater"] == rater
                }
                order = [
                    {"clip": c.clip, "text": c.text} for c in order_for(rater, clips)
                ]
                self.send_json({"clips": order, "rated": rated})
            elif path.startswith("/audio/") and path[7:] in by_id:
                self.send(
                    HTTPStatus.OK, leveled_wav(by_id[path[7:]].audio), "audio/wav"
                )
            else:
                self.send(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/api/rate":
                return self.send(HTTPStatus.NOT_FOUND)
            try:
                body = self.read_json()
                rater, clip = clean_rater(body.get("rater")), by_id[body["clip"]]
                with lock:
                    append_rating(ratings_path, rater, clip, int(body["score"]))
            except KeyError as e:
                return self.send_json(
                    {"error": f"unknown or missing {e}"}, HTTPStatus.BAD_REQUEST
                )
            except (ValueError, TypeError) as e:
                return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            self.send_json({"ok": True})

    return Handler


def ab_handler(pairs: list[Pair], ratings_path: Path) -> type[JsonHandler]:
    """Pairwise CCR ratings: GET /, /api/pairs?rater=, /audio/<pair>/<1|2>; POST /api/rate."""
    by_id = {p.pair: p for p in pairs}
    lock = threading.Lock()

    class Handler(JsonHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            parts = path.strip("/").split("/")
            if path == "/":
                self.send(
                    HTTPStatus.OK, AB_PAGE.read_bytes(), "text/html; charset=utf-8"
                )
            elif path == "/api/pairs":
                try:
                    rater = self.query_rater()
                except ValueError as e:
                    return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
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
                body = self.read_json()
                rater, pair = clean_rater(body.get("rater")), by_id[body["pair"]]
                with lock:
                    append_ccr(ratings_path, rater, pair, int(body["score"]))
            except KeyError as e:
                return self.send_json(
                    {"error": f"unknown or missing {e}"}, HTTPStatus.BAD_REQUEST
                )
            except (ValueError, TypeError) as e:
                return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            self.send_json({"ok": True})

    return Handler
