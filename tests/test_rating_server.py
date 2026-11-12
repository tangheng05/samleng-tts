import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from samleng.bench import Item
from samleng.mos import build_clips, build_pairs, load_ratings
from samleng.rating_server import MAX_BODY, ab_handler, clean_rater, mos_handler

ITEMS = [Item("pure-01", "pure", "សួស្តី"), Item("pure-02", "pure", "អរគុណ")]


@pytest.fixture
def systems(tmp_path):
    out = {}
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        for it in ITEMS:
            tone = 0.1 * np.sin(np.arange(1600) / 5)
            sf.write(folder / f"{it.id}.wav", tone, 16000)
        out[name] = folder
    return out


@pytest.fixture
def serve():
    servers = []

    def start(handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def post(url, body: bytes):
    req = urllib.request.Request(url, data=body, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_clean_rater():
    assert clean_rater("  Dara   S ") == "Dara S"
    for bad in ("", "   ", None, 7, "x" * 41, "a\x00b"):
        with pytest.raises(ValueError):
            clean_rater(bad)


def test_mos_round_trip_hides_systems_and_stores_ratings(systems, serve, tmp_path):
    clips = build_clips(ITEMS, systems)
    ratings = tmp_path / "study" / "ratings.jsonl"
    url = serve(mos_handler(clips, ratings))

    assert get(url + "/")[0] == 200
    listing = json.loads(get(url + "/api/clips?rater=Heng")[1])
    assert len(listing["clips"]) == 4 and listing["rated"] == {}
    assert not any(name in json.dumps(listing) for name in ("/a/", "/b/", "system"))
    clip = listing["clips"][0]["clip"]
    assert get(f"{url}/audio/{clip}")[1][:4] == b"RIFF"

    rate = url + "/api/rate"
    ok = json.dumps({"rater": " Heng ", "clip": clip, "score": 4}).encode()
    assert post(rate, ok) == (200, {"ok": True})
    assert json.loads(get(url + "/api/clips?rater=Heng")[1])["rated"] == {clip: 4}
    assert [r["rater"] for r in load_ratings(ratings)] == ["Heng"]

    for bad in (
        {"rater": "Heng", "clip": clip, "score": 9},
        {"rater": "Heng", "clip": "nope", "score": 3},
        {"rater": "", "clip": clip, "score": 3},
        {"rater": "x" * 60, "clip": clip, "score": 3},
        [1, 2],
    ):
        assert post(rate, json.dumps(bad).encode())[0] == 400
    assert post(rate, b" " * (MAX_BODY + 1))[0] == 400
    assert len(load_ratings(ratings)) == 1


def test_ab_round_trip(systems, serve, tmp_path):
    pairs = build_pairs(ITEMS, ("a", systems["a"]), ("b", systems["b"]), checks=1)
    ratings = tmp_path / "ab" / "ratings.jsonl"
    url = serve(ab_handler(pairs, ratings))

    listing = json.loads(get(url + "/api/pairs?rater=Heng")[1])
    assert len(listing["pairs"]) == 3
    pair = listing["pairs"][0]["pair"]
    assert get(f"{url}/audio/{pair}/1")[1][:4] == b"RIFF"
    assert get(f"{url}/audio/{pair}/2")[1][:4] == b"RIFF"

    rate = url + "/api/rate"
    body = json.dumps({"rater": "Heng", "pair": pair, "score": -2}).encode()
    assert post(rate, body) == (200, {"ok": True})
    assert (
        post(rate, json.dumps({"rater": "Heng", "pair": pair, "score": 5}).encode())[0]
        == 400
    )
    assert len(load_ratings(ratings, key="pair")) == 1
