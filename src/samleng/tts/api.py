"""Commercial reference voices over plain REST, configured like the AI-tutor's providers.

Keys come from the environment or a gitignored `.env`. Audio is cached per
(system, settings, text) so reruns cost nothing. gen_seconds is wall-clock time
including the network.
"""

import base64
import functools
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import soundfile as sf

from samleng.bench import Item

GEMINI_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI38_MODEL = "gemini-3.8-flash-tts"
GEMINI_VOICE = "Callirrhoe"
# Without this prefix the model answers the text instead of reading it.
GEMINI_STYLE = "Say in a warm, friendly voice: "
GEMINI_SR = 24000
AZURE_VOICE = "km-KH-SreymomNeural"
AZURE_RATE = "-10%"
AZURE_FORMAT = "riff-24khz-16bit-mono-pcm"
ATTEMPTS = 3


class ApiError(Exception):
    def __init__(self, msg: str, status: int | None = None, retryable: bool = False):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable or status is None or status == 429 or status >= 500


class EmptyReply(ApiError):
    """The model answered without audio: a quality failure, not an outage."""

    def __init__(self, msg: str):
        super().__init__(msg, retryable=True)


def load_env(path: Path) -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                env[key.strip()] = value.strip().strip("'\"")
    return env | dict(os.environ)


def _need(env: dict, key: str) -> str:
    if not env.get(key):
        raise SystemExit(f"{key} is not set (environment or .env)")
    return env[key]


def gemini_request(
    text: str, env: dict, model: str = GEMINI_MODEL, style: str = GEMINI_STYLE
) -> tuple[str, dict, bytes]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    headers = {
        "x-goog-api-key": _need(env, "GEMINI_API_KEY"),
        "Content-Type": "application/json",
    }
    body = {
        "contents": [{"parts": [{"text": style + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_VOICE}}
            },
        },
    }
    return url, headers, json.dumps(body).encode()


def gemini_audio(reply: bytes) -> np.ndarray:
    cand = (json.loads(reply).get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts") or [{}]
    data = parts[0].get("inlineData", {}).get("data")
    if not data:
        # The preview model sometimes returns success with no audio.
        reason = cand.get("finishReason", "unknown")
        raise EmptyReply(f"Gemini returned no audio ({reason})")
    return np.frombuffer(base64.b64decode(data), dtype="<i2")


def azure_request(text: str, env: dict) -> tuple[str, dict, bytes]:
    region = _need(env, "AZURE_SPEECH_REGION")
    headers = {
        "Ocp-Apim-Subscription-Key": _need(env, "AZURE_SPEECH_KEY"),
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": AZURE_FORMAT,
        "User-Agent": "samleng-tts",
    }
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="km-KH"><voice name="{AZURE_VOICE}">'
        f'<prosody rate="{AZURE_RATE}">{escape(text)}</prosody></voice></speak>'
    )
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    return url, headers, ssml.encode()


def azure_audio(reply: bytes) -> tuple[np.ndarray, int]:
    return sf.read(io.BytesIO(reply), dtype="int16")


def _decode(system: str, reply: bytes) -> tuple[np.ndarray, int]:
    if system.startswith("gemini"):
        return gemini_audio(reply), GEMINI_SR
    return azure_audio(reply)


SYSTEMS = {
    "gemini_tts": (gemini_request, f"{GEMINI_MODEL}|{GEMINI_VOICE}|{GEMINI_STYLE}"),
    "gemini38_tts": (
        # 3.8 reads a style prefix aloud and refuses system instructions.
        functools.partial(gemini_request, model=GEMINI38_MODEL, style=""),
        f"{GEMINI38_MODEL}|{GEMINI_VOICE}|",
    ),
    "azure_km": (azure_request, f"{AZURE_VOICE}|{AZURE_RATE}|{AZURE_FORMAT}"),
}


def http_post(url: str, headers: dict, body: bytes) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise ApiError(f"{e.code}: {e.read()[:300]!r}", status=e.code) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ApiError(str(e)) from e


def _call(system, text, env, post, sleep) -> tuple[np.ndarray, int]:
    build = SYSTEMS[system][0]
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return _decode(system, post(*build(text, env)))
        except ApiError as e:
            if not e.retryable or attempt == ATTEMPTS:
                raise
            sleep(0.5 * attempt)
    raise AssertionError("unreachable")


def synthesize(
    system: str,
    items: list[Item],
    out_dir: Path,
    env: dict,
    cache_dir: Path,
    post: Callable[[str, dict, bytes], bytes] = http_post,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        failed = False
        key = f"{system}|{SYSTEMS[system][1]}|{it.text}"
        cached = cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.wav"
        meta = cached.with_suffix(".json")
        if cached.exists() and meta.exists():
            wav, sr = sf.read(cached, dtype="int16")
            gen = json.loads(meta.read_text())["gen_seconds"]
        else:
            t = time.time()
            try:
                wav, sr = _call(system, it.text, env, post, sleep)
            except EmptyReply as e:
                # Scored as silence and not cached, so a later run retries it.
                print(f"{system} {it.id}: {e}", flush=True)
                failed, wav, sr = True, np.zeros(2400, dtype=np.int16), 24000
            gen = time.time() - t
            if not failed:
                sf.write(cached, wav, sr, subtype="PCM_16")
                meta.write_text(json.dumps({"gen_seconds": gen}))
        path = out_dir / f"{it.id}.wav"
        sf.write(path, wav, sr, subtype="PCM_16")
        rows.append(
            {
                "id": it.id,
                "audio": str(path),
                "seconds": len(wav) / sr,
                "gen_seconds": gen,
            }
            | ({"failed": True} if failed else {})
        )
    return rows
