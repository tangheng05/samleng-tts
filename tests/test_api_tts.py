import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf

from samleng.bench import Item
from samleng.tts import api

ENV = {
    "GEMINI_API_KEY": "g-key",
    "AZURE_SPEECH_KEY": "a-key",
    "AZURE_SPEECH_REGION": "southeastasia",
}
PCM = np.array([0, 1000, -1000, 32767], dtype="<i2")


def gemini_reply(pcm=PCM):
    data = base64.b64encode(pcm.tobytes()).decode()
    parts = [{"inlineData": {"data": data}}] if len(pcm) else [{}]
    return json.dumps({"candidates": [{"content": {"parts": parts}}]}).encode()


def wav_bytes(pcm=PCM, sr=24000):
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakePost:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, body))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_gemini_request_matches_ai_tutor():
    url, headers, body = api.gemini_request("សួស្តី", ENV)
    assert url.endswith("/models/gemini-2.5-flash-preview-tts:generateContent")
    assert headers["x-goog-api-key"] == "g-key"
    cfg = json.loads(body)
    assert cfg["contents"][0]["parts"][0]["text"] == api.GEMINI_STYLE + "សួស្តី"
    voice = cfg["generationConfig"]["speechConfig"]["voiceConfig"]
    assert voice["prebuiltVoiceConfig"]["voiceName"] == "Callirrhoe"


def test_gemini_audio_decodes_pcm_and_flags_empty_replies():
    assert np.array_equal(api.gemini_audio(gemini_reply()), PCM)
    with pytest.raises(api.ApiError) as e:
        api.gemini_audio(gemini_reply(np.array([], dtype="<i2")))
    assert e.value.retryable


def test_azure_request_escapes_ssml_and_asks_for_wav():
    url, headers, body = api.azure_request("A & <B>", ENV)
    assert url == "https://southeastasia.tts.speech.microsoft.com/cognitiveservices/v1"
    assert headers["Ocp-Apim-Subscription-Key"] == "a-key"
    assert headers["X-Microsoft-OutputFormat"] == "riff-24khz-16bit-mono-pcm"
    ssml = body.decode()
    assert "A &amp; &lt;B&gt;" in ssml and "km-KH-SreymomNeural" in ssml
    assert 'rate="-10%"' in ssml


def test_missing_key_is_a_clear_error():
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        api.gemini_request("x", {})


def test_synthesize_writes_wavs_and_reuses_the_cache(tmp_path):
    items = [Item(id="a-01", category="pure", text="សួស្តី")]
    post = FakePost(gemini_reply())
    kw = {"env": ENV, "cache_dir": tmp_path / "cache", "post": post}
    rows = api.synthesize("gemini_tts", items, tmp_path / "out", **kw)
    wav, sr = sf.read(rows[0]["audio"], dtype="int16")
    assert sr == 24000 and np.array_equal(wav, PCM)
    assert rows[0]["seconds"] == pytest.approx(len(PCM) / 24000)

    again = api.synthesize("gemini_tts", items, tmp_path / "out2", **kw)
    assert len(post.calls) == 1
    assert again[0]["gen_seconds"] == rows[0]["gen_seconds"]


def test_azure_wav_reply_is_saved(tmp_path):
    items = [Item(id="a-01", category="pure", text="សួស្តី")]
    post = FakePost(wav_bytes())
    rows = api.synthesize(
        "azure_km", items, tmp_path / "out", env=ENV, cache_dir=tmp_path, post=post
    )
    assert np.array_equal(sf.read(rows[0]["audio"], dtype="int16")[0], PCM)


def test_retries_transient_errors_but_not_client_errors(tmp_path):
    items = [Item(id="a-01", category="pure", text="x")]
    post = FakePost(api.ApiError("boom", status=500), gemini_reply())
    api.synthesize(
        "gemini_tts",
        items,
        tmp_path / "o",
        env=ENV,
        cache_dir=tmp_path,
        post=post,
        sleep=lambda s: None,
    )
    assert len(post.calls) == 2

    post = FakePost(api.ApiError("bad", status=400), gemini_reply())
    with pytest.raises(api.ApiError):
        api.synthesize(
            "gemini_tts",
            [Item(id="b", category="pure", text="y")],
            tmp_path / "o",
            env=ENV,
            cache_dir=tmp_path,
            post=post,
            sleep=lambda s: None,
        )
    assert len(post.calls) == 1


def test_load_env_reads_dotenv_and_lets_real_env_win(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nGEMINI_API_KEY=file\nAZURE_SPEECH_REGION='eastus'\n\n")
    monkeypatch.setenv("GEMINI_API_KEY", "shell")
    env = api.load_env(f)
    assert env["GEMINI_API_KEY"] == "shell"
    assert env["AZURE_SPEECH_REGION"] == "eastus"
