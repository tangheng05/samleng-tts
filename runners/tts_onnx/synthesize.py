"""Synthesize with small Khmer VITS voices exported to ONNX (CPU, no PyTorch).

Input lines: {"id": ..., "text": ...}. Writes <out_dir>/<id>.wav and <out_dir>/results.jsonl
with {"id", "audio", "seconds", "gen_seconds"} per line.

  sengtha: sengtha/khmer-tts-female-v2, single female voice, plain graphemes.
  zio:     sombochea/zio-tts, multi-voice, word-segmented graphemes (default voice).
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import hf_hub_download
from khmernltk import word_tokenize

KHMER = re.compile("[ក-៿]")


def sengtha_voice(repo: str, rev: str):
    sess = ort.InferenceSession(
        hf_hub_download(repo, "khmer_tts_v3.onnx", revision=rev)
    )
    meta_path = hf_hub_download(repo, "tts_meta.json", revision=rev)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    id_of = {c: i for i, c in enumerate(meta["vocab"])}
    blank = id_of[meta["blank"]]

    def speak(text: str) -> np.ndarray:
        ids = [id_of[c] for c in text if c in id_of]
        if meta["add_blank"]:
            ids = [blank] + [t for i in ids for t in (i, blank)]
        x = np.array([ids], dtype=np.int64)
        lengths = np.array([x.shape[1]], dtype=np.int64)
        return np.asarray(sess.run(None, {"x": x, "x_lengths": lengths})[0]).squeeze()

    return speak, meta["sample_rate"]


def zio_voice(repo: str, rev: str):
    """Mirrors upstream examples/onnx_infer.py, minus its digit normalizer."""
    sess = ort.InferenceSession(hf_hub_download(repo, "zio-tts.onnx", revision=rev))
    vocab_path = hf_hub_download(repo, "vocab.json", revision=rev)
    v = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    sid = np.array([v["voices"][v["default_voice"]]], dtype=np.int64)
    # Trained on word-spaced Khmer (SLR42); upstream segments with khmer-nltk too.
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    def speak(text: str) -> np.ndarray:
        if KHMER.search(text):
            text = " ".join(w for w in word_tokenize(text) if w.strip())
        chars = [v["char_to_id"][c] for c in text.lower() if c in v["char_to_id"]]
        ids = [v["blank_id"]] * (2 * len(chars) + 1)
        ids[1::2] = chars
        x = np.array([ids], dtype=np.int64)
        feed = {
            "input": x,
            "input_lengths": np.array([x.shape[1]], dtype=np.int64),
            "scales": scales,
            "sid": sid,
        }
        return np.asarray(sess.run(None, feed)[0]).squeeze()

    return speak, 24000


MODELS = {
    "sengtha": (
        sengtha_voice,
        "sengtha/khmer-tts-female-v2",
        "c6fde511a839c4f20cdce96e2a35735d493030d2",
    ),
    "zio": (zio_voice, "sombochea/zio-tts", "71d5fbc535a024ffe86bb3d798264d58c6c6e096"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load, repo, rev = MODELS[args.model]
    speak, sr = load(repo, rev)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as out:
        for it in items:
            # VITS samples noise inside the graph; reseed so audio is reproducible.
            ort.set_seed(args.seed)
            t = time.time()
            wav = speak(it["text"])
            gen = time.time() - t
            path = out_dir / f"{it['id']}.wav"
            sf.write(path, wav, sr)
            row = {
                "id": it["id"],
                "audio": str(path),
                "seconds": len(wav) / sr,
                "gen_seconds": gen,
            }
            out.write(json.dumps(row) + "\n")
            print(f"{it['id']}: {row['seconds']:.1f}s audio in {gen:.2f}s", flush=True)


if __name__ == "__main__":
    main()
