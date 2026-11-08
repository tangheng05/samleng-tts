"""Synthesize with a Coqui VITS checkpoint (sengtha's PyTorch weights, or our fine-tunes).

Input lines: {"id": ..., "text": ...}. Writes <out_dir>/<id>.wav and <out_dir>/results.jsonl
with {"id", "audio", "seconds", "gen_seconds"} per line.

  uv run python synthesize.py manifest.jsonl out_dir                     # sengtha base
  uv run python synthesize.py manifest.jsonl out_dir --checkpoint X.pth --config config.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from TTS.utils.synthesizer import Synthesizer

REPO = "sengtha/khmer-tts-female-v2"
REV = "c6fde511a839c4f20cdce96e2a35735d493030d2"
# The v3 ONNX we benchmark bakes these in; match them so both paths sound alike.
LENGTH_SCALE, NOISE_SCALE, NOISE_SCALE_DP = 1.15, 0.5, 0.6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = args.checkpoint or hf_hub_download(REPO, "best_model.pth", revision=REV)
    cfg = args.config or hf_hub_download(REPO, "config.json", revision=REV)
    synth = Synthesizer(tts_checkpoint=ckpt, tts_config_path=cfg, use_cuda=False)
    model = synth.tts_model
    model.length_scale = LENGTH_SCALE
    model.inference_noise_scale = NOISE_SCALE
    model.inference_noise_scale_dp = NOISE_SCALE_DP
    sr = synth.output_sample_rate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as out:
        for it in items:
            torch.manual_seed(args.seed)
            t = time.time()
            wav = np.asarray(
                synth.tts(it["text"], split_sentences=False), dtype=np.float32
            )
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
