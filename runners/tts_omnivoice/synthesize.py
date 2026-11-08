"""Synthesize sentences from a JSONL manifest with OmniVoice (auto voice, no reference).

Input lines: {"id": ..., "text": ...}. Writes <out_dir>/<id>.wav and <out_dir>/results.jsonl
with {"id", "audio", "seconds", "gen_seconds"} per line.
"""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice

MODEL_ID = "k2-fsa/OmniVoice"
SR = 24000


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--revision", default="c5fdb5ccb189668d56333f77ba2629f4cd7535f4")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    model = OmniVoice.from_pretrained(
        MODEL_ID,
        revision=args.revision,
        device_map=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as out:
        for it in items:
            # Per-item seed: same audio for a sentence regardless of run order.
            torch.manual_seed(args.seed)
            t = time.time()
            wav = model.generate(text=it["text"], num_step=args.steps)[0]
            gen = time.time() - t
            path = out_dir / f"{it['id']}.wav"
            sf.write(path, wav, SR)
            row = {
                "id": it["id"],
                "audio": str(path),
                "seconds": len(wav) / SR,
                "gen_seconds": gen,
            }
            out.write(json.dumps(row) + "\n")
            print(f"{it['id']}: {row['seconds']:.1f}s audio in {gen:.1f}s", flush=True)


if __name__ == "__main__":
    main()
