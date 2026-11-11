"""Synthesize sentences from a JSONL manifest with VoxCPM2.

Input lines: {"id": ..., "text": ...}. Writes <out_dir>/<id>.wav and <out_dir>/results.jsonl
with {"id", "audio", "seconds", "gen_seconds"} per line.
"""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM

MODEL_ID = "openbmb/VoxCPM2"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--reference-wav", help="voice to clone (16kHz+, a few seconds)")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--resume", action="store_true", help="skip ids already in results.jsonl"
    )
    args = ap.parse_args()

    device = pick_device()
    # torch.compile optimizations are CUDA-only.
    model = VoxCPM.from_pretrained(
        MODEL_ID, load_denoiser=False, optimize=device == "cuda", device=device
    )
    sr = model.tts_model.sample_rate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    extra = {"reference_wav_path": args.reference_wav} if args.reference_wav else {}
    results = out_dir / "results.jsonl"
    done = set()
    if args.resume and results.exists():
        with open(results, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    with open(results, "a" if done else "w", encoding="utf-8") as out:
        for it in items:
            if it["id"] in done:
                continue
            # Per-item seed: same audio for a sentence regardless of run order.
            torch.manual_seed(it.get("seed", args.seed))
            # Voice design: an optional description like "young female teacher" in front.
            text = f"({it['control']}){it['text']}" if it.get("control") else it["text"]
            t = time.time()
            wav = model.generate(
                text=text,
                cfg_value=args.cfg,
                inference_timesteps=args.steps,
                **extra,
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
            out.flush()
            print(f"{it['id']}: {row['seconds']:.1f}s audio in {gen:.1f}s", flush=True)


if __name__ == "__main__":
    main()
