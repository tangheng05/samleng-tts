"""Transcribe audio listed in a JSONL manifest with Qwen3-ASR-0.6B-Khmer.

Input lines: {"id": ..., "audio": "path.wav"}. Output lines: {"id": ..., "text": ...}.
"""

import argparse
import json

import torch
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel

MODEL_ID = "seanghay/Qwen3-ASR-0.6B-Khmer"
# Every CER so far was judged by this revision.
REVISION = "351f1cc594f084e4899ef6cf7bbb17ec64484080"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", help="cpu, mps or cuda:0 (default: best available)")
    args = ap.parse_args()

    device = args.device or pick_device()
    model = Qwen3ASRModel.from_pretrained(
        snapshot_download(MODEL_ID, revision=REVISION),
        dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map=device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=256,
    )

    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    results = model.transcribe(audio=[it["audio"] for it in items])
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps({"id": it["id"], "text": r.text}, ensure_ascii=False) + "\n"
            for it, r in zip(items, results, strict=True)
        )


if __name__ == "__main__":
    main()
