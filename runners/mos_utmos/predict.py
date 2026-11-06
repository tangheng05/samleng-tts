"""Predict naturalness (1-5) for audio listed in a JSONL manifest with UTMOSv2.

Input lines: {"id": ..., "audio": "path.wav"}. Output lines: {"id": ..., "utmos": ...}.
UTMOSv2 is trained on English/Chinese/Japanese: check it against human ratings first.
"""

import argparse
import json

import numpy as np
import torch
import utmosv2
from huggingface_hub import hf_hub_download

REPO = "sarulab-speech/UTMOSv2"
REVISION = "506474f2b33dc77c234d668cc419be1861899cad"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repetitions", type=int, default=5, help="random crops averaged")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # The library downloads weights with wget; fetch a pinned revision instead.
    ckpt = hf_hub_download(REPO, "fold0_s42_best_model.pth", revision=REVISION)
    model = utmosv2.create_model(checkpoint_path=ckpt, device=args.device)
    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    with open(args.out, "w", encoding="utf-8") as out:
        for it in items:
            # UTMOSv2 crops each clip at random; reseed so scores are reproducible.
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            mos = model.predict(
                input_path=it["audio"],
                device=args.device,
                num_repetitions=args.repetitions,
                verbose=False,
            )
            out.write(
                json.dumps({"id": it["id"], "utmos": round(float(mos), 4)}) + "\n"
            )
            print(f"{it['id']}: {float(mos):.2f}", flush=True)


if __name__ == "__main__":
    main()
