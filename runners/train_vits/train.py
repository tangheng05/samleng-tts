"""Fine-tune sengtha's VITS on clips from scripts/prepare_data.py.

  uv run python train.py manifest.jsonl out_dir --max-cer 0.1
  uv run python train.py manifest.jsonl out_dir --limit 16 --epochs 1   # smoke test

The Coqui trainer has no MPS support, so this runs on CPU here; use a cloud GPU for real runs.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits

REPO = "sengtha/khmer-tts-female-v2"
REV = "c6fde511a839c4f20cdce96e2a35735d493030d2"


def manifest_formatter(max_cer: float, limit: int | None):
    def fmt(root_path, meta_file, **kwargs):
        samples = []
        with open(Path(root_path) / meta_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if "reject" in r or r.get("cer", 1.0) > max_cer:
                continue
            samples.append(
                {
                    "text": r["text"],
                    "audio_file": r["audio"],
                    "speaker_name": r.get("speaker", "spk"),
                    "root_path": str(Path(r["audio"]).parent),
                    "audio_unique_name": r["id"],
                }
            )
        return samples[:limit] if limit else samples

    return fmt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--max-cer", type=float, default=0.3)
    ap.add_argument("--limit", type=int, help="use only the first N clips")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-clips", type=int, default=4)
    args = ap.parse_args()

    config = VitsConfig()
    config.load_json(hf_hub_download(REPO, "config.json", revision=REV))
    config.datasets = [
        BaseDatasetConfig(
            path=str(args.manifest.parent), meta_file_train=args.manifest.name
        )
    ]
    config.output_path = str(args.out_dir)
    config.batch_size = config.eval_batch_size = args.batch_size
    config.epochs = args.epochs
    config.lr_gen = config.lr_disc = args.lr
    config.mixed_precision = False
    config.eval_split_size = args.eval_clips
    config.test_sentences = []
    config.print_step = 1 if args.limit else 25

    train, evals = load_tts_samples(
        config,
        eval_split=True,
        eval_split_size=config.eval_split_size,
        formatter=manifest_formatter(args.max_cer, args.limit),
    )
    print(f"{len(train)} train clips, {len(evals)} eval clips", flush=True)
    model = Vits.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(restore_path=hf_hub_download(REPO, "best_model.pth", revision=REV)),
        config,
        str(args.out_dir),
        model=model,
        train_samples=train,
        eval_samples=evals,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
