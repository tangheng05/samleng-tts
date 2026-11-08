"""Fine-tune sengtha's VITS on a checked manifest: every row without `reject`.

  uv run python train.py MANIFEST OUT_DIR
  uv run python train.py MANIFEST OUT_DIR --continue-run OUT_DIR/<run folder>   # resume
  uv run python train.py MANIFEST OUT_DIR --limit 16 --epochs 1                 # smoke test

The data scripts decide what to keep. The Coqui trainer has no MPS support, so on a Mac
this trains on CPU: slow but free.
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


def manifest_formatter(limit: int | None):
    def fmt(root_path, meta_file, **kwargs):
        samples = []
        with open(Path(root_path) / meta_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if "reject" in r or "audio" not in r:
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
    ap.add_argument("--continue-run", type=Path, help="run folder to resume")
    ap.add_argument("--save-step", type=int, default=1000)
    ap.add_argument("--limit", type=int, help="use only the first N clips")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-clips", type=int, default=4)
    args = ap.parse_args()

    config = VitsConfig()
    if args.continue_run:
        config.load_json(str(args.continue_run / "config.json"))
    else:
        config.load_json(hf_hub_download(REPO, "config.json", revision=REV))
        configure(config, args)
    config.epochs = args.epochs
    config.save_step = args.save_step
    # Checkpoints are ~1 GB each (optimizer state included).
    config.save_n_checkpoints = 3

    train, evals = load_tts_samples(
        config,
        eval_split=True,
        eval_split_size=config.eval_split_size,
        formatter=manifest_formatter(args.limit),
    )
    print(f"{len(train)} train clips, {len(evals)} eval clips", flush=True)
    model = Vits.init_from_config(config)
    if args.continue_run:
        targs = TrainerArgs(continue_path=str(args.continue_run))
    else:
        start = hf_hub_download(REPO, "best_model.pth", revision=REV)
        targs = TrainerArgs(restore_path=start)
    out = str(args.continue_run or args.out_dir)
    trainer = Trainer(
        targs, config, out, model=model, train_samples=train, eval_samples=evals
    )
    trainer.fit()


def configure(config: VitsConfig, args: argparse.Namespace) -> None:
    config.datasets = [
        BaseDatasetConfig(
            path=str(args.manifest.parent), meta_file_train=args.manifest.name
        )
    ]
    config.output_path = str(args.out_dir)
    config.batch_size = config.eval_batch_size = args.batch_size
    config.lr_gen = config.lr_disc = args.lr
    config.mixed_precision = False
    config.eval_split_size = args.eval_clips
    config.test_sentences = []
    config.print_step = 1 if args.limit else 25


if __name__ == "__main__":
    main()
