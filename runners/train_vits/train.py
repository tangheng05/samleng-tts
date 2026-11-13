"""Fine-tune sengtha's VITS on a checked manifest: every row without `reject`.

  uv run python train.py MANIFEST OUT_DIR
  uv run python train.py MANIFEST OUT_DIR --continue-run OUT_DIR/<run folder>   # resume
  uv run python train.py MANIFEST OUT_DIR --limit 16 --epochs 1                 # smoke test

The data scripts decide what to keep. `--device mps` trains on the Apple GPU through
Accelerate (about 2.7x faster than CPU); every run can resume with `--continue-run`.
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits

REPO = "sengtha/khmer-tts-female-v2"
REV = "c6fde511a839c4f20cdce96e2a35735d493030d2"


class DeviceAwareTrainer(Trainer):
    """Coqui's trainer only moves training batches through Accelerate; do eval too."""

    def get_eval_dataloader(self, samples, *, verbose=True):
        loader = super().get_eval_dataloader(samples, verbose=verbose)
        return self.prepare_accelerate_loader(loader)

    def save_checkpoint(self) -> None:
        super().save_checkpoint()
        torch.cuda.empty_cache()

    def train_step(self, *args, **kwargs):
        out = super().train_step(*args, **kwargs)
        if torch.backends.mps.is_available() and self.total_steps_done % 25 == 0:
            # Batches grow from short to long clips; the cache reached 15.8 GB by the end of an epoch.
            torch.mps.empty_cache()
            gb = torch.mps.driver_allocated_memory() / 1e9
            print(f"     | > mps_gb: {gb:.1f}", flush=True)
        return out


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
    ap.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="mps: Apple GPU via Accelerate",
    )
    ap.add_argument("--limit", type=int, help="use only the first N clips")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-clips", type=int, default=4)
    # Each loader worker is a separate process with its own memory; on a 16 GB Mac, 8 train
    # workers plus 4 eval workers got an eval worker killed at the end of the first epoch.
    ap.add_argument(
        "--workers", type=int, default=4, help="train data loader processes"
    )
    ap.add_argument(
        "--eval-workers",
        type=int,
        default=0,
        help="eval data loader processes (0 = load in the main process)",
    )
    args = ap.parse_args()

    config = VitsConfig()
    if args.continue_run:
        config.load_json(str(args.continue_run / "config.json"))
    else:
        config.load_json(hf_hub_download(REPO, "config.json", revision=REV))
        configure(config, args)
    config.epochs = args.epochs
    config.num_loader_workers = args.workers
    config.num_eval_loader_workers = args.eval_workers
    config.save_step = args.save_step
    # Checkpoints are ~1 GB each (optimizer state included).
    config.save_n_checkpoints = 3

    train, evals = load_tts_samples(
        config,
        eval_split=True,
        eval_split_size=config.eval_split_size,
        formatter=manifest_formatter(args.limit),
    )
    # A clip can have two rows (English in both scripts): keep held-out audio out of training.
    held = {s["audio_file"] for s in evals}
    train = [s for s in train if s["audio_file"] not in held]
    print(f"{len(train)} train clips, {len(evals)} eval clips", flush=True)
    model = Vits.init_from_config(config)
    if args.continue_run:
        targs = TrainerArgs(continue_path=str(args.continue_run))
    else:
        start = hf_hub_download(REPO, "best_model.pth", revision=REV)
        targs = TrainerArgs(restore_path=start)
    # The trainer only knows CUDA; Accelerate is its route to Apple's GPU.
    targs.use_accelerate = args.device == "mps"
    if args.device == "mps":
        # Its cache frees after eval and each epoch are CUDA-only; without them MPS ran out of memory.
        torch.cuda.empty_cache = torch.mps.empty_cache
    out = str(args.continue_run or args.out_dir)
    trainer = DeviceAwareTrainer(
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
