"""Average the weights of several checkpoints of one run (often a little smoother than any single one).

  uv run python average.py out.pth ckpt_a.pth ckpt_b.pth [ckpt_c.pth ...]

Keeps model weights only (no optimizer state), so the result is for synthesis, not resuming.
Load it with: synthesize.py --checkpoint out.pth --config <run folder>/config.json
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("give at least two checkpoints")

    ckpts = [
        torch.load(p, map_location="cpu", weights_only=False) for p in args.checkpoints
    ]
    first = ckpts[0]["model"]
    avg = {}
    for key, value in first.items():
        if torch.is_floating_point(value):
            avg[key] = sum(c["model"][key] for c in ckpts) / len(ckpts)
        else:
            avg[key] = value
    torch.save(
        {
            "config": ckpts[-1]["config"],
            "model": avg,
            "step": max(c["step"] for c in ckpts),
            "averaged_from": [str(p) for p in args.checkpoints],
        },
        args.out,
    )
    print(f"averaged {len(ckpts)} checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
