"""Voice consistency per system: mean pairwise speaker-embedding cosine over its clips.

uv run python scripts/spk.py outputs/baseline/pilot_v1_verbalized
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from samleng.eval.spk import SpeakerEncoder, consistency


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    results = pd.read_csv(args.run_dir / "results.csv")
    enc = SpeakerEncoder()
    rows = []
    for system, g in results.groupby("system"):
        embs = np.stack([enc.embed(args.run_dir / system / f"{i}.wav") for i in g.id])
        rows.append(
            {
                "system": system,
                "clips": len(embs),
                "spk_consistency": round(consistency(embs), 3),
            }
        )
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows).sort_values("spk_consistency", ascending=False)
    table.to_csv(args.run_dir / "spk.csv", index=False)
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    main()
