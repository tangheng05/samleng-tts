"""Voice the fixed probe sentences with a run's newest checkpoint and rebuild the comparison page.

  uv run python probe.py ../../outputs/train/distill_v1_A/<run folder>
Writes ../../outputs/train/probes/<step>/*.wav and ../../outputs/train/probes/listen.html.
"""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

PROBES = Path(__file__).resolve().parents[2] / "outputs" / "train" / "probes"


def newest_checkpoint(run: Path) -> tuple[int, Path]:
    steps = [
        (int(m[1]), p)
        for p in run.glob("checkpoint_*.pth")
        if (m := re.search(r"_(\d+)\.pth$", p.name))
    ]
    if not steps:
        raise SystemExit(f"no checkpoint_*.pth in {run}")
    return max(steps)


def page() -> None:
    rows = [
        json.loads(line)
        for line in (PROBES / "probes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    steps = (d.name for d in PROBES.iterdir() if d.is_dir() and d.name.isdigit())
    columns = ["base", *sorted(steps, key=int)]
    head = "".join(
        f"<th>{'before (sengtha)' if c == 'base' else f'step {c}'}</th>"
        for c in columns
    )
    body = "".join(
        f"<tr><td><span class='g'>{r['group']}</span><br><span class='km'>{html.escape(r['text'])}</span></td>"
        + "".join(
            f"<td><audio controls preload='none' src='{c}/{r['id']}.wav'></audio></td>"
            for c in columns
        )
        + "</tr>"
        for r in rows
    )
    (PROBES / "listen.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training Progress</title><style>
:root{{--bg:#fafafa;--card:#fff;--ink:#222;--muted:#666;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#162024;--card:#1D292E;--ink:#E4E9E2;--muted:#97A5A1;--line:#33434A}}}}
body{{font-family:-apple-system,sans-serif;margin:24px auto;padding:0 16px;max-width:1200px;background:var(--bg);color:var(--ink)}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;background:var(--card)}} td,th{{border:1px solid var(--line);padding:8px;vertical-align:top}}
th{{font-size:13px;text-align:left}} .km{{font-family:"Khmer Sangam MN","Noto Sans Khmer",sans-serif;font-size:17px}}
.g{{color:var(--muted);font-size:12px}} audio{{width:210px}}</style></head><body>
<h1>Samleng training progress</h1>
<p><b>unseen</b> = English words never in the training data (real learning); <b>seen</b> = trained words.</p>
<div class="wrap"><table><tr><th>sentence</th>{head}</tr>{body}</table></div></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    step, ckpt = newest_checkpoint(run)
    out = PROBES / str(step)
    if not (out / "results.jsonl").exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "synthesize.py"),
                str(PROBES / "probes.jsonl"),
                str(out),
                "--checkpoint",
                str(ckpt),
                "--config",
                str(run / "config.json"),
            ],
            check=True,
        )
    page()
    print(f"step {step} -> {PROBES / 'listen.html'}")


if __name__ == "__main__":
    main()
