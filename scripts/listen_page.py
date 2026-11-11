"""Build a side-by-side listening page: one row per sentence, one column per version.

uv run python scripts/listen_page.py outputs/train/probes/probes.jsonl out.html \\
    "Original=outputs/train/pick_v1A/base_ls1.15" "Step 16k=outputs/train/pick_v1A/step16000_ls1.15"

Each folder holds <id>.wav. Audio paths in the page are relative to it, so open it next to the folders.
"""

import argparse
import html
import os
from pathlib import Path

from samleng.runners import read_jsonl

STYLE = """
:root{--paper:#ECEFE8;--surface:#F6F8F3;--ink:#1C2A2E;--muted:#5D6B69;--line:#C9D0C6;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--paper:#162024;--surface:#1D292E;--ink:#E4E9E2;--muted:#97A5A1;--line:#33434A;color-scheme:dark}}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 "Avenir Next",system-ui,sans-serif}
main{padding:1.5rem 1rem 4rem;max-width:100rem;margin:0 auto}
.km{font-family:"Khmer Sangam MN","Noto Sans Khmer","Khmer MN",sans-serif;font-size:1.05rem}
table{border-collapse:collapse;width:100%}
th,td{border-top:1px solid var(--line);padding:.6rem .5rem;vertical-align:top;text-align:left}
th{position:sticky;top:0;background:var(--paper);font-size:.85rem}
audio{width:12rem;height:2rem}
.muted{color:var(--muted);font-size:.8rem}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences", type=Path, help="JSONL with id and text")
    ap.add_argument("out", type=Path)
    ap.add_argument("columns", nargs="+", metavar="LABEL=DIR")
    ap.add_argument("--title", default="Listening comparison")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cols = [c.split("=", 1) for c in args.columns]
    head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in cols)
    rows = []
    for s in read_jsonl(args.sentences):
        cells = []
        for _, folder in cols:
            wav = Path(folder) / f"{s['id']}.wav"
            if wav.exists():
                src = html.escape(os.path.relpath(wav, args.out.parent))
                cells.append(
                    f'<td><audio controls preload="none" src="{src}"></audio></td>'
                )
            else:
                cells.append("<td class='muted'>missing</td>")
        rows.append(
            f"<tr><td><span class='muted'>{html.escape(s['id'])}</span><br>"
            f"<span class='km'>{html.escape(s['text'])}</span></td>{''.join(cells)}</tr>"
        )
    page = (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>'
        f"<style>{STYLE}</style></head><body><main><h1>{html.escape(args.title)}</h1>"
        f"<p>{html.escape(args.note)}</p><table><tr><th>Sentence</th>{head}</tr>"
        f"{''.join(rows)}</table></main></body></html>"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"{len(rows)} sentences x {len(cols)} versions -> {args.out}")


if __name__ == "__main__":
    main()
