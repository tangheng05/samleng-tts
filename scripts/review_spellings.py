"""Open a browser page to review drafted Khmer-script spellings of English words.

uv run python scripts/review_spellings.py outputs/translit/draft3.json texts/distill_mixed_v1.txt
uv run python scripts/review_spellings.py texts/en_km_train.json texts/*.txt --suggest new.json

Words where the draft and the CMUdict converter (samleng.en2km) disagree most come first.
With --suggest, the suggested spellings are pre-filled and listed first, showing the current one.
"Copy changes" puts the owner's edits on the clipboard as JSON.
"""

import argparse
import html
import json
import re
import webbrowser
from pathlib import Path

from rapidfuzz import fuzz

from samleng.en2km import En2Km
from samleng.paths import OUTPUT_DIR

STYLE = """
:root{--paper:#ECEFE8;--surface:#F6F8F3;--ink:#1C2A2E;--muted:#5D6B69;--line:#C9D0C6;--flag:#B8901F;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--paper:#162024;--surface:#1D292E;--ink:#E4E9E2;--muted:#97A5A1;--line:#33434A;--flag:#E0BD4E;color-scheme:dark}}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 "Avenir Next",system-ui,sans-serif}
main{max-width:52rem;margin:0 auto;padding:2rem 1rem 6rem}
.km{font-family:"Khmer Sangam MN","Noto Sans Khmer","Khmer MN",sans-serif}
h2{margin-top:2.5rem;font-size:1.1rem;color:var(--muted)}
.row{border-top:1px solid var(--line);padding:.8rem 0;display:grid;grid-template-columns:9rem 1fr;gap:.3rem 1rem}
.en{font-weight:600;padding-top:.4rem}
input{font-size:1.4rem;padding:.3rem .6rem;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:inherit}
input.changed{border-color:var(--flag);border-width:2px}
.ex{grid-column:2;color:var(--muted);font-size:1.05rem} mark{background:var(--flag);color:#000;padding:0 .2rem;border-radius:3px}
.alt{grid-column:2;color:var(--muted);font-size:.85rem}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--surface);border-top:1px solid var(--line);padding:.8rem;display:flex;gap:1rem;justify-content:center;align-items:center}
button{font:inherit;font-weight:600;border:0;border-radius:999px;padding:.6rem 1.4rem;background:var(--ink);color:var(--paper);cursor:pointer}
"""
SCRIPT = """
const ins=[...document.querySelectorAll("input")];
const upd=()=>{let n=0;ins.forEach(i=>{const c=i.value.trim()!==i.dataset.orig;i.classList.toggle("changed",c);n+=c});document.getElementById("n").textContent=n+" changed"};
ins.forEach(i=>i.addEventListener("input",upd));
document.getElementById("copy").onclick=async()=>{
  const ch={};ins.forEach(i=>{if(i.value.trim()!==i.dataset.orig)ch[i.dataset.term]=i.value.trim()});
  const txt="Spelling changes: "+(Object.keys(ch).length?JSON.stringify(ch):"none, all drafts OK");
  try{await navigator.clipboard.writeText(txt);document.getElementById("n").textContent="Copied!"}catch(e){prompt("Copy this:",txt)}
};
"""
# Below this draft-vs-converter similarity a word goes in the "check first" section.
CHECK_BELOW = 50


def example(word: str, lines: list[str]) -> str:
    pat = re.compile(rf"(?<![A-Za-z-]){re.escape(word)}(?![A-Za-z-])", re.IGNORECASE)
    for line in lines:
        if pat.search(line):
            return pat.sub(
                lambda m: f"<mark>{html.escape(m[0])}</mark>", html.escape(line)
            )
    return ""


def row(
    word: str, km: str, alt: str | None, lines: list[str], note: str = "converter"
) -> str:
    w, k = html.escape(word), html.escape(km)
    alt_html = f'<div class="alt km">{note}: {html.escape(alt)}</div>' if alt else ""
    return (
        f'<div class="row"><div class="en">{w}</div>'
        f'<input class="km" data-term="{w}" data-orig="{k}" value="{k}">'
        f'<div class="ex km">{example(word, lines)}</div>{alt_html}</div>'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("draft", type=Path, help="JSON {english: khmer}")
    ap.add_argument("texts", nargs="+", type=Path, help="files with example sentences")
    ap.add_argument(
        "--suggest", type=Path, help="JSON of suggested changes to show first"
    )
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    suggest = (
        json.loads(args.suggest.read_text(encoding="utf-8")) if args.suggest else {}
    )
    lines = [
        ln for t in args.texts for ln in t.read_text(encoding="utf-8").splitlines()
    ]
    conv = En2Km()
    scored = []
    for word, km in draft.items():
        if word in suggest:
            continue
        alt = conv.word(word)
        sim = fuzz.ratio(km, alt) if alt else 0
        scored.append((sim, word, km, alt))
    scored.sort()
    check = [s for s in scored if s[0] < CHECK_BELOW]
    rest = sorted((s for s in scored if s[0] >= CHECK_BELOW), key=lambda s: s[1])
    suggested = "".join(
        row(w, k, draft.get(w), lines, note="current")
        for w, k in sorted(suggest.items())
    )
    if suggest:
        check, rest = [], sorted(scored, key=lambda s: s[1])
    body = (
        "<h1>How should each English word be spelled in Khmer?</h1>"
        "<p>Each box is my draft of how a Khmer speaker says the English word, in Khmer script. "
        "It must match the <em>sound</em> of the English word (the training audio says the "
        "English word), not its Khmer translation. Fix any that sound wrong, then press "
        "<em>Copy changes</em> and paste into Claude Code.</p>"
        + (
            f"<h2>My {len(suggest)} suggested changes (pre-filled; your current spelling "
            "is shown below each)</h2>" + suggested
            if suggest
            else ""
        )
        + (
            f"<h2>Check these first: {len(check)} words where my draft and the "
            "rule-based converter disagree most</h2>"
            + "".join(row(w, k, a, lines) for _, w, k, a in check)
            if check
            else ""
        )
        + f"<h2>The other {len(rest)} words</h2>"
        + "".join(row(w, k, a, lines) for _, w, k, a in rest)
    )
    page = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>English Khmer Spellings</title><style>{STYLE}</style></head><body>"
        f"<main>{body}</main>"
        '<div class="bar"><span id="n">0 changed</span><button id="copy">Copy changes</button></div>'
        f"<script>{SCRIPT}</script></body></html>"
    )
    out = OUTPUT_DIR / "translit" / f"review_{args.draft.stem}.html"
    out.write_text(page, encoding="utf-8")
    print(f"{len(draft)} words ({len(suggest) or len(check)} listed first) -> {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
