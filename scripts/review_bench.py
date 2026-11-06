"""Open a browser page to review benchmark sentences (Khmer renders badly in terminals).

uv run python scripts/review_bench.py bench/pilot_v1.jsonl --question "How do you say +855 97 123 4567?"
"""

import argparse
import html
import webbrowser
from pathlib import Path

from samleng.paths import OUTPUT_DIR
from samleng.runners import read_jsonl
from samleng.text import verbalize

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review {name}</title>
<style>
  :root {{ --paper:#ECEFE8; --surface:#F6F8F3; --ink:#1C2A2E; --muted:#5D6B69;
    --line:#C9D0C6; --flag:#B8901F; color-scheme: light; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --paper:#162024; --surface:#1D292E;
    --ink:#E4E9E2; --muted:#97A5A1; --line:#33434A; --flag:#E0BD4E; color-scheme: dark; }} }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
    font:400 16px/1.5 "Avenir Next", Avenir, system-ui, sans-serif; }}
  main {{ max-width:52rem; margin:0 auto; padding:2.5rem 1.25rem 6rem; }}
  h1 {{ font-size:1.6rem; margin:0 0 .5rem; }}
  h2 {{ font-size:1.15rem; margin:2.5rem 0 .75rem; }}
  p {{ max-width:40rem; }}
  .km {{ font-family:"Khmer Sangam MN", "Noto Sans Khmer", "Khmer MN", sans-serif; }}
  .row {{ border-top:1px solid var(--line); padding:1rem 0; display:grid;
    grid-template-columns:7rem 1fr; gap:.25rem 1rem; }}
  .row.flagged {{ border-left:3px solid var(--flag); padding-left:.75rem; }}
  .id {{ color:var(--muted); font-size:.9rem; padding-top:.35rem; }}
  .text {{ font-size:1.45rem; line-height:1.8; }}
  .spoken {{ grid-column:2; color:var(--muted); font-size:1.05rem; }}
  .spoken b {{ font-family:"Avenir Next", sans-serif; font-weight:500; font-size:.8rem; }}
  .controls {{ grid-column:2; display:flex; gap:.75rem; align-items:center; flex-wrap:wrap; }}
  label {{ display:flex; gap:.35rem; align-items:center; cursor:pointer; }}
  input[type=text] {{ flex:1 1 18rem; font:inherit; padding:.45rem .7rem; border-radius:8px;
    border:1px solid var(--line); background:var(--surface); color:inherit; }}
  .q {{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
    padding:1rem 1.25rem; margin:1rem 0; }}
  .q input {{ width:100%; box-sizing:border-box; margin-top:.5rem; }}
  .bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--surface);
    border-top:1px solid var(--line); padding:.8rem 1.25rem; display:flex; gap:1rem;
    justify-content:center; align-items:center; }}
  button {{ font:inherit; font-weight:600; border:0; border-radius:999px; padding:.65rem 1.4rem;
    background:var(--ink); color:var(--paper); cursor:pointer; }}
  :focus-visible {{ outline:2px solid var(--flag); outline-offset:2px; }}
</style></head><body><main>
<h1>Review {count} sentences</h1>
<p>Read each sentence as a native speaker. If something sounds unnatural or is misspelled,
tick <em>Needs a fix</em> and type the correction or a note. Grey text under number sentences
is how the system will read it aloud: check that too. When you're done, press
<em>Copy feedback</em> and paste it into Claude Code.</p>
{questions}
{sections}
</main>
<div class="bar"><textarea id="fallback" class="km" rows="3" cols="40" hidden></textarea><span id="status">0 flagged</span><button id="copy" type="button">Copy feedback</button></div>
<script>
const rows = [...document.querySelectorAll(".row")];
const update = () => {{
  const n = rows.filter(r => r.querySelector("input[type=checkbox]").checked).length;
  rows.forEach(r => r.classList.toggle("flagged", r.querySelector("input[type=checkbox]").checked));
  document.getElementById("status").textContent = `${{n}} flagged`;
}};
document.addEventListener("input", update);
document.getElementById("copy").addEventListener("click", async () => {{
  const lines = [];
  for (const r of rows) {{
    const fix = r.querySelector("input[type=checkbox]").checked;
    const note = r.querySelector("input[type=text]").value.trim();
    if (fix || note) lines.push(`${{r.dataset.id}}: ${{note || "needs a fix"}}`);
  }}
  for (const q of document.querySelectorAll(".q input")) {{
    if (q.value.trim()) lines.push(`${{q.dataset.q}}: ${{q.value.trim()}}`);
  }}
  const text = lines.length ? lines.join("\\n") : "All sentences OK.";
  try {{
    await navigator.clipboard.writeText(text);
    document.getElementById("status").textContent = "Copied. Paste it into Claude Code.";
  }} catch {{
    const box = document.getElementById("fallback");
    box.hidden = false; box.value = text; box.select();
    document.getElementById("status").textContent = "Copy the selected text with Cmd+C.";
  }}
}});
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--all", action="store_true", help="include reviewed rows")
    ap.add_argument("--question", action="append", default=[], help="extra question")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.bench) if args.all or not r.get("reviewed")]
    sections = []
    for cat in dict.fromkeys(r["category"] for r in rows):
        items = []
        for r in (r for r in rows if r["category"] == cat):
            spoken = verbalize(r["text"])
            said = (
                f"<div class='spoken km'><b>Read as</b> {html.escape(spoken)}</div>"
                if spoken.replace(" ", "") != r["text"].replace(" ", "")
                and any(c.isdigit() for c in r["text"])
                else ""
            )
            items.append(
                f"<div class='row' data-id='{r['id']}'><div class='id'>{r['id']}</div>"
                f"<div class='text km' lang='km'>{html.escape(r['text'])}</div>{said}"
                "<div class='controls'><label><input type='checkbox'> Needs a fix</label>"
                f"<input type='text' class='km' aria-label='Note for {r['id']}' "
                "placeholder='Correction or note'></div></div>"
            )
        sections.append(f"<h2>{cat.capitalize()}</h2>" + "".join(items))
    questions = "".join(
        f"<div class='q'><strong>Q{i}.</strong> <span class='km'>{html.escape(t)}</span>"
        f"<input type='text' class='km' data-q='Q{i}' aria-label='Answer to Q{i}'></div>"
        for i, t in enumerate(args.question, 1)
    )

    out = OUTPUT_DIR / "review" / f"{Path(args.bench).stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE.format(
            name=out.stem,
            count=len(rows),
            questions=questions,
            sections="".join(sections),
        ),
        encoding="utf-8",
    )
    print(f"{len(rows)} sentences -> {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
