"""Score baseline TTS systems on a benchmark file.

uv run python scripts/baseline.py bench/pilot_v0.jsonl
"""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from samleng.bench import load_bench
from samleng.eval.report import summarize
from samleng.paths import OUTPUT_DIR
from samleng.runners import read_jsonl, run, write_jsonl
from samleng.text import verbalize
from samleng.tts import mms

SYSTEMS = ("mms", "voxcpm2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    ap.add_argument("--category", help="only run one category")
    ap.add_argument(
        "--verbalize", action="store_true", help="spell out numbers before TTS"
    )
    args = ap.parse_args()

    items = load_bench(args.bench)
    if args.category:
        items = [it for it in items if it.category == args.category]
    by_id = {it.id: it for it in items}
    tts_items = (
        [replace(it, text=verbalize(it.text)) for it in items]
        if args.verbalize
        else items
    )
    run_name = Path(args.bench).stem + (f"_{args.category}" if args.category else "")
    out = (
        OUTPUT_DIR / "baseline" / (run_name + ("_verbalized" if args.verbalize else ""))
    )

    audio: dict[str, list[dict]] = {}
    if "mms" in args.systems:
        audio["mms"] = mms.synthesize(tts_items, out / "mms")
    if "voxcpm2" in args.systems:
        manifest = out / "voxcpm2_in.jsonl"
        write_jsonl(manifest, [{"id": it.id, "text": it.text} for it in tts_items])
        run("tts_voxcpm", "synthesize.py", manifest, out / "voxcpm2")
        audio["voxcpm2"] = read_jsonl(out / "voxcpm2" / "results.jsonl")

    asr_in, asr_out = out / "asr_in.jsonl", out / "asr_out.jsonl"
    write_jsonl(
        asr_in,
        [
            {"id": f"{s}/{r['id']}", "audio": r["audio"]}
            for s, rs in audio.items()
            for r in rs
        ],
    )
    run("asr_qwen", "transcribe.py", asr_in, asr_out)
    hyps = {r["id"]: r["text"] for r in read_jsonl(asr_out)}

    df = pd.DataFrame(
        {
            "system": s,
            "id": r["id"],
            "category": by_id[r["id"]].category,
            "ref": by_id[r["id"]].text,
            "hyp": hyps[f"{s}/{r['id']}"],
            "seconds": r["seconds"],
            "gen_seconds": r["gen_seconds"],
        }
        for s, rs in audio.items()
        for r in rs
    )
    df.to_csv(out / "results.csv", index=False)
    summary = summarize(df)
    (out / "summary.md").write_text(
        summary.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
