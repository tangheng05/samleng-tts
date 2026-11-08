"""Score baseline TTS systems on a benchmark file.

uv run python scripts/baseline.py bench/pilot_v0.jsonl
"""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from samleng.bench import load_bench
from samleng.eval.report import merge_results, summarize
from samleng.paths import OUTPUT_DIR
from samleng.runners import read_jsonl, run, write_jsonl
from samleng.text import transliterate, verbalize
from samleng.tts import mms

# transformers VITS models: label -> (model id, pinned revision)
VITS = {
    "mms": ("facebook/mms-tts-khm", None),
    "mms_kto": (
        "khmerttsopensource/khmer-tts",
        "c9380a9d1c817f2b21980d54f29f7a61b0d626f5",
    ),
    "mms_krorng": (
        "KrorngAI/mms-tts-khm-finetuned",
        "a0ce13ee0d7e1d1027b948b5067093fc0bce4fb5",
    ),
}
# Models in isolated runners: label -> (runner, script, args before the manifest)
RUNNER_TTS = {
    "voxcpm2": ("tts_voxcpm", "synthesize.py", ()),
    "omnivoice": ("tts_omnivoice", "synthesize.py", ()),
    "sengtha": ("tts_onnx", "synthesize.py", ("sengtha",)),
    "zio": ("tts_onnx", "synthesize.py", ("zio",)),
}
SYSTEMS = (*VITS, *RUNNER_TTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    ap.add_argument("--category", help="only run one category")
    ap.add_argument(
        "--verbalize", action="store_true", help="spell out numbers before TTS"
    )
    ap.add_argument(
        "--transliterate",
        action="store_true",
        help="write known English words in Khmer script before TTS",
    )
    args = ap.parse_args()

    items = load_bench(args.bench)
    if args.category:
        items = [it for it in items if it.category == args.category]
    by_id = {it.id: it for it in items}
    tts_items = items
    if args.verbalize:
        tts_items = [replace(it, text=verbalize(it.text)) for it in tts_items]
    if args.transliterate:
        tts_items = [replace(it, text=transliterate(it.text)) for it in tts_items]
    run_name = Path(args.bench).stem + (f"_{args.category}" if args.category else "")
    run_name += "_verbalized" * args.verbalize + "_translit" * args.transliterate
    out = OUTPUT_DIR / "baseline" / run_name

    audio: dict[str, list[dict]] = {}
    for label, (model_id, rev) in VITS.items():
        if label in args.systems:
            audio[label] = mms.synthesize(
                tts_items, out / label, model_id=model_id, revision=rev
            )
    for label, (runner, script, prefix) in RUNNER_TTS.items():
        if label not in args.systems:
            continue
        manifest = out / f"{label}_in.jsonl"
        write_jsonl(manifest, [{"id": it.id, "text": it.text} for it in tts_items])
        run(runner, script, *prefix, manifest, out / label)
        audio[label] = read_jsonl(out / label / "results.jsonl")

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
    if (out / "results.csv").exists():
        df = merge_results(pd.read_csv(out / "results.csv"), df)
    df.to_csv(out / "results.csv", index=False)
    summary = summarize(df)
    (out / "summary.md").write_text(
        summary.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
