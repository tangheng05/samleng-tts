# Benchmark files

One JSON object per line: `{"id", "category", "text", "reviewed"}`. Every sentence was written for this project and reviewed by a native Khmer speaker. Load with `samleng.bench.load_bench`.

| File | Sentences | Notes |
|---|---|---|
| `khmertts_bench_v1.jsonl` | 320 | **The benchmark.** pure 80, numbers 60, mixed 70, names 50, long 40, english 20. Frozen: never edit a row; add a new version instead. Also on [Hugging Face](https://huggingface.co/datasets/hengthegoat/khmertts-bench-v1) (CC-BY-4.0). |
| `khmertts_bench_v1_mixed_split.json` | | Held-out split of the 70 mixed sentences (seed 0): 64 `banned` English words never appear in training text, so the 36 `unseen` sentences measure generalization to new English words; 28 `seen`, 6 `partial`. Built by `samleng.data.mixed_split`. |
| `pilot_v1.jsonl` | 100 | Phase 0 pilot (pure 40, numbers 25, mixed 25, long 10); its rows are part of v1. Frozen. |
| `pilot_v0.jsonl` | 40 | First pilot; frozen so the Phase 0 numbers stay reproducible. |

## Categories

| Category | What it tests |
|---|---|
| `pure` | everyday Khmer: statements, questions, requests, colloquial particles |
| `long` | sentences of several clauses |
| `names` | people, provinces, landmarks, foreign names |
| `numbers` | prices, dates, times, phone numbers, percentages, fractions, scores, Khmer digits |
| `mixed` | Khmer sentences with English words (code-switching) |
| `english` | English-only sentences (scored by Whisper word error rate) |

## Scoring

- Khmer categories: character error rate against the transcript of [Qwen3-ASR-0.6B-Khmer](https://huggingface.co/seanghay/Qwen3-ASR-0.6B-Khmer), after spelling out numbers and removing spaces and punctuation (`samleng.eval.cer`).
- `mixed`: `mixed_cer` writes known English words in Khmer script on both sides, because the recognizer spells English inconsistently. It tracks human ratings (Spearman −0.74) where plain CER does not (−0.44).
- Report the **median** per-sentence error with the count of recognizer loops, never the corpus error alone.

## Keeping it clean

Training text is checked against these files: a sentence sharing a 20-character span with any benchmark sentence, or containing a held-out English word, is refused (`samleng.data.LeakCheck`, `samleng.data.distill_problems`).
