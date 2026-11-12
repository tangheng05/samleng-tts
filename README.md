# Samleng-TTS

Real-time, native-sounding Khmer text-to-speech that runs on everyday hardware, plus an open benchmark for Khmer TTS with human ratings.

> **Status (November 2026):** benchmark published; a small voice distilled from a 2B teacher is training. First results below are preliminary.

Write-ups: [How good is open Khmer TTS in 2026?](docs/posts/01-how-good-is-open-khmer-tts.md) · [Every open Khmer voice, measured](docs/posts/02-every-open-khmer-tts-model.md)

## What is here

1. **KhmerTTS-Bench v1**: 320 native-reviewed sentences (pure Khmer, long, names, numbers, Khmer–English mixed, English) with fixed scoring code. Also on Hugging Face: [hengthegoat/khmertts-bench-v1](https://huggingface.co/datasets/hengthegoat/khmertts-bench-v1).
2. **A leaderboard** of every open Khmer TTS model we could run, plus Gemini TTS as a commercial reference.
3. **A distillation pipeline**: a large teacher ([VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), 2B) reads thousands of sentences in one fixed voice; a small VITS student (about 36M parameters at inference, 10× faster than real time on a CPU) learns from that audio. See [docs/pipeline.md](docs/pipeline.md).
4. **A Khmer text front end**: numbers, dates, times, money, abbreviations and English words turned into what a Khmer speaker would say.
5. **Listening-test tools**: blind 1–5 ratings and paired A/B comparisons with proper statistics.

## Why

Open Khmer TTS has improved fast, but there was no shared way to measure it, and models still struggle with real-world text: numbers, dates, money and Khmer–English code-switching. Automatic scores also miss what native listeners hear, so this project pairs every automatic number with human ratings.

## KhmerTTS-Bench v1 results (320 sentences)

[`bench/khmertts_bench_v1.jsonl`](bench/khmertts_bench_v1.jsonl): 80 pure Khmer, 40 long, 50 names, 60 numbers/dates/money, 70 Khmer–English mixed, 20 English, all reviewed by a native speaker. Text is fed to each model as written (numbers as digits). Also on Hugging Face: [hengthegoat/khmertts-bench-v1](https://huggingface.co/datasets/hengthegoat/khmertts-bench-v1) (CC-BY-4.0).

| | Pure Khmer | Long | Names | Numbers (digits) | All Khmer | English WER | Speed (RTF) | License |
|---|---|---|---|---|---|---|---|---|
| Gemini 3.8 TTS (`gemini-3.8-flash-tts`) | 0.0% | 0.0% | 2.0% | 0.0% | 2.2% | 0% | 0.84 (API) | commercial |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (2.3B) | 5.9% | 0.0% | 3.6% | 6.4% | 6.4% | 0% | 1.86 (MPS) | Apache-2.0 |
| [sengtha](https://huggingface.co/sengtha/khmer-tts-female-v2) (VITS) | 7.8% | 3.3% | 7.8% | 35.0% | 17.2% | 165% | **0.16 (CPU)** | CC-BY-SA-4.0 |
| Gemini 2.5 TTS (`gemini-2.5-flash-preview-tts`) | 13.0% | 5.4% | 10.0% | 18.3% | 15.4% | 0% | 1.09 (API) | commercial |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | 10.6% | 6.6% | 7.8% | 35.4% | 17.3% | 0% | 1.43 (MPS) | CC-BY-NC-4.0 weights |
| [khmerttsopensource/khmer-tts](https://huggingface.co/khmerttsopensource/khmer-tts) (MMS fine-tune) | 22.2% | 16.6% | 26.0% | 48.4% | 31.5% | 100% | 0.08 (CPU) | CC-BY-NC-4.0 |
| [MMS-TTS Khmer](https://huggingface.co/facebook/mms-tts-khm) | 23.1% | 19.0% | 30.0% | 51.4% | 35.3% | 100% | 0.08 (CPU) | CC-BY-NC-4.0 |
| [KrorngAI/mms-tts-khm-finetuned](https://huggingface.co/KrorngAI/mms-tts-khm-finetuned) | 63.9% | 61.5% | 69.7% | 72.5% | 69.4% | 100% | 0.08 (CPU) | CC-BY-NC-4.0 |
| [Zio](https://huggingface.co/sombochea/zio-tts) | 70.4% | 60.2% | 58.6% | 70.5% | 69.9% | 179% | 0.17 (CPU) | CC-BY-SA-4.0 |

- Median CER per category (lower is better); English is scored by [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) word error rate (over 100% = the recognizer heard many extra words). "All Khmer" is every category except English, mixed included.
- Mixed sentences have no column: CER is not a valid metric there (see below). Human ratings of mixed sentences come in the next listening study.
- **Gemini 2.5 returned no audio for 8 of 70 mixed sentences** (11%) after four attempts each, e.g. "WiFi នៅហាងកាហ្វេនេះលឿនណាស់។". It appears to treat some code-switched text as something to answer rather than read. These count as silence.
- **Gemini 3.8 read all 320 sentences** and is the most accurate system here. It reads a style prefix such as "Say in a warm, friendly voice:" out loud, so it gets the bare text (same voice, Callirrhoe). Gemini 2.5 needs that prefix to read rather than answer.
- With our text frontend (numbers spelled out), number errors mostly disappear; v1 results with the frontend will follow. Zio's own number normalizer is not used.

## Distillation results so far (preliminary)

A first student trained on only 1.2 hours of teacher audio. Scores are on the 300 Khmer sentences of the benchmark with the text front end (numbers spelled out); lower CER is better. MOS is a blind 1–5 rating of 25 sentences by **one** native speaker (the author), so read it as a signal, not a result.

| | All-Khmer CER (median) | Mixed sentences with never-seen English words | MOS (1 rater) |
|---|---|---|---|
| Base model, before our training | 10.4% | 36.7% | 2.16 ± 0.37 |
| Student A, 1.2 h of teacher audio | 5.7% (3 seeds) | 23.7% | 3.40 ± 0.29 |
| Student B, same audio, English written in Khmer script | 6.1% | 13.2% | not rated |
| Teacher (VoxCPM2, fixed voice) | 3.1% | 12.9% | 3.72 ± 0.25 |

- Writing known English words in Khmer script before synthesis helps more than training on Latin letters: it alone takes the base model from 36.7% to 16.9% on those sentences.
- Student A reads English in Latin letters, student B in Khmer script. The teacher's error rate is for the new voice used in the next run; its rating is for the same voice as the students.
- The same checkpoint scored with three random seeds moves about 1 point, so smaller differences are not reported as real.
- With Khmer-script input, "never-seen" means the Latin form never appeared in training text; a few common loanwords (Facebook, video …) do occur in Khmer script in the training sentences.
- A larger run (about 8.7 hours of teacher audio in a new voice, English in both scripts) is training now. Multi-rater listening tests follow.

## Pilot results (Phase 0, 100 sentences)

[`bench/pilot_v1.jsonl`](bench/pilot_v1.jsonl): 40 pure Khmer, 10 long, 25 numbers/dates/money, 25 Khmer–English mixed, all reviewed by a native speaker. All rows except "raw digits" use the text frontend.

| | MMS-khm (36M) | VoxCPM2 (2.3B) |
|---|---|---|
| Pure Khmer, CER median (corpus) | 23.1% (28.3%) | 2.7% (8.0%) |
| Long sentences | 20.0% (19.9%) | 0.5% (1.5%) |
| Numbers, raw digits | 53.8% (60.8%) | 8.0% (13.1%) |
| Numbers, with our text frontend | 26.9% (29.0%) | 0.0% (4.3%) |
| Mixed Khmer–English, **MOS** (1–5) | 1.56 ± 0.21 | 3.72 ± 0.25 |
| Speed, median real-time factor (middle 50%) | 0.08 (0.08–0.09) | 2.1 (1.9–2.9) |

- **CER** (character error rate) comes from transcribing the audio with [Qwen3-ASR-0.6B-Khmer](https://huggingface.co/seanghay/Qwen3-ASR-0.6B-Khmer). Numbers are spelled out, and spaces and punctuation are removed before scoring.
- **Text frontend:** numbers, dates, times and money are spelled out in Khmer words before synthesis ([Tha](https://github.com/seanghay/tha) plus our fixes for thousands separators, °C and 24-hour times).
- **Mixed sentences use human ratings, not CER.** The ASR spells English words inconsistently, so CER doesn't track quality there (Spearman −0.06 between CER and human scores for VoxCPM2). MOS is a blind 1–5 rating with ITU-T P.800 labels by one native speaker (the author), with 25 sentences, a 95% CI over sentences, and loudness-normalized clips. VoxCPM2 scored higher on all 25 sentences (paired Wilcoxon p < 0.001).
- An automatic naturalness predictor (UTMOSv2) did not agree with the human ratings on 30 rated mixed clips (Spearman 0.25) and couldn't separate the two systems, so it is not reported.
- **Real-time factor** is generation time ÷ audio length (below 1 = faster than real time). MMS runs on CPU, VoxCPM2 on Apple Silicon (MPS) with 16 GB of memory. We report the median because a few VoxCPM2 clips are much slower when memory is tight.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` (`brew install ffmpeg`).

```bash
uv sync                      # main environment (Python 3.11)
cp .env.example .env         # optional: data/output folders, API keys
uv run pytest -q             # unit tests, no model downloads

# Score open models on the benchmark (downloads models on first use)
uv run python scripts/baseline.py bench/khmertts_bench_v1.jsonl --systems mms mms_kto --verbalize

# Blind listening test on the clips of two systems
uv run python scripts/mos.py serve --study demo --bench bench/khmertts_bench_v1.jsonl --category pure \
    --system mms=baseline/khmertts_bench_v1_verbalized/mms --system mms_kto=baseline/khmertts_bench_v1_verbalized/mms_kto
uv run python scripts/mos.py report --study demo
```

The text front end on its own:

```python
from samleng.text import khmer_script, verbalize

verbalize(
    "ថ្ងៃទី 14/04/2026 ម៉ោង 7:30 ព្រឹក"
)  # ថ្ងៃទី ដប់បួន ខែមេសា ឆ្នាំ ពីរពាន់ម្ភៃប្រាំមួយ ម៉ោង ប្រាំពីរកន្លះ ព្រឹក
khmer_script("សូមផ្ញើ file តាម USB")  # សូមផ្ញើ ហ្វាយ តាម យូ អេស ប៊ី
```

Models that need conflicting dependencies (the teacher, the ASR judge, the VITS trainer) live in isolated environments under [`runners/`](runners/README.md) and are called through JSONL files.

## Repository layout

| Path | What it holds |
|---|---|
| [`src/samleng/`](src/samleng) | The library: text front end (`text`, `en2km`, `phrases`), scoring (`eval/`), data checks (`data`), listening tests (`mos`, `rating_server`), baselines (`tts/`) |
| [`scripts/`](scripts) | Command-line entry points: benchmark runs, teacher-data checks, training manifest, checkpoint scoring, rating pages ([index](docs/pipeline.md#scripts)) |
| [`runners/`](runners/README.md) | One isolated environment per model: teacher, judge, student training, other TTS systems |
| [`bench/`](bench/README.md) | Benchmark sentences and the held-out split for English words |
| [`texts/`](texts/README.md) | Sentences the teacher reads, approved English spellings, pronunciation fixes |
| [`ops/`](ops/README.md) | Restart-safe job queue for an always-on Mac, checkpoint watcher, lab notebook |
| [`docs/`](docs) | Pipeline guide and blog posts |
| [`tests/`](tests) | Unit tests (run in CI on macOS with ruff) |

## How results stay trustworthy

- **Held-out test set.** No benchmark sentence, and none of 64 held-out English words, may appear in training text; the builders refuse them.
- **Pinned models and seeds.** The teacher, the judge and every evaluated model load at a fixed revision; synthesis is seeded; lock files pin all dependencies.
- **Paired statistics.** Systems are compared per sentence (Wilcoxon signed-rank); several seeds before a difference is called real.
- **Guardrails next to error rates.** Each scored checkpoint also reports voice similarity, speaking pace, silence and clipping, so a lower error rate cannot hide a drifting voice.
- **Ears over metrics.** A native speaker checks samples of every batch of teacher audio; an automatic naturalness predictor (UTMOSv2) was tested and dropped because it did not track Khmer ratings.
- **A lab notebook.** [`ops/notebook.jsonl`](ops/notebook.jsonl) has one line per finished experiment: question, setup, result, verdict.

## Limitations

- Human ratings so far come from a single rater. A study with 10+ native listeners is planned.
- The student's voice is synthetic (designed with the teacher), and the student cannot be better than its teacher.
- The ASR judge cannot hear some errors (vowel-series mistakes, a swallowed syllable); those are caught by listening only.
- English words outside the approved spelling list are read from Latin letters and are less reliable.

## Roadmap

1. Finish the 8.7-hour run and release the model (`CC-BY-SA-4.0`) with a streaming server and a demo.
2. Multi-rater A/B study against Gemini TTS; a challenge set for errors that CER cannot hear.
3. A better teacher (fine-tuned on real Khmer speech) and, if ratings call for it, a stronger 24 kHz student.
4. A technical report.

## Ethics

No voices are used without consent; the current voice is synthetic. Generated audio should be labeled as AI-generated.

## Acknowledgements

| Project | Used for | License |
|---|---|---|
| [Tha](https://github.com/seanghay/tha) | Khmer text normalization | Apache-2.0 |
| [Qwen3-ASR-0.6B-Khmer](https://huggingface.co/seanghay/Qwen3-ASR-0.6B-Khmer) | ASR judge for CER | Apache-2.0 |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | Large baseline and teacher model | Apache-2.0 |
| [sengtha/khmer-tts-female-v2](https://huggingface.co/sengtha/khmer-tts-female-v2) (trained on DDD-Cambodia) | Evaluated; base of our fine-tune | CC-BY-SA-4.0 |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), [Zio](https://huggingface.co/sombochea/zio-tts), [khmerttsopensource/khmer-tts](https://huggingface.co/khmerttsopensource/khmer-tts), [KrorngAI/mms-tts-khm-finetuned](https://huggingface.co/KrorngAI/mms-tts-khm-finetuned) | Evaluated | see table above |
| [OpenSLR 42](https://openslr.org/42/) (Google, Khmer) | Real-speech data pipeline, distillation text | CC-BY-SA-4.0 |
| [Khmer Wikipedia](https://km.wikipedia.org) via [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.km) | Distillation text (`texts/distill_wiki_v3.txt`) | CC-BY-SA-3.0 / GFDL |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | ASR judge for English | MIT |
| [WavLM-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) | Speaker similarity | not stated on the model card |
| [CMUdict](https://github.com/cmusphinx/cmudict) | English → Khmer-script converter | BSD-2-Clause |
| [khmer-nltk](https://github.com/VietHoang1512/khmer-nltk) | Word segmentation for Zio | Apache-2.0 |
| [khmercut](https://github.com/seanghay/khmercut) | Word segmentation for phrase spacing | Apache-2.0 |
| [Coqui TTS](https://github.com/idiap/coqui-ai-TTS) | VITS fine-tuning | MPL-2.0 |
| [MMS-TTS Khmer](https://huggingface.co/facebook/mms-tts-khm) | Small baseline | CC-BY-NC-4.0 |
| [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | Naturalness predictor (evaluated, not used) | MIT |

## License

Code: MIT. Benchmark and our own training sentences and spelling lists: CC-BY-4.0 (details in [`texts/README.md`](texts/README.md); the Wikipedia sentences stay CC-BY-SA-3.0). Models and third-party datasets carry their own licenses (see the tables above); our released voice will be CC-BY-SA-4.0, the license of its base model.

## Citation

```bibtex
@misc{samleng2026,
  title  = {Samleng-TTS: an open benchmark and a distilled real-time voice for Khmer},
  author = {Tang, Heng},
  year   = {2026},
  url    = {https://github.com/tangheng05/samleng-tts}
}
```
