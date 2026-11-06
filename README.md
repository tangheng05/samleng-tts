# Samleng-TTS

Real-time, native-sounding Khmer text-to-speech that runs on everyday hardware, plus an open benchmark for Khmer TTS with human ratings. *Samleng* means "voice" in Khmer.

> Status: **Phase 0 — baseline results in.** Next: a larger benchmark (KhmerTTS-Bench v1) and a speech data pipeline.

## Why

Open Khmer TTS has improved fast, but there's no shared way to measure it, and models still struggle with real-world text: numbers, dates, money, and Khmer–English code-switching. This project:

1. Builds **KhmerTTS-Bench**, a public benchmark (CER, real-time factor, and native-speaker MOS).
2. Distills a large multilingual TTS model into a **small model that runs in real time on CPU / Apple Silicon**.
3. Ships it as a streaming API and demo.

## Baseline results (pilot, 100 sentences)

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

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```bash
brew install ffmpeg
uv sync
cp .env.example .env
uv run pytest
```

Set `SAMLENG_DATA` / `SAMLENG_OUTPUT` in `.env` to keep data and outputs somewhere else, such as an external drive.

## Human ratings (MOS)

`uv run python scripts/mos.py serve` opens a local page for blind 1–5 listening scores; `uv run python scripts/mos.py report` gives MOS per system (CI over sentences) and paired Wilcoxon tests.

## Layout

```
src/samleng/       library code (eval harness, data pipeline, models)
scripts/           runnable entry points
bench/             benchmark sentences and metadata
docs/              write-ups and results
runners/           isolated envs for models with conflicting deps
tests/
```

## Ethics

No voices are used without consent. Generated audio should be labeled as AI-generated.

## Acknowledgements

| Project | Used for | License |
|---|---|---|
| [Tha](https://github.com/seanghay/tha) | Khmer text normalization | Apache-2.0 |
| [Qwen3-ASR-0.6B-Khmer](https://huggingface.co/seanghay/Qwen3-ASR-0.6B-Khmer) | ASR judge for CER | Apache-2.0 |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | Large baseline (future teacher model) | Apache-2.0 |
| [MMS-TTS Khmer](https://huggingface.co/facebook/mms-tts-khm) | Small baseline | CC-BY-NC-4.0 |
| [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | Naturalness predictor (evaluated, not used) | MIT |

## License

Code: MIT. Models and datasets carry their own licenses.
