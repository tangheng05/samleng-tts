# Samleng-TTS

Real-time, native-sounding Khmer text-to-speech that runs on everyday hardware, plus an open benchmark for Khmer TTS with human ratings. *Samleng* means "voice" in Khmer.

> Status: **KhmerTTS-Bench v1 results in** (7 open models + Gemini). Now training a small real-time voice distilled from VoxCPM2.

Write-ups: [How good is open Khmer TTS in 2026?](docs/posts/01-how-good-is-open-khmer-tts.md) · [Every open Khmer voice, measured](docs/posts/02-every-open-khmer-tts-model.md)

## Why

Open Khmer TTS has improved fast, but there's no shared way to measure it, and models still struggle with real-world text: numbers, dates, money, and Khmer–English code-switching. This project:

1. Builds **KhmerTTS-Bench**, a public benchmark (CER, real-time factor, and native-speaker MOS).
2. Distills a large multilingual TTS model into a **small model that runs in real time on CPU / Apple Silicon**.
3. Ships it as a streaming API and demo.

## KhmerTTS-Bench v1 results (320 sentences)

[`bench/khmertts_bench_v1.jsonl`](bench/khmertts_bench_v1.jsonl): 80 pure Khmer, 40 long, 50 names, 60 numbers/dates/money, 70 Khmer–English mixed, 20 English, all reviewed by a native speaker. Text is fed to each model as written (numbers as digits). Also on Hugging Face: [hengthegoat/khmertts-bench-v1](https://huggingface.co/datasets/hengthegoat/khmertts-bench-v1) (CC-BY-4.0).

| | Pure Khmer | Long | Names | Numbers (digits) | All Khmer | English WER | Speed (RTF) | License |
|---|---|---|---|---|---|---|---|---|
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (2.3B) | 5.9% | 0.0% | 3.6% | 6.4% | 6.4% | 0% | 1.86 (MPS) | Apache-2.0 |
| [sengtha](https://huggingface.co/sengtha/khmer-tts-female-v2) (VITS) | 7.8% | 3.3% | 7.8% | 35.0% | 17.2% | 165% | **0.16 (CPU)** | CC-BY-SA-4.0 |
| Gemini TTS (`gemini-2.5-flash-preview-tts`) | 13.0% | 5.4% | 10.0% | 18.3% | 15.4% | 0% | 1.09 (API) | commercial |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | 10.6% | 6.6% | 7.8% | 35.4% | 17.3% | 0% | 1.43 (MPS) | CC-BY-NC-4.0 weights |
| [khmerttsopensource/khmer-tts](https://huggingface.co/khmerttsopensource/khmer-tts) (MMS fine-tune) | 22.2% | 16.6% | 26.0% | 48.4% | 31.5% | 100% | 0.08 (CPU) | CC-BY-NC-4.0 |
| [MMS-TTS Khmer](https://huggingface.co/facebook/mms-tts-khm) | 23.1% | 19.0% | 30.0% | 51.4% | 35.3% | 100% | 0.08 (CPU) | CC-BY-NC-4.0 |
| [KrorngAI/mms-tts-khm-finetuned](https://huggingface.co/KrorngAI/mms-tts-khm-finetuned) | 63.9% | 61.5% | 69.7% | 72.5% | 69.4% | 100% | 0.08 (CPU) | CC-BY-NC-4.0 |
| [Zio](https://huggingface.co/sombochea/zio-tts) | 70.4% | 60.2% | 58.6% | 70.5% | 69.9% | 179% | 0.17 (CPU) | CC-BY-SA-4.0 |

- Median CER per category (lower is better); English is scored by [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) word error rate (over 100% = the recognizer heard many extra words). "All Khmer" is every category except English, mixed included.
- Mixed sentences have no column: CER is not a valid metric there (see below). Human ratings of mixed sentences come in the next listening study.
- **Gemini returned no audio for 8 of 70 mixed sentences** (11%) after four attempts each, e.g. "WiFi នៅហាងកាហ្វេនេះលឿនណាស់។". It appears to treat some code-switched text as something to answer rather than read. These count as silence.
- With our text frontend (numbers spelled out), number errors mostly disappear; v1 results with the frontend will follow. Zio's own number normalizer is not used.

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
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | Large baseline and teacher model | Apache-2.0 |
| [sengtha/khmer-tts-female-v2](https://huggingface.co/sengtha/khmer-tts-female-v2) (trained on DDD-Cambodia) | Evaluated; base of our fine-tune | CC-BY-SA-4.0 |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), [Zio](https://huggingface.co/sombochea/zio-tts), [khmerttsopensource/khmer-tts](https://huggingface.co/khmerttsopensource/khmer-tts), [KrorngAI/mms-tts-khm-finetuned](https://huggingface.co/KrorngAI/mms-tts-khm-finetuned) | Evaluated | see table above |
| [OpenSLR 42](https://openslr.org/42/) (Google, Khmer) | Real-speech data pipeline, distillation text | CC-BY-SA-4.0 |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | ASR judge for English | MIT |
| [WavLM-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) | Speaker similarity | not stated on the model card |
| [CMUdict](https://github.com/cmusphinx/cmudict) | English → Khmer-script converter | BSD-2-Clause |
| [khmer-nltk](https://github.com/VietHoang1512/khmer-nltk) | Word segmentation for Zio | Apache-2.0 |
| [Coqui TTS](https://github.com/idiap/coqui-ai-TTS) | VITS fine-tuning | MPL-2.0 |
| [MMS-TTS Khmer](https://huggingface.co/facebook/mms-tts-khm) | Small baseline | CC-BY-NC-4.0 |
| [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | Naturalness predictor (evaluated, not used) | MIT |

## License

Code: MIT. Models and datasets carry their own licenses.
