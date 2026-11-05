# Khmer Voice

Real-time, native-sounding Khmer text-to-speech that runs on a Mac mini, plus an open benchmark for Khmer TTS with human ratings.

> Status: **Phase 0 — foundations + baseline.**

## Why

Open Khmer TTS has improved fast, but there's no shared way to measure it, and models still struggle with real-world text: numbers, dates, money, and Khmer–English code-switching. This project:

1. Builds **KhmerTTS-Bench**, a public benchmark (CER, speaker similarity, UTMOS, real-time factor, and native-speaker MOS).
2. Distills a large multilingual TTS model into a **small model that runs in real time on CPU / Apple Silicon**.
3. Ships it as a streaming API and demo.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```bash
brew install ffmpeg
uv sync
cp .env.example .env
uv run pytest
```

Development happens on a MacBook Pro M4; training and evals run on a Mac mini M4 (16GB). Set `KHMER_VOICE_DATA` / `KHMER_VOICE_OUTPUT` in `.env` to point at external storage on the training machine.

## Layout

```
src/khmer_voice/   library code (eval harness, data pipeline, models)
scripts/           runnable entry points
bench/             benchmark sentences and metadata
docs/              write-ups and results
tests/
```

## Built on

- [Tha](https://github.com/seanghay/tha) — Khmer text normalization
- [Qwen3-ASR-0.6B-Khmer](https://huggingface.co/seanghay/Qwen3-ASR-0.6B-Khmer) — Khmer ASR for intelligibility scoring
- [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — teacher model
- [MMS-TTS Khmer](https://huggingface.co/facebook/mms-tts-khm) — baseline

## Ethics

No voices are used without consent. Generated audio should be labeled as AI-generated. Each dataset and model's license is tracked in `docs/`.

## License

Code: MIT. Models and datasets carry their own licenses.
