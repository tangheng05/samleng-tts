# Samleng-TTS

Real-time, native-sounding Khmer text-to-speech that runs on everyday hardware, plus an open benchmark for Khmer TTS with human ratings. *Samleng* means "voice" in Khmer.

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

No voices are used without consent. Generated audio should be labeled as AI-generated. Each dataset and model's license is tracked in `docs/`.

## License

Code: MIT. Models and datasets carry their own licenses.
