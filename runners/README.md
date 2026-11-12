# Runners

Each model with conflicting dependencies gets its own uv project here, so the main `samleng` package never imports them.

Contract: a runner reads a JSONL manifest and writes a JSONL of results. The main package only reads and writes files.

| Runner | Model | Pins |
|---|---|---|
| `asr_qwen` | Qwen3-ASR-0.6B-Khmer, the Khmer judge | `transformers==4.57.6`, weights at a fixed HF revision |
| `tts_voxcpm` | VoxCPM2, the teacher (voice cloning, voice design, respelling hints, `--resume`) | `datasets<4`, Python <3.13, weights at a fixed HF revision |
| `mos_utmos` | UTMOSv2 (predicted naturalness) | git commit `cc2700d`, weights at a fixed HF revision |
| `tts_omnivoice` | OmniVoice (auto voice) | `omnivoice==0.2.1`, weights at a fixed HF revision |
| `tts_onnx` | sengtha and Zio VITS voices (ONNX, CPU) | `khmer-nltk==1.6` (Zio's word segmenter), weights at fixed HF revisions |
| `train_vits` | Fine-tune (`train.py`, CPU or Apple GPU), run (`synthesize.py`), average (`average.py`) sengtha's Coqui VITS checkpoint | `coqui-tts[codec]==0.28.0`, `torch==2.15.0`, `torchaudio==2.11.1`, `six` (coqui-tts doesn't declare these), FFmpeg (for torchcodec) |

```bash
cd runners/asr_qwen
uv run python transcribe.py manifest.jsonl out.jsonl
```

Every script takes a manifest (`{"id", "text"}` or `{"id", "audio"}` per line) and writes
`results.jsonl` or an output JSONL. From the main package, call them with
`samleng.runners.run("asr_qwen", "transcribe.py", manifest, out)`.

Models are loaded at pinned revisions, so a silent upstream update cannot change the teacher
or the judge between experiments.
