# Runners

Each model with conflicting dependencies gets its own uv project here, so the main `samleng` package never imports them.

Contract: a runner reads a JSONL manifest and writes a JSONL of results. The main package only reads and writes files.

| Runner | Model | Pins |
|---|---|---|
| `asr_qwen` | Qwen3-ASR-0.6B-Khmer | `transformers==4.57.6` |
| `tts_voxcpm` | VoxCPM2 | `datasets<4`, Python <3.13 |
| `mos_utmos` | UTMOSv2 (predicted naturalness) | git commit `cc2700d`, weights at a fixed HF revision |
| `tts_omnivoice` | OmniVoice (auto voice) | `omnivoice==0.2.1`, weights at a fixed HF revision |
| `tts_onnx` | sengtha and Zio VITS voices (ONNX, CPU) | `khmer-nltk==1.6` (Zio's word segmenter), weights at fixed HF revisions |
| `train_vits` | Fine-tune and run sengtha's Coqui VITS checkpoint (CPU: the trainer has no MPS) | `coqui-tts[codec]==0.28.0`, `torch==2.15.0`, `torchaudio==2.11.1`, `six` (coqui-tts doesn't declare these), FFmpeg (for torchcodec) |

```bash
cd runners/asr_qwen
uv run python transcribe.py manifest.jsonl out.jsonl
```
