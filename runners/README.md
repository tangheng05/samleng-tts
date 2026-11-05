# Runners

Each model with conflicting dependencies gets its own uv project here, so the main `samleng` package never imports them.

Contract: a runner reads a JSONL manifest and writes a JSONL of results. The main package only reads and writes files.

| Runner | Model | Pins |
|---|---|---|
| `asr_qwen` | Qwen3-ASR-0.6B-Khmer | `transformers==4.57.6` |
| `tts_voxcpm` | VoxCPM2 | `datasets<4`, Python <3.13 |

```bash
cd runners/asr_qwen
uv run python transcribe.py manifest.jsonl out.jsonl
```
