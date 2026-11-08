"""Score the `english` bench category: Whisper transcribes, WER compares.

Our Khmer judge (Qwen3-ASR-Khmer) can't be trusted on English, so English
sentences get their own ASR and the standard Whisper text normalizer.
"""

from pathlib import Path

import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

MODEL_ID = "openai/whisper-large-v3-turbo"
REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

_normalize = EnglishTextNormalizer({})


def english_wer(reference: str | list[str], hypothesis: str | list[str]) -> float:
    refs = [reference] if isinstance(reference, str) else reference
    hyps = [hypothesis] if isinstance(hypothesis, str) else hypothesis
    return jiwer.wer([_normalize(r) for r in refs], [_normalize(h) for h in hyps])


def transcribe_english(paths: list[str | Path]) -> list[str]:
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    asr = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        revision=REVISION,
        device=device,
        dtype=torch.float16 if device == "mps" else torch.float32,
    )
    out = asr([str(p) for p in paths], generate_kwargs={"language": "en"}, batch_size=8)
    return [o["text"].strip() for o in out]
