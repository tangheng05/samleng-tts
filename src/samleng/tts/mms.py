import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

from samleng.bench import Item

MODEL_ID = "facebook/mms-tts-khm"


def synthesize(
    items: list[Item],
    out_dir: Path,
    seed: int = 0,
    model_id: str = MODEL_ID,
    revision: str | None = None,
) -> list[dict]:
    """Any transformers VITS model (MMS and its fine-tunes).

    Runs on CPU on purpose: it's the realistic baseline for cheap hardware.
    """
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = VitsModel.from_pretrained(model_id, revision=revision).eval()
    sr = model.config.sampling_rate
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        torch.manual_seed(seed)  # VITS samples durations; fix it for reproducibility
        t = time.time()
        with torch.no_grad():
            wav = model(**tok(it.text, return_tensors="pt")).waveform[0].numpy()
        gen = time.time() - t
        path = out_dir / f"{it.id}.wav"
        sf.write(path, wav, sr)
        rows.append(
            {
                "id": it.id,
                "audio": str(path),
                "seconds": len(wav) / sr,
                "gen_seconds": gen,
            }
        )
    return rows
