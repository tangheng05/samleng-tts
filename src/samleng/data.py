"""Turn public Khmer speech corpora into clean training clips."""

import librosa
import numpy as np

from samleng.audio import normalize_loudness
from samleng.eval.cer import normalize_khmer

# sengtha, the Phase 2 base model, is a 22.05 kHz VITS.
TARGET_SR = 22050
PAD_S = 0.05


def prepare_audio(
    y: np.ndarray, sr: int, target_sr: int = TARGET_SR, top_db: float = 40.0
) -> np.ndarray:
    """Mono, resampled, edge silence trimmed (with a short pad), level normalized."""
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    _, (start, end) = librosa.effects.trim(y, top_db=top_db)
    pad = int(PAD_S * target_sr)
    y = y[max(0, start - pad) : end + pad]
    return normalize_loudness(y, target_sr)


class LeakCheck:
    """True if a training transcript shares a long span with any bench sentence.

    Bench text must never be trained on, or benchmark scores stop meaning anything.
    """

    def __init__(self, bench_texts: list[str], n: int = 20):
        self.n, self.short, self.grams = n, [], set()
        for text in bench_texts:
            s = normalize_khmer(text, spell_numbers=False)
            if len(s) <= n:
                self.short.append(s)
            else:
                self.grams |= {s[i : i + n] for i in range(len(s) - n + 1)}

    def __call__(self, text: str) -> bool:
        s = normalize_khmer(text, spell_numbers=False)
        if any(b in s for b in self.short):
            return True
        return any(s[i : i + self.n] in self.grams for i in range(len(s) - self.n + 1))


def parse_line_index(lines) -> list[tuple[str, str, str]]:
    """OpenSLR TTS index: `<speaker-prefixed id>\\t[\\t]<text>` → (id, speaker, text)."""
    rows = []
    for line in lines:
        cols = [c.strip() for c in line.split("\t") if c.strip()]
        if len(cols) >= 2:
            rows.append((cols[0], cols[0].split("_")[1], cols[-1]))
    return rows
