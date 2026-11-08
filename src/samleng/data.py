"""Turn public Khmer speech corpora into clean training clips."""

import random
import re

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


_KHMER_GAP = re.compile(r"(?<=[ក-៿]) +(?=[ក-៿])")


def unsegment(text: str) -> str:
    """Drop word-segmentation spaces between Khmer words; TTS stretches at each one."""
    return _KHMER_GAP.sub("", text)


_EN_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*[A-Za-z]|[A-Za-z]")


def english_words(text: str) -> set[str]:
    return {w.lower() for w in _EN_WORD.findall(text)}


def mixed_split(items, seed: int = 0) -> dict:
    """Hold out half the bench's mixed sentences: their English words never go into training.

    After fine-tuning, `unseen` measures generalization to new English words, `seen`
    the gain on trained words; `partial` sentences mix both.
    """
    words = {it.id: english_words(it.text) for it in items if it.category == "mixed"}
    ids = sorted(words)
    random.Random(seed).shuffle(ids)
    held = ids[: len(ids) // 2]
    banned = set().union(*(words[i] for i in held))
    unseen = sorted(i for i, w in words.items() if w and w <= banned)
    seen = sorted(i for i, w in words.items() if not w & banned)
    partial = sorted(set(words) - set(unseen) - set(seen))
    return {
        "seed": seed,
        "banned": sorted(banned),
        "unseen": unseen,
        "seen": seen,
        "partial": partial,
    }


def distill_problems(text: str, leak: LeakCheck, banned: set[str]) -> list[str]:
    """Why a sentence can't be training text: bench overlap, or a held-out English word.

    Held-out words also count inside longer words ("poster" teaches "post").
    """
    words = english_words(text)
    problems = []
    if any(b in w for w in words for b in banned if w == b or len(b) > 3):
        problems.append("held-out word")
    if leak(text):
        problems.append("bench overlap")
    return problems


def distill_reject(
    row: dict,
    rate: tuple[float, float],
    min_spk: float = 0.9,
    max_cer: float = 0.3,
    max_cer_mixed: float = 0.6,
) -> str | None:
    """Why a teacher clip shouldn't be trained on, or None to keep it.

    Mixed clips get a looser CER limit: the ASR writes English in Khmer script.
    """
    if row["spk"] < min_spk:
        return "voice"
    limit = max_cer_mixed if row["source"] == "mixed_v1" else max_cer
    if row["cer"] > limit:
        return "misread"
    if not rate[0] <= row["cps"] <= rate[1]:
        return "speed"
    return None
