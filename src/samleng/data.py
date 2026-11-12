"""Turn public Khmer speech corpora into clean training clips."""

import random
import re
from collections import Counter
from collections.abc import Iterable

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


def mixed_group(item_id: str, category: str, split: dict) -> str:
    """`mixed` becomes mixed_unseen / mixed_seen / mixed_partial; other categories pass through."""
    if category != "mixed":
        return category
    for group in ("unseen", "seen", "partial"):
        if item_id in split[group]:
            return f"mixed_{group}"
    raise KeyError(f"{item_id} is not in the mixed split")


def has_held_out_word(text: str, banned: set[str]) -> bool:
    """True if text has a held-out English word, also inside a longer word ("poster" has "post")."""
    return any(
        b in w for w in english_words(text) for b in banned if w == b or len(b) > 3
    )


def distill_problems(text: str, leak: LeakCheck, banned: set[str]) -> list[str]:
    """Why a sentence can't be training text: bench overlap, or a held-out English word.

    Held-out words also count inside longer words ("poster" teaches "post").
    """
    problems = []
    if has_held_out_word(text, banned):
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
    limit = max_cer_mixed if row["source"].startswith("mixed") else max_cer
    if row["cer"] > limit:
        return "misread"
    if not rate[0] <= row["cps"] <= rate[1]:
        return "speed"
    return None


def both_scripts(row: dict, words: dict[str, str]) -> list[dict]:
    """A training row, plus a copy with its English in Khmer script when every word has a spelling.

    The same audio then teaches both inputs: dictionary words arrive in Khmer script at run
    time, unknown words in Latin letters. Abbreviations are spelled out in both.
    """
    from samleng.text import expand_acronyms, transliterate

    latin = expand_acronyms(row["text"])
    out = [row | {"text": latin}]
    khmer = transliterate(latin, words)
    if khmer != latin and not _EN_WORD.search(khmer):
        out.append(row | {"id": row["id"] + "-km", "text": khmer})
    return out


# sengtha's alphabet after its cleaner lowercases the text: Khmer block, a-z, its punctuation.
UNREADABLE = re.compile(r"[^\u1780-\u17ffa-z!,.?:;()\"'«»“”\-–—%/ ]")


def training_rows(
    rows: Iterable[dict], words: dict[str, str], banned: set[str]
) -> tuple[list[dict], Counter]:
    """Rows a student trains on, English in both scripts, and counts of what was left out.

    Left out: rows the checks rejected (by reason) and rows with a character the model
    cannot read. A held-out English word is an error: the bench split would be spoiled.
    """
    out, dropped = [], Counter()
    for r in rows:
        if "reject" in r:
            dropped[r["reject"]] += 1
            continue
        if has_held_out_word(r["text"], banned):
            raise ValueError(f"{r['id']}: held-out English word in training text")
        versions = both_scripts(r, words)
        if any(UNREADABLE.search(v["text"].lower()) for v in versions):
            dropped["unreadable"] += 1
            continue
        out += versions
    ids = Counter(r["id"] for r in out)
    if clash := [i for i, n in ids.items() if n > 1]:
        raise ValueError(f"duplicate ids: {clash[:5]}")
    return out, dropped


def khmer_script_leaks(spellings: dict[str, str], texts: Iterable[str]) -> set[str]:
    """Held-out English words that still occur in training text, written in Khmer script.

    `spellings` maps each held-out word to its Khmer spelling ("" if it has none). Common
    loanwords (Facebook, video) appear in ordinary Khmer text, so a sentence with them is not
    strictly unseen. A word with no spelling, or one too short to match safely, counts as leaked.
    """
    targets = {
        w: normalize_khmer(km, spell_numbers=False) for w, km in spellings.items()
    }
    leaked = {w for w, km in targets.items() if len(km) < 3}
    todo = {w: km for w, km in targets.items() if w not in leaked}
    for text in texts:
        if not todo:
            break
        t = normalize_khmer(text, spell_numbers=False)
        for w in [w for w, km in todo.items() if km in t]:
            leaked.add(w)
            del todo[w]
    return leaked


def strict_unseen(items, split: dict, leaked: set[str]) -> set[str]:
    """Ids of `unseen` mixed sentences none of whose English words leaked in Khmer script."""
    unseen = set(split["unseen"])
    return {
        it.id for it in items if it.id in unseen and not english_words(it.text) & leaked
    }
