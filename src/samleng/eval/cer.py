import string
import unicodedata

import jiwer
from rapidfuzz import fuzz

from samleng.text import english_spans, transliterate, verbalize

# Khmer khan, bariyoosan, colon and other sentence marks; ៗ (repetition) is kept.
_KHMER_PUNCT = "។៕៖៘៙៚"
_DROP = set(_KHMER_PUNCT + string.punctuation + "«»“”‘’…")


def normalize_khmer(text: str, spell_numbers: bool = True) -> str:
    # Subscript ដ and ត look and sound the same; the ASR always writes ្ត.
    text = unicodedata.normalize("NFC", text).replace("្ដ", "្ត")
    if spell_numbers:
        text = verbalize(text)
    return "".join(
        ch
        for ch in text
        if ch not in _DROP and not ch.isspace() and unicodedata.category(ch) != "Cf"
    )


def khmer_cer(reference: str | list[str], hypothesis: str | list[str]) -> float:
    refs = [reference] if isinstance(reference, str) else reference
    hyps = [hypothesis] if isinstance(hypothesis, str) else hypothesis
    return jiwer.cer(
        [normalize_khmer(r) for r in refs], [normalize_khmer(h) for h in hyps]
    )


def mixed_cer(reference: str | list[str], hypothesis: str | list[str]) -> float:
    """CER with known English words written in Khmer script on both sides.

    The ASR writes English sometimes in Latin, sometimes in Khmer script; this counts both.
    """
    refs = [reference] if isinstance(reference, str) else reference
    hyps = [hypothesis] if isinstance(hypothesis, str) else hypothesis
    # Lowercase after normalizing: verbalize needs "°C" intact.
    return jiwer.cer(
        [normalize_khmer(transliterate(r)).lower() for r in refs],
        [normalize_khmer(transliterate(h)).lower() for h in hyps],
    )


# Near-miss spellings count for longer words; short ones must match exactly.
_HIT_RATIO, _HIT_MIN_LEN = 70, 5


def english_hits(reference: str, hypothesis: str) -> tuple[int, int]:
    """(English words heard, English words in reference), matched in Khmer script."""
    hyp = normalize_khmer(transliterate(hypothesis)).lower()
    spans = english_spans(reference)
    hits = 0
    for _, khmer in spans:
        target = normalize_khmer(khmer, spell_numbers=False)
        if target in hyp or (
            len(target) >= _HIT_MIN_LEN
            and fuzz.partial_ratio(target, hyp) >= _HIT_RATIO
        ):
            hits += 1
    return hits, len(spans)
