import string
import unicodedata

import jiwer

from samleng.text import verbalize

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
