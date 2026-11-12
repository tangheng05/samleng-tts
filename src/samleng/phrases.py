"""Phrase spaces for long unspaced Khmer text.

Khmer writers put spaces between phrases, not words, and often leave them out. The TTS
models read a space as a short pause, and long runs without one get rushed. `phrase_spaces`
splits such runs at the word boundaries where writers most often put a space (weights learned
from Khmer Wikipedia by scripts/phrase_breaks.py). Existing spaces are never touched.
"""

import functools
import json
import math
import re

from khmercut import tokenize

from samleng.paths import ROOT

MODEL = ROOT / "texts" / "phrase_breaks.json"
# Khmer letters, vowels and signs; no punctuation (។ ៕ ៖) and no digits.
RUN = re.compile(r"[\u1780-\u17d3\u17d7\u17dc\u17dd]+")
# A run this long gets split; a split never leaves a piece shorter than MIN_PIECE (characters).
MAX_RUN, MIN_PIECE = 40, 8
# Below this log-odds a boundary is not a believable phrase break: the run stays whole.
MIN_SCORE = -0.5
# Spelled-out numbers are one phrase ("មួយរយ ហុកសិបប្រាំបី" must not be split).
NUMBER = re.compile(
    "(?:សូន្យ|មួយ|ពីរ|បី|បួន|ប្រាំ|ដប់|ម្ភៃ|សាមសិប|សែសិប|ហាសិប|ហុកសិប|ចិតសិប|ប៉ែតសិប|កៅសិប"
    "|រយ|ពាន់|ម៉ឺន|សែន|លាន|កោដិ)+"
)
START, END, OTHER = "<s>", "</s>", "<other>"


def words(run: str) -> list[str]:
    """Words of one unspaced Khmer run (khmercut CRF); the run itself if segmentation fails."""
    out = [w for w in tokenize(run) if w]
    return out if "".join(out) == run else [run]


@functools.cache
def _model() -> dict:
    return json.loads(MODEL.read_text(encoding="utf-8"))


def context(ws: list[str], i: int) -> dict[str, str]:
    """The words around the boundary before ws[i]."""
    return {
        "before2": ws[i - 2] if i >= 2 else START,
        "before": ws[i - 1],
        "after": ws[i],
        "after2": ws[i + 1] if i + 1 < len(ws) else END,
    }


def break_scores(ws: list[str], model: dict) -> list[float]:
    """Log-odds of a phrase space before ws[i], for i = 1..len(ws)-1."""
    scores = []
    for i in range(1, len(ws)):
        if NUMBER.fullmatch(ws[i - 1]) and NUMBER.fullmatch(ws[i]):
            scores.append(-math.inf)
            continue
        s = model["bias"]
        for name, word in context(ws, i).items():
            table = model["weights"][name]
            s += table.get(word, table[OTHER])
        scores.append(s)
    return scores


def split(
    ws: list[str],
    scores: list[float],
    max_run: int = MAX_RUN,
    min_piece: int = MIN_PIECE,
    min_score: float = MIN_SCORE,
) -> list[str]:
    """Split at the best-scoring boundary until every piece fits in max_run."""
    total = sum(map(len, ws))
    if total <= max_run:
        return ["".join(ws)]
    best, left = None, 0
    for i in range(1, len(ws)):
        left += len(ws[i - 1])
        if left < min_piece or total - left < min_piece:
            continue
        # -inf marks a forbidden break (inside a number), whatever the floor is.
        if scores[i - 1] < min_score or scores[i - 1] == -math.inf:
            continue
        if best is None or scores[i - 1] > scores[best - 1]:
            best = i
    if best is None:
        return ["".join(ws)]
    rest = (max_run, min_piece, min_score)
    return split(ws[:best], scores[: best - 1], *rest) + split(
        ws[best:], scores[best:], *rest
    )


def phrase_spaces(
    text: str, max_run: int = MAX_RUN, min_score: float = MIN_SCORE
) -> str:
    """Add spaces inside Khmer runs longer than max_run characters.

    A run with no believable break (min_score) stays whole, even if long.
    """

    def spaced(m: re.Match) -> str:
        if len(m[0]) <= max_run:
            return m[0]
        ws = words(m[0])
        scores = break_scores(ws, _model())
        return " ".join(split(ws, scores, max_run, min_score=min_score))

    return RUN.sub(spaced, text)
