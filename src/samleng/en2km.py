"""English words → Khmer script, the way Khmer speakers spell English loanwords.

Pronunciation comes from CMUdict (BSD-style license, fetched by `fetch_cmudict`).
The core rule: a Khmer vowel sign sounds different after an a-series and an o-series
consonant, so each syllable first picks its vowel, then the consonant letter (or the
៉/៊ series marks) whose series makes that vowel sound right.
"""

import hashlib
import urllib.request
from pathlib import Path

from samleng.paths import DATA_DIR

CMUDICT_COMMIT = "74790861f652b15e4ac49015a90074ad62a27690"
CMUDICT_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
CMUDICT_PATH = DATA_DIR / "cmudict" / "cmudict.dict"

A, O = 0, 1  # consonant series  # noqa: E741

# Onset consonants: (a-series letter, o-series letter).
ONSET = {
    "P": ("ផ", "ព"),
    "B": ("ប", "ប៊"),
    "T": ("ថ", "ធ"),
    "D": ("ដ", "ឌ"),
    "K": ("ខ", "ឃ"),
    "G": ("ហ្គ", "គ"),
    "CH": ("ឆ", "ឈ"),
    "JH": ("ជ៉", "ជ"),
    "F": ("ហ្វ", "ហ្វ៊"),
    "V": ("វ៉", "វ"),
    "W": ("វ៉", "វ"),
    "TH": ("ថ", "ធ"),
    "DH": ("ដ", "ឌ"),
    "S": ("ស", "ស៊"),
    "SH": ("ស", "ស៊"),
    "Z": ("ហ្ស", "ហ្ស៊"),
    "ZH": ("ហ្ស", "ហ្ស៊"),
    "HH": ("ហ", "ហ៊"),
    "M": ("ម៉", "ម"),
    "N": ("ណ", "ន"),
    "NG": ("ង៉", "ង"),
    "L": ("ឡ", "ល"),
    "R": ("រ៉", "រ"),
    "Y": ("យ៉", "យ"),
}
# After s, or as the first letter of a cluster, stops are unaspirated.
PLAIN = {"P": ("ប", "ព"), "T": ("ត", "ទ"), "K": ("ក", "គ"), "G": ("ក", "គ")}
# Subscript forms inside a cluster (s-t → ស្ត, p-r → ព្រ).
SUB = {
    "R": "្រ",
    "L": "្ល",
    "W": "្វ",
    "Y": "្យ",
    "T": "្ត",
    "P": "្ព",
    "K": "្ក",
    "M": "្ម",
    "N": "្ន",
}
CLUSTERS = {
    ("S", "T"),
    ("S", "P"),
    ("S", "K"),
    ("S", "M"),
    ("S", "N"),
    ("S", "W"),
    ("S", "L"),
    ("P", "R"),
    ("B", "R"),
    ("T", "R"),
    ("D", "R"),
    ("K", "R"),
    ("G", "R"),
    ("F", "R"),
    ("P", "L"),
    ("B", "L"),
    ("K", "L"),
    ("G", "L"),
    ("F", "L"),
    ("S", "L"),
    ("K", "W"),
    ("T", "W"),
    ("D", "W"),
    ("K", "Y"),
    ("P", "Y"),
    ("B", "Y"),
    ("F", "Y"),
    ("M", "Y"),
}
# Vowels: (Khmer vowel, series it needs). Stress digits are stripped first.
VOWEL = {
    "IY": ("ី", O),
    "IH": ("ី", O),
    "EY": ("េ", A),
    "EH": ("េ", A),
    "AE": ("េ", A),
    "AA": ("ូ", A),
    "AH": ("ា", A),
    "AO": ("ូ", A),
    "OW": ("ូ", A),
    "UW": ("ូ", O),
    "UH": ("ុ", O),
    "ER": ("ើ", O),
    "AY": ("ាយ", A),
    "AW": ("ៅ", A),
    "OY": ("យ", O),
}
GLIDE_FINAL = {"AY", "OY"}  # their Khmer form already ends in a consonant
SHORT = {"AE", "EH", "IH", "AH", "UH"}
DOUBLES = {"P", "T", "K", "CH", "JH"}  # only these double after a short vowel
CODA = {
    "P": "ប",
    "B": "ប",
    "T": "ត",
    "D": "ត",
    "K": "ក",
    "G": "ក",
    "CH": "ច",
    "JH": "ជ",
    "F": "ហ្វ",
    "V": "វ",
    "S": "ស",
    "Z": "ស",
    "SH": "ស",
    "ZH": "ស",
    "TH": "ធ",
    "DH": "ត",
    "M": "ម",
    "N": "ន",
    "NG": "ង",
    "L": "ល",
    "R": "រ",
    "HH": "",
    "W": "",
    "Y": "",
}
# Letter names for spelled-out abbreviations, as in the owner-approved ATM/GPS/PDF/QR/AI/OK.
LETTERS = {
    "A": "អេ",
    "B": "ប៊ី",
    "C": "ស៊ី",
    "D": "ឌី",
    "E": "អ៊ី",
    "F": "អេហ្វ",
    "G": "ជី",
    "H": "អេឆឹ",
    "I": "អាយ",
    "J": "ជេ",
    "K": "ខេ",
    "L": "អែល",
    "M": "អឹម",
    "N": "អិន",
    "O": "អូ",
    "P": "ភី",
    "Q": "ឃ្យូ",
    "R": "អរ",
    "S": "អេស",
    "T": "ធី",
    "U": "យូ",
    "V": "វី",
    "W": "ដាប់ប៊លយូ",
    "X": "អិច",
    "Y": "វ៉ាយ",
    "Z": "ហ្ស៊ិត",
}


def fetch_cmudict(path: Path = CMUDICT_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/cmusphinx/cmudict/{CMUDICT_COMMIT}/cmudict.dict"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != CMUDICT_SHA256:
            raise ValueError("cmudict.dict checksum mismatch")
        path.write_bytes(data)
    return path


def load_cmudict(path: Path = CMUDICT_PATH) -> dict[str, list[str]]:
    """First pronunciation of each word; `word(2)` alternates and comments are skipped."""
    out = {}
    for line in fetch_cmudict(path).read_text(encoding="utf-8").splitlines():
        word, *phones = line.split("#")[0].split()
        if phones and "(" not in word:
            out[word] = phones
    return out


def spell_letters(word: str) -> str:
    return "".join(LETTERS[c] for c in word.upper() if c in LETTERS)


def _is_vowel(p: str) -> bool:
    return p[-1].isdigit()


def syllables(phones: list[str]) -> list[tuple[list[str], str, list[str]]]:
    """(onset, vowel, coda) per vowel; consonants between vowels prefer the next onset."""
    idx = [i for i, p in enumerate(phones) if _is_vowel(p)]
    if not idx:
        return []
    out = []
    start = 0
    for n, v in enumerate(idx):
        nxt = idx[n + 1] if n + 1 < len(idx) else None
        between = phones[v + 1 : nxt] if nxt is not None else phones[v + 1 :]
        if nxt is None:
            coda = between
        elif len(between) >= 2 and tuple(between[-2:]) in CLUSTERS:
            coda = between[:-2]
        else:
            coda = between[:-1]
        out.append((phones[start:v], phones[v], coda))
        start = v + 1 + len(coda)
    return out


def _onset(cons: list[str], series: int) -> str:
    if not cons:
        return "អ" if series == A else "អ៊"
    if len(cons) >= 2 and tuple(cons[-2:]) in CLUSTERS:
        first, second = cons[-2:]
        head = PLAIN.get(first, ONSET[first])[series].rstrip("៉៊")
        if first == "S":
            head = "ស"
        return head + SUB.get(second, "")
    return ONSET[cons[-1]][series]


def _coda(cons: list[str], vowel: str) -> str:
    if not cons or vowel in GLIDE_FINAL:
        return ""
    if cons[0] == "R" and len(cons) > 1:
        cons = cons[1:]  # card → kaat: the r is not pronounced before a final stop
    return CODA.get(cons[0], "")


def _vowel(
    v: str, stressed: bool, onset: list[str], coda: list[str], nxt
) -> tuple[str, int]:
    before_r = coda[:1] == ["R"] or (
        not coda and nxt is not None and nxt[0][:1] == ["R"]
    )
    if v == "ER" and not stressed and nxt is None:
        return "័រ", O  # final unstressed -er: manager → ម៉ាណេជ័រ
    if v == "AA" and coda[:1] == ["R"]:
        return "ា", A  # card → កាត
    if v == "EH" and before_r:
        return "ែ", O  # share → ស៊ែរ
    if v == "AW" and coda[:1] == ["N"]:
        return "ោ", A  # download → ដោន
    if not stressed and v in {"IH", "AH"} and onset and coda and coda[:1] != ["NG"]:
        return "ិ", O  # ticket → …កិត
    return VOWEL[v]


class En2Km:
    def __init__(self, cmudict: dict[str, list[str]] | None = None):
        self.cmu = load_cmudict() if cmudict is None else cmudict

    def word(self, word: str) -> str | None:
        if "-" in word:
            parts = [self.word(p) for p in word.split("-") if p]
            return None if None in parts else "".join(parts)
        phones = self.cmu.get(word.lower())
        if phones is None:
            return None
        # Khmer speakers drop the y-glide in "tube": យូធូប.
        phones = [
            p
            for i, p in enumerate(phones)
            if not (
                p == "Y"
                and 0 < i < len(phones) - 1
                and not _is_vowel(phones[i - 1])
                and phones[i + 1].startswith("UW")
            )
        ]
        sylls = syllables(phones)
        out = []
        for i, (onset, vowel, coda) in enumerate(sylls):
            v = vowel.rstrip("012")
            stressed = not vowel.endswith("0")
            nxt = sylls[i + 1] if i + 1 < len(sylls) else None
            if nxt and v == "ER" and not nxt[0]:
                nxt[0].append("R")  # battery → ធឺ·រី: the r starts the next syllable
                sign, series = "ឺ", O
            else:
                sign, series = _vowel(v, stressed, onset, coda, nxt)
            # English short vowels can't end a stressed syllable: ticket → ធីក·កិត.
            if (
                stressed
                and v in SHORT
                and not coda
                and nxt
                and nxt[0] in [[d] for d in DOUBLES]
            ):
                coda = list(nxt[0])
            out.append(_onset(onset, series) + sign + _coda(coda, v))
        return "".join(out)

    def phrase(self, text: str) -> str:
        """Every word converted; abbreviations (all caps, ≤ 5 letters) spelled out."""
        parts = []
        for w in text.split():
            if w.isupper() and len(w) <= 5:
                parts.append(spell_letters(w))
            else:
                parts.append(self.word(w) or spell_letters(w))
        return " ".join(parts)
