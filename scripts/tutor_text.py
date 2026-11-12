"""Generate the templated part of the v5 teacher text: spoken maths and English-word drills.

uv run python scripts/tutor_text.py     # writes texts/distill_maths_v5.txt and texts/distill_mixed_v5.txt

Tutoring talk repeats a few sentence shapes with different numbers and words, so templates
fit here (the hand-written v5 files cover free conversation). English words come from the
owner-approved spellings only, so every sentence can also be written in Khmer script.
"""

import json
import random

from samleng.paths import ROOT
from samleng.text import EN_KM, train_words

MATHS = [
    ("{a} បូក {b} ស្មើនឹងប៉ុន្មាន?", "add"),
    ("{a} បូក {b} ស្មើនឹង {c}។", "add"),
    ("តើ {a} បូក {b} បានប៉ុន្មាន?", "add"),
    ("{c} ដក {a} ស្មើនឹងប៉ុន្មាន?", "add"),
    ("{c} ដក {a} នៅសល់ {b}។", "add"),
    ("{a} គុណនឹង {b} ស្មើនឹងប៉ុន្មាន?", "mul"),
    ("{a} គុណនឹង {b} ស្មើនឹង {c}។", "mul"),
    ("{c} ចែកនឹង {a} ស្មើនឹងប៉ុន្មាន?", "mul"),
    ("{c} ចែកនឹង {a} ស្មើនឹង {b}។", "mul"),
    ("តើ {a} ធំជាង {b} ឬតូចជាង {b}?", "pair"),
    ("លេខ {a} ធំជាងលេខ {b}។", "more"),
    ("ចូររាប់ពី {a} ដល់ {b}។", "less"),
    ("បន្ទាប់ពីលេខ {a} គឺលេខអ្វី?", "one"),
    ("លេខ {a} នៅចន្លោះលេខអ្វី និងលេខអ្វី?", "one"),
    ("បើប្អូនមានប៊ិច {c} ដើម ហើយឱ្យមិត្ត {a} ដើម តើនៅសល់ប៉ុន្មានដើម?", "add"),
    ("ក្នុងថ្នាក់មានសិស្សប្រុស {a} នាក់ និងសិស្សស្រី {b} នាក់ សរុបមានប៉ុន្មាននាក់?", "add"),
    ("សៀវភៅមួយក្បាលថ្លៃ {a} រៀល បើទិញ {b} ក្បាល ត្រូវចំណាយប៉ុន្មាន?", "price"),
    ("ពាក់កណ្តាលនៃ {c} គឺ {a}។", "half"),
    ("{a} ភាគរយ ស្មើនឹង {a} ភាគ 100។", "one"),
    ("ចម្លើយគឺ {c} ត្រូវហើយ។", "add"),
    ("មិនមែន {a} ទេ ចម្លើយត្រឹមត្រូវគឺ {b}។", "pair"),
    ("សូមសរសេរលេខ {c} នៅលើក្តារខៀន។", "mul"),
]
WORDS = [
    "ពាក្យ {w} អានយ៉ាងម៉េច?",
    "សូមអានតាមខ្ញុំ {w}។",
    "ប្អូនស្គាល់ពាក្យ {w} ទេ?",
    "ពាក្យ {w} មានន័យថាអ្វី?",
    "សាកប្រកបពាក្យ {w} មើល។",
    "ចូរតែងប្រយោគមួយដោយប្រើពាក្យ {w}។",
    "ថ្ងៃនេះយើងរៀនពាក្យថ្មីមួយគឺ {w}។",
    "ត្រូវហើយ ពាក្យនេះអានថា {w}។",
    "សូមសរសេរពាក្យ {w} ចំនួនបីដង។",
    "ពាក្យ {w} និងពាក្យ {v} ខុសគ្នាយ៉ាងម៉េច?",
    "មិនមែន {w} ទេ គឺ {v}។",
    "ប្អូនចាំពាក្យ {w} ដែលយើងរៀនម្សិលមិញទេ?",
    "ពាក្យ {w} ពិបាកអានបន្តិច សាកម្តងទៀត។",
    "ល្អណាស់ ប្អូនអានពាក្យ {w} បានត្រឹមត្រូវ។",
]
# Not teaching words: letters, abbreviations (written in capitals elsewhere) and filler words.
SKIP = {
    "from",
    "very",
    "well",
    "really",
    "maybe",
    "again",
    "same",
    "sure",
    "done",
    "anyway",
}
# Names keep their capital letter, as people write them.
PROPER = {
    "sunday",
    "monday",
    "january",
    "google",
    "netflix",
    "minecraft",
    "gmail",
    "python",
    "excel",
    "zoom",
    "lego",
}


def maths(rng: random.Random, n: int) -> list[str]:
    out = []
    while len(out) < n:
        text, kind = MATHS[len(out) % len(MATHS)]
        a, b = rng.randint(2, 60), rng.randint(2, 40)
        if kind == "mul":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            c = a * b
        elif kind == "half":
            c = 2 * a
        elif kind == "price":
            a, b, c = (
                rng.choice([500, 1000, 1500, 2000, 2500, 5000]),
                rng.randint(2, 9),
                0,
            )
        elif kind in ("more", "less", "pair"):
            a, b = sorted({a, b + 61}, reverse=kind == "more")
            c = 0
        else:
            c = a + b
        line = text.format(a=a, b=b, c=c)
        if line not in out:
            out.append(line)
    return out


def english(rng: random.Random, n: int, banned: set[str]) -> list[str]:
    words = sorted(
        w
        for w in {**EN_KM, **train_words()}
        if w.isalpha()
        and len(w) >= 4
        and w not in SKIP
        and not any(b in w for b in banned)
    )
    rng.shuffle(words)
    words = [w.capitalize() if w in PROPER else w for w in words]
    return [
        WORDS[i % len(WORDS)].format(
            w=words[i % len(words)], v=words[-1 - i % len(words)]
        )
        for i in range(n)
    ]


def main() -> None:
    split = json.loads(
        (ROOT / "bench" / "khmertts_bench_v1_mixed_split.json").read_text()
    )
    rng = random.Random(0)
    for name, lines in (
        ("distill_maths_v5.txt", maths(rng, 300)),
        ("distill_mixed_v5.txt", english(rng, 500, set(split["banned"]))),
    ):
        (ROOT / "texts" / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(len(lines), "->", name)


if __name__ == "__main__":
    main()
