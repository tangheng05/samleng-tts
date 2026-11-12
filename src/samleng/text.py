"""Text front end: Khmer text as written -> the words a voice should say.

Numbers, dates, times and money are spelled out (`verbalize`), abbreviations are spelled
letter by letter (`expand_acronyms`) and known English words can be written in Khmer script
(`transliterate`, `khmer_script`). Readings marked owner-confirmed were checked by ear.
"""

import functools
import json
import re

import tha.cardinals
import tha.currency
import tha.datetime
import tha.decimals
import tha.normalize
import tha.phone_numbers

from samleng.en2km import spell_letters
from samleng.paths import ROOT

# Order matters: specific patterns (dates, phones, money) before bare numbers.
_STEPS = (
    tha.normalize.processor,
    tha.datetime.date_processor,
    tha.datetime.time_processor,
    tha.phone_numbers.processor,
    tha.currency.processor,
    tha.decimals.processor,
    tha.cardinals.processor,
)


_PERIODS = ("ព្រឹក", "ថ្ងៃត្រង់", "រសៀល", "ល្ងាច", "យប់")
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])(\s*)")
_MONTHS = (
    "មករា",
    "កុម្ភៈ",
    "មីនា",
    "មេសា",
    "ឧសភា",
    "មិថុនា",
    "កក្កដា",
    "សីហា",
    "កញ្ញា",
    "តុលា",
    "វិច្ឆិកា",
    "ធ្នូ",
)
_DAY_WORDS = ("ថ្ងៃទី", "ថ្ងៃ")
_DATE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4})(?![\d/])")
# Two-part only, so 14/04/2026 is left to the date rule above.
_FRACTION = re.compile(r"(?<![\d/])(\d+)/(\d+)(?![\d/])")
# A hyphen is minus in sums; only a preceding score word makes it ទល់ (owner-confirmed).
_SCORE = re.compile(r"(លទ្ធផល|ពិន្ទុ|ឈ្នះ|ចាញ់|ស្មើ)\s*(\d+)\s*-\s*(\d+)")


def _spoken_time(m: re.Match) -> str:
    hour, minutes, space = int(m[1]), m[2], m[3]
    period = ""
    # Owner-confirmed: 14:00 is said ម៉ោងពីររសៀល. Evening hours are left as-is.
    if 13 <= hour <= 17:
        hour -= 12
        if not m.string[m.end() :].startswith(_PERIODS):
            period = " រសៀល"
    if int(minutes) == 0:
        clock = str(hour)
    elif minutes == "30":
        clock = f"{hour}កន្លះ"  # owner-confirmed: 7:30 is ប្រាំពីរកន្លះ
    else:
        clock = f"{hour}:{minutes}"
    return clock + period + space


def _after_day_word(m: re.Match) -> bool:
    return m.string[: m.start()].rstrip().endswith(_DAY_WORDS)


def _date(m: re.Match) -> str:
    # Owner (2026-11-12): 14/04/2026 is said "day 14, month Mesa, year 2026", not "14 4 2026".
    day, month = int(m[1]), int(m[2])
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m[0]
    lead = "" if _after_day_word(m) else "ថ្ងៃទី "
    return f"{lead}{day} ខែ{_MONTHS[month - 1]} ឆ្នាំ {int(m[3])}"


def _fraction(m: re.Match) -> str:
    # After ថ្ងៃ/ថ្ងៃទី, 14/04 is a day and month.
    if _after_day_word(m):
        day, month = int(m[1]), int(m[2])
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{day} ខែ{_MONTHS[month - 1]}"
        return f"{day} {month}"
    return f"{m[1]} ភាគ {m[2]}"


def verbalize(text: str) -> str:
    """Spell out numbers, dates, times and money in Khmer words."""
    # Tha reads "25,000" as "25 comma 000"; Khmer uses "." for decimals.
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = _TIME.sub(_spoken_time, text)
    text = _DATE.sub(_date, text)
    text = _FRACTION.sub(_fraction, text)
    text = _SCORE.sub(r"\1 \2 ទល់ \3", text)
    text = text.replace("%", " ភាគរយ").replace("°C", " អង្សាសេ")
    for step in _STEPS:
        text = step(text)
    return text.replace("▁", "")


# Owner-reviewed spellings (2026-11-08) for every English word in bench v1 mixed.
EN_KM = {
    "ai": "អេអាយ",
    "atm": "អេធីអឹម",
    "deadline": "ដេតឡាញ",
    "facebook": "ហ្វេសប៊ុក",
    "gps": "ជីភីអេស",
    "grab": "ក្រេប",
    "ok": "អូខេ",
    "online": "អនឡាញ",
    "pdf": "ភីឌីអេហ្វ",
    "qr": "ឃ្យូអរ",
    "sorry": "សូរី",
    "telegram": "តេឡេក្រាម",
    "thank you": "ថេងគ្យូ",
    "wifi": "វ៉ាយហ្វាយ",
    "youtube": "យូធូប",
    "account": "អាខោន",
    "app": "អេប",
    "call": "ខល",
    "check": "ឆែក",
    "class": "ក្លាស",
    "code": "កូដ",
    "download": "ដោនឡូត",
    "email": "អ៊ីមែល",
    "engineer": "អ៊ីនជីនៀ",
    "file": "ហ្វាយ",
    "work from home": "វើកហ្វ្រមហូម",
    "laptop": "ឡេបថប",
    "link": "លីង",
    "meeting": "មីធីង",
    "message": "មែសេច",
    "password": "ផាសវើដ",
    "post": "ផុស",
    "print": "ព្រីន",
    "project": "ប្រូជិច",
    "save": "សេវ",
    "share": "ស៊ែរ",
    "software": "សូហ្វវែរ",
    "startup": "ស្តាតអាប់",
    "update": "អាប់ដេត",
    "version": "វើសិន",
    "video": "វីដេអូ",
    "boss": "បូស",
    "coffee shop": "ខហ្វីសប",
    "congratulations": "ខុងក្រេជូលេសិន",
    "driver": "ដ្រាយវើ",
    "happy birthday": "ហេបភី បឺធដេ",
    "instagram": "អ៊ីនស្តាក្រាម",
    "manager": "ម៉ាណេជ័រ",
    "teacher": "ធីឆឺ",
    "tiktok": "ធិកថុក",
    "weekend": "វីកអិន",
    "zoom": "ហ្ស៊ូម",
    "approve": "អាប្រូវ",
    "backup": "បែកអាប់",
    "battery": "ប៊ែតធឺរី",
    "breakfast": "ប្រេកហ្វាស",
    "browser": "ប្រោវស៊ើ",
    "budget": "បាជ់ជិត",
    "busy": "ប៊ីស៊ី",
    "camera": "ខេមេរ៉ា",
    "cancel": "ខែនស៊ល",
    "card": "កាត",
    "cash": "ខេស",
    "channel": "ឆាណែល",
    "charge": "ឆាជ",
    "charger": "ឆាជ័រ",
    "check-in": "ឆែកអ៊ីន",
    "coding": "ខូឌីង",
    "comment": "ខមមិន",
    "confirm": "ខុនហ្វឺម",
    "copy": "ខូពី",
    "counter": "ខោនធ័រ",
    "deliver": "ឌីលីវើ",
    "design": "ឌីហ្សាញ",
    "focus": "ហ្វូខឹស",
    "follow": "ហ្វូឡូ",
    "freelancer": "ហ្វ្រីឡេនស៊ើ",
    "game": "ហ្គេម",
    "goal": "ហ្គោល",
    "homework": "ហូមវើក",
    "hotel": "ហូតែល",
    "intern": "អ៊ីនធើន",
    "invoice": "អ៊ីនវយ",
    "like": "ឡាយ",
    "login": "ឡុកអ៊ីន",
    "logout": "ឡុកអៅ",
    "mall": "ម៉ល",
    "marketing": "ម៉ាឃីធីង",
    "order": "អ័រដឺ",
    "page": "ផេក",
    "partner": "ផាតន័រ",
    "paste": "ផេស",
    "presentation": "ប្រេហ្សិនតេសិន",
    "report": "រីផត",
    "reset": "រីសិត",
    "resort": "រីសត",
    "restart": "រីស្តាត",
    "scan": "ស្កេន",
    "service": "សឺវីស",
    "session": "សេសិន",
    "shopping": "សបពីង",
    "skip": "ស្គីប",
    "staff": "ស្តាហ្វ",
    "story": "ស្តូរី",
    "subscribe": "សាប់ស្គ្រាយ",
    "sweet": "ស្វីត",
    "text": "ធិច",
    "ticket": "ធីកកិត",
    "trust": "ត្រាស់",
    "upload": "អាប់ឡូត",
    "website": "វេបសាយ",
    "workshop": "វើកសប",
}


@functools.cache
def _pattern(keys: frozenset[str]) -> re.Pattern:
    return re.compile(
        r"(?<![A-Za-z-])("
        + "|".join(re.escape(w) for w in sorted(keys, key=len, reverse=True))
        + r")(?![A-Za-z-])",
        re.IGNORECASE,
    )


@functools.cache
def train_words() -> dict[str, str]:
    """Owner-approved spellings of the distillation text's English words (not the bench's)."""
    return json.loads((ROOT / "texts" / "en_km_train.json").read_text(encoding="utf-8"))


def english_spans(text: str) -> list[tuple[str, str]]:
    """(English, approved Khmer spelling) for each known English word or phrase in text."""
    return [
        (m[1], EN_KM[m[1].lower()]) for m in _pattern(frozenset(EN_KM)).finditer(text)
    ]


_ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,6}(?![A-Za-z0-9])")


def expand_acronyms(text: str) -> str:
    """Write all-caps abbreviations in Khmer script (AI → អេអាយ) before TTS.

    The model lowercases its input, so the capitals are the only hint to spell a word out.
    Owner-approved word readings win (RAM → រ៉ែម); letter readings get a space between
    letters, which the model reads as a short pause (clearer than running them together).
    """
    words = {**EN_KM, **train_words()}

    def spell(m: re.Match) -> str:
        approved = words.get(m[0].lower())
        if approved and approved != spell_letters(m[0]):
            return approved
        return " ".join(spell_letters(c) for c in m[0])

    return _ACRONYM.sub(spell, text)


def transliterate(text: str, words: dict[str, str] = EN_KM) -> str:
    """Rewrite known English words in Khmer script, as Khmer speakers say them."""
    return _pattern(frozenset(words)).sub(lambda m: words[m[1].lower()], text)


def khmer_script(text: str) -> str:
    """Spoken form with every known English word in Khmer script (what variant B reads)."""
    words = {**EN_KM, **train_words()}
    return transliterate(expand_acronyms(verbalize(text)), words)
