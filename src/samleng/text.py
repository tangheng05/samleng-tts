import functools
import json
import re

import tha.cardinals
import tha.currency
import tha.datetime
import tha.decimals
import tha.normalize
import tha.phone_numbers

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
# Two-part only, so dates like 14/04/2026 stay with the date reader.
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


def _fraction(m: re.Match) -> str:
    # After ថ្ងៃ/ថ្ងៃទី, 14/04 is a day and month: read it like a full date, minus the year.
    if m.string[: m.start()].rstrip().endswith(("ថ្ងៃទី", "ថ្ងៃ")):
        return f"{int(m[1])} {int(m[2])}"
    return f"{m[1]} ភាគ {m[2]}"


def verbalize(text: str) -> str:
    """Spell out numbers, dates, times and money in Khmer words."""
    # Tha reads "25,000" as "25 comma 000"; Khmer uses "." for decimals.
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = _TIME.sub(_spoken_time, text)
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
    "qr": "ឃ្យូអា",
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


def transliterate(text: str, words: dict[str, str] = EN_KM) -> str:
    """Rewrite known English words in Khmer script, as Khmer speakers say them."""
    return _pattern(frozenset(words)).sub(lambda m: words[m[1].lower()], text)
