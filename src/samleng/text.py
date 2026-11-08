import re

import tha.cardinals
import tha.currency
import tha.datetime
import tha.decimals
import tha.normalize
import tha.phone_numbers

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


def _spoken_time(m: re.Match) -> str:
    hour, minutes, space = int(m[1]), m[2], m[3]
    period = ""
    # Owner-confirmed: 14:00 is said ម៉ោងពីររសៀល. Evening hours are left as-is.
    if 13 <= hour <= 17:
        hour -= 12
        if not m.string[m.end() :].startswith(_PERIODS):
            period = " រសៀល"
    clock = str(hour) if int(minutes) == 0 else f"{hour}:{minutes}"
    return clock + period + space


def verbalize(text: str) -> str:
    """Spell out numbers, dates, times and money in Khmer words."""
    # Tha reads "25,000" as "25 comma 000"; Khmer uses "." for decimals.
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = _TIME.sub(_spoken_time, text)
    text = text.replace("%", " ភាគរយ").replace("°C", " អង្សាសេ")
    for step in _STEPS:
        text = step(text)
    return text.replace("▁", "")


# Owner-reviewed spellings (2026-11-08) for the English words in bench/pilot_v1 mixed.
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
}
_EN = re.compile(
    r"(?<![A-Za-z])("
    + "|".join(re.escape(w) for w in sorted(EN_KM, key=len, reverse=True))
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)


def transliterate(text: str) -> str:
    """Rewrite known English words in Khmer script, as Khmer speakers say them."""
    return _EN.sub(lambda m: EN_KM[m[1].lower()], text)
