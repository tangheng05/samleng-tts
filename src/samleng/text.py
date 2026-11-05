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


def verbalize(text: str) -> str:
    """Spell out numbers, dates, times and money in Khmer words."""
    text = text.replace("%", " ភាគរយ")
    for step in _STEPS:
        text = step(text)
    return text.replace("▁", "")
