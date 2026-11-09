import pytest

from samleng.en2km import En2Km, spell_letters, syllables

# Tiny CMUdict excerpt so the tests need no download.
CMU = {
    "meeting": ["M", "IY1", "T", "IH0", "NG"],
    "like": ["L", "AY1", "K"],
    "print": ["P", "R", "IH1", "N", "T"],
    "zoom": ["Z", "UW1", "M"],
    "card": ["K", "AA1", "R", "D"],
    "email": ["IY0", "M", "EY1", "L"],
}


@pytest.fixture
def conv():
    return En2Km(CMU)


def test_syllables_split_on_vowels_with_legal_onsets():
    assert syllables(["P", "R", "IH1", "N", "T"]) == [(["P", "R"], "IH1", ["N", "T"])]
    assert syllables(["M", "IY1", "T", "IH0", "NG"]) == [
        (["M"], "IY1", []),
        (["T"], "IH0", ["NG"]),
    ]


def test_vowel_picks_the_consonant_series(conv):
    # ី needs an o-series consonant (ម, ធ); ាយ needs an a-series one (ឡ, not ល).
    assert conv.word("meeting") == "មីធីង"
    # ាយ already ends in a consonant, so the English final k is dropped.
    assert conv.word("like") == "ឡាយ"


def test_clusters_and_vowel_initial_words(conv):
    assert conv.word("print") == "ព្រីន"
    assert conv.word("email").startswith("អ៊ី")


def test_unknown_word_returns_none_and_acronyms_spell_letters(conv):
    assert conv.word("blorptastic") is None
    assert spell_letters("ATM") == "អេធីអឹម"
    assert spell_letters("GPS") == "ជីភីអេស"
