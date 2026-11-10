from samleng.eval.cer import english_hits, khmer_cer, mixed_cer, normalize_khmer
from samleng.text import EN_KM


def test_normalize_drops_spaces_zwsp_and_punctuation():
    assert normalize_khmer("សួស្តី\u200b ខ្ញុំ។") == "សួស្តីខ្ញុំ"


def test_normalize_keeps_repetition_mark():
    assert normalize_khmer("ផ្សេងៗ") == "ផ្សេងៗ"


def test_cer_ignores_word_spacing():
    assert khmer_cer("ថ្ងៃនេះ អាកាស", "ថ្ងៃនេះអាកាស") == 0.0


def test_cer_counts_character_errors():
    assert khmer_cer("abcd", "abxd") == 0.25


def test_cer_is_corpus_level_over_lists():
    assert khmer_cer(["ab", "cd"], ["ab", "cx"]) == 0.25


def test_cer_treats_digits_and_spoken_numbers_as_equal():
    assert khmer_cer("ឆ្នាំ 1998", "ឆ្នាំ មួយពាន់ប្រាំបួនរយកៅសិបប្រាំបី") == 0.0


def test_subscript_da_and_ta_are_the_same_letter_for_cer():
    # They look identical and are read the same; writers use either.
    assert khmer_cer("ម្ដង ស្ដាប់", "ម្តងស្តាប់") == 0


def test_mixed_cer_accepts_english_in_either_script():
    ref = "សូម upload រូបថតនៅលើ website។"
    khmer_script = f"សូម{EN_KM['upload']}រូបថតនៅលើ{EN_KM['website']}"
    assert khmer_cer(ref, khmer_script) > 0.3
    assert mixed_cer(ref, khmer_script) == 0
    assert mixed_cer(ref, "សូម Upload រូបថតនៅលើ Website") == 0


def test_english_hits_counts_each_english_word_once():
    ref = "សូម upload រូបថតនៅលើ website។"
    assert english_hits(ref, f"សូម{EN_KM['upload']}រូបថតនៅលើវេបសៃ") == (2, 2)
    assert english_hits(ref, f"សូម{EN_KM['upload']}រូបថតនៅលើផ្ទះ") == (1, 2)
    assert english_hits(ref, "សូមរូបថត") == (0, 2)
    assert english_hits("ខ្ញុំសុខសប្បាយ", "ខ្ញុំសុខសប្បាយ") == (0, 0)
