from samleng.eval.cer import khmer_cer, normalize_khmer


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
