from samleng.phrases import MAX_RUN, RUN, phrase_spaces

LONG = "ត្រង់ចំណុចខាងត្បូងខ្ទង់រយហុកសិបប្រាំបីតាមបណ្តោយផ្លូវជាតិលេខពីរស្ថិតក្នុងសង្កាត់ចាក់អង្រែលើខណ្ឌមានជ័យ"


def test_short_text_is_unchanged():
    assert phrase_spaces("ពូកែណាស់ ប្អូនឆ្លើយត្រូវហើយ។") == "ពូកែណាស់ ប្អូនឆ្លើយត្រូវហើយ។"


def test_long_run_gets_spaces_and_nothing_else_changes():
    out = phrase_spaces(LONG + " ។")
    assert " " in out.removesuffix(" ។")
    assert out.replace(" ", "") == (LONG + "។")
    # With no confidence floor every piece fits.
    forced = phrase_spaces(LONG, min_score=float("-inf"))
    assert all(len(piece) <= MAX_RUN for piece in RUN.findall(forced))


def test_spelled_out_numbers_stay_together():
    number = "មួយពាន់ប្រាំបួនរយកៅសិបប្រាំបី"
    text = "គាត់កើតនៅឆ្នាំ" + number + "ក្នុងខេត្តបាត់ដំបងហើយរៀននៅទីនោះ"
    for floor in (-0.5, float("-inf")):
        assert number in phrase_spaces(text, max_run=20, min_score=floor)


def test_existing_spaces_latin_and_digits_are_kept():
    text = "សូមផ្ញើ email មកខ្ញុំ នៅម៉ោង 9 ព្រឹក។"
    assert phrase_spaces(text) == text
    mixed = LONG + " check-in 25 " + LONG
    assert phrase_spaces(mixed).replace(" ", "") == mixed.replace(" ", "")
    assert " check-in 25 " in phrase_spaces(mixed)


def test_same_input_same_output():
    assert phrase_spaces(LONG) == phrase_spaces(LONG)
