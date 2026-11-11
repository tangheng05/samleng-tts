from samleng.text import EN_KM, expand_acronyms, train_words, transliterate, verbalize


def test_khmer_digits_are_spelled_out():
    assert verbalize("២៥ នាក់") == "ម្ភៃប្រាំ នាក់"


def test_percent_is_spoken():
    assert "ភាគរយ" in verbalize("10%")


def test_no_tha_separator_leaks():
    assert "▁" not in verbalize("ឆ្នាំ 1998")


def test_thousands_separators_are_not_read_as_comma():
    assert verbalize("25,000 រៀល") == "ពីរម៉ឺនប្រាំពាន់ រៀល"
    assert "ក្បៀស" not in verbalize("1,234,500")


def test_decimal_point_still_read():
    assert verbalize("1.5") == "មួយចុចប្រាំ"


def test_celsius_is_spoken():
    assert verbalize("38°C") == "សាមសិបប្រាំបី អង្សាសេ"


def test_afternoon_24h_time_is_read_as_rosiel():
    assert verbalize("ម៉ោង 14:00។") == verbalize("ម៉ោង 2 រសៀល។")


def test_whole_hour_has_no_zero():
    assert verbalize("ម៉ោង 8:00 ព្រឹក") == verbalize("ម៉ោង 8 ព្រឹក")


def test_existing_period_word_not_doubled():
    assert verbalize("ម៉ោង 15:00 រសៀល") == verbalize("ម៉ោង 3 រសៀល")
    assert verbalize("3:30 រសៀល").count("រសៀល") == 1


def test_english_words_are_written_in_khmer_script():
    assert transliterate("ខ្ញុំមាន meeting ម៉ោងប្រាំបួន។") == "ខ្ញុំមាន មីធីង ម៉ោងប្រាំបួន។"


def test_transliteration_ignores_case_and_prefers_phrases():
    assert transliterate("Thank you ណាស់") == "ថេងគ្យូ ណាស់"
    assert transliterate("ONLINE") == "អនឡាញ"


def test_transliteration_keeps_unknown_words_and_partial_matches():
    assert transliterate("Netflix") == "Netflix"
    assert transliterate("apps") == "apps"


def test_fractions_are_read_with_phiek_and_dates_are_untouched():
    assert verbalize("3/4").replace(" ", "") == "បីភាគបួន"
    assert "ភាគ" not in verbalize("ថ្ងៃទី 14/04/2026")


def test_transliteration_leaves_hyphenated_compounds_whole():
    assert transliterate("សូម check-up នៅទីនេះ") == "សូម check-up នៅទីនេះ"


def test_half_hour_is_read_as_kanlah():
    # Owner-confirmed: 7:30 ព្រឹក is said ប្រាំពីរកន្លះព្រឹក.
    assert verbalize("ម៉ោង 7:30 ព្រឹក").replace(" ", "") == "ម៉ោងប្រាំពីរកន្លះព្រឹក"
    assert verbalize("ម៉ោង 14:30").replace(" ", "") == "ម៉ោងពីរកន្លះរសៀល"
    assert "កន្លះ" not in verbalize("ម៉ោង 7:15 ព្រឹក")


def test_scores_are_read_with_tul_but_math_stays_minus():
    # Owner-confirmed: a 2-1 result is ពីរទល់មួយ; in a sum, 5-3 is still ប្រាំដកបី.
    assert "ពីរទល់មួយ" in verbalize("ចប់ដោយលទ្ធផល 2-1។").replace(" ", "")
    assert "មួយទល់មួយ" in verbalize("ក្រុមទាំងពីរស្មើ 1-1").replace(" ", "")
    assert verbalize("5-3").replace(" ", "") == "ប្រាំដកបី"


def test_every_english_word_in_bench_v1_mixed_has_a_spelling():
    import re

    from samleng.bench import load_bench
    from samleng.paths import ROOT

    bench = load_bench(ROOT / "bench" / "khmertts_bench_v1.jsonl")
    mixed = [it for it in bench if it.category == "mixed"]
    left = [
        (it.id, w)
        for it in mixed
        for w in re.findall(r"[A-Za-z]+", transliterate(it.text))
    ]
    assert left == []


def test_short_date_after_thngai_is_not_a_fraction():
    short = verbalize("ថ្ងៃទី 14/04 ជាថ្ងៃចូលឆ្នាំ។")
    assert "ភាគ" not in short and "/" not in short
    full = verbalize("ថ្ងៃទី 14/04/2026 ជាថ្ងៃចូលឆ្នាំ។")
    assert short.replace(" ", "") == full.replace(" ", "").replace("ពីរពាន់ម្ភៃប្រាំមួយ", "")


def test_transliterate_with_training_words():
    words = {**EN_KM, **train_words()}
    assert transliterate("Netflix", words) == words["netflix"]
    assert (
        transliterate("សូម check-up នៅទីនេះ", words) == f"សូម {words['check-up']} នៅទីនេះ"
    )
    assert transliterate("Netflix") == "Netflix"


def test_training_words_are_khmer_and_skip_eval_words():
    words = train_words()
    assert len(words) >= 517
    assert not set(words) & set(EN_KM)
    assert all("\u1780" <= c <= "\u17ff" for v in words.values() for c in v)


def test_expand_acronyms_uses_letter_names():
    # The model lowercases text, so "AI" would read like the word "ai" without this.
    assert expand_acronyms("ក្រុមហ៊ុនប្រើ AI ដើម្បីជួយ") == "ក្រុមហ៊ុនប្រើ អេអាយ ដើម្បីជួយ"
    assert expand_acronyms("RAM 8GB") == f"{train_words()['ram']} 8GB"
    assert expand_acronyms("ChatGPT") == "ChatGPT"
    assert expand_acronyms("GPT") == "ជីភីធី"
    assert expand_acronyms("I like Facebook") == "I like Facebook"
