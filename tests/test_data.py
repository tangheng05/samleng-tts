import numpy as np

from samleng.data import (
    TARGET_SR,
    LeakCheck,
    distill_problems,
    distill_reject,
    english_words,
    mixed_group,
    mixed_split,
    parse_line_index,
    prepare_audio,
    unsegment,
)


def tone(seconds, sr, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_prepare_audio_resamples_trims_and_levels():
    sr = 48000
    silence = np.zeros(sr, dtype=np.float32)
    y = np.concatenate([silence, tone(1.0, sr, amp=0.01), silence])
    out = prepare_audio(y, sr)
    # 2 s of silence removed; trim works in frames, so allow some edge.
    assert TARGET_SR * 1.0 <= len(out) <= TARGET_SR * 1.25
    assert np.abs(out).max() > 0.01  # quiet input was boosted


def test_leak_check_catches_bench_text_inside_a_longer_clip():
    bench = ["ខ្ញុំចូលចិត្តញ៉ាំបាយជាមួយគ្រួសាររៀងរាល់ល្ងាច។"]
    leaks = LeakCheck(bench)
    assert leaks("ម្សិលមិញ ខ្ញុំចូលចិត្តញ៉ាំបាយជាមួយគ្រួសាររៀងរាល់ល្ងាច ណាស់")
    assert not leaks("ថ្ងៃនេះអាកាសធាតុល្អណាស់។")


def test_leak_check_ignores_spacing_and_punctuation():
    leaks = LeakCheck(["ថ្ងៃនេះអាកាសធាតុល្អណាស់។"])
    assert leaks("ថ្ងៃនេះ អាកាសធាតុ ល្អណាស់")


def test_parse_line_index_handles_both_openslr_layouts():
    lines = [
        "khm_7548_1882674737\tសួស្តី",
        "khm_0308_0011865986\t\tអរគុណ ច្រើន\n",
        "",
    ]
    assert parse_line_index(lines) == [
        ("khm_7548_1882674737", "7548", "សួស្តី"),
        ("khm_0308_0011865986", "0308", "អរគុណ ច្រើន"),
    ]


def test_unsegment_joins_khmer_words_but_keeps_spaces_next_to_latin_and_digits():
    # Owner heard sengtha stretch every word when OpenSLR's word spaces are kept.
    assert unsegment("ស្ពាន វែង ជាង គេ") == "ស្ពានវែងជាងគេ"
    assert unsegment("ចាក់ វីដេអូ YouTube ឱ្យ ខ្ញុំ") == "ចាក់វីដេអូ YouTube ឱ្យខ្ញុំ"
    assert unsegment("ឆ្នាំ 2024 នេះ") == "ឆ្នាំ 2024 នេះ"


def test_english_words_are_lowercased_and_keep_hyphenated_compounds():
    assert english_words("សូម check-in នៅ Hotel ABC") == {"check-in", "hotel", "abc"}


def test_mixed_split_holds_out_words_from_training():
    from samleng.bench import Item

    items = [
        Item("m1", "mixed", "ខ្ញុំ check email"),
        Item("m2", "mixed", "បើក app"),
        Item("m3", "mixed", "ផ្ញើ email ឥឡូវ"),
        Item("m4", "mixed", "ទិញ ticket"),
    ]
    split = mixed_split(items, seed=1)
    assert split == mixed_split(items, seed=1)
    assert not set(split["unseen"]) & set(split["seen"] + split["partial"])
    banned = set(split["banned"])
    for it in items:
        words = english_words(it.text)
        if it.id in split["unseen"]:
            assert words <= banned
        if it.id in split["seen"]:
            assert not words & banned
        if it.id in split["partial"]:
            assert words & banned and words - banned


def test_distill_problems_flags_held_out_words_even_inside_longer_words():
    leak = LeakCheck(["ថ្ងៃនេះអាកាសធាតុល្អណាស់។"])
    banned = {"post", "email"}
    assert distill_problems("សូមផ្ញើ file ឥឡូវ", leak, banned) == []
    assert distill_problems("សូមផ្ញើ email ឥឡូវ", leak, banned) == ["held-out word"]
    assert distill_problems("ធ្វើ poster ថ្មី", leak, banned) == ["held-out word"]
    assert distill_problems("ថ្ងៃនេះ អាកាសធាតុ ល្អណាស់", leak, banned) == ["bench overlap"]


def test_distill_reject_checks_voice_then_misreads_then_speed():
    ok = {"source": "openslr42", "spk": 0.97, "cer": 0.05, "cps": 12.0}
    rate = (9.0, 16.0)
    assert distill_reject(ok, rate) is None
    assert distill_reject(ok | {"spk": 0.85}, rate) == "voice"
    assert distill_reject(ok | {"cer": 0.4}, rate) == "misread"
    assert distill_reject(ok | {"source": "mixed_v1", "cer": 0.4}, rate) is None
    assert distill_reject(ok | {"source": "mixed_v1", "cer": 0.7}, rate) == "misread"
    assert distill_reject(ok | {"cps": 20.0}, rate) == "speed"
    # Later batches are mixed too: the looser limit must follow the kind, not the version.
    assert distill_reject(ok | {"source": "mixed_v2", "cer": 0.4}, rate) is None
    assert distill_reject(ok | {"source": "numbers_v2", "cer": 0.4}, rate) == "misread"


def test_mixed_group_relabels_only_mixed():
    split = {"unseen": ["mixed-01"], "seen": ["mixed-02"], "partial": ["mixed-03"]}
    assert mixed_group("mixed-01", "mixed", split) == "mixed_unseen"
    assert mixed_group("mixed-02", "mixed", split) == "mixed_seen"
    assert mixed_group("mixed-03", "mixed", split) == "mixed_partial"
    assert mixed_group("pure-01", "pure", split) == "pure"


def test_both_scripts_adds_a_khmer_copy_only_when_fully_spelled():
    from samleng.data import both_scripts

    words = {"file": "ហ្វាយ"}
    row = {"id": "m-1", "text": "សូមផ្ញើ file តាម USB ។", "audio": "a.wav"}
    latin, khmer = both_scripts(row, words)
    assert (
        latin["id"] == "m-1" and "file" in latin["text"] and "USB" not in latin["text"]
    )
    assert khmer["id"] == "m-1-km" and khmer["audio"] == "a.wav"
    assert "ហ្វាយ" in khmer["text"] and "file" not in khmer["text"]
    # An unknown word stays Latin, so there is no Khmer-script copy.
    assert (
        len(both_scripts({"id": "m-2", "text": "សូមផ្ញើ file និង folder ។"}, words)) == 1
    )
    # No English at all: one row, unchanged.
    plain = {"id": "p-1", "text": "ពូកែណាស់ ។"}
    assert both_scripts(plain, words) == [plain]


def test_training_rows_filters_and_refuses_held_out_words():
    import pytest

    from samleng.data import training_rows

    words = {"file": "ហ្វាយ"}
    rows = [
        {"id": "a", "text": "សូមផ្ញើ file ។"},
        {"id": "b", "text": "ពូកែណាស់ ។"},
        {"id": "c", "text": "មិនល្អ ។", "reject": "misread"},
        {"id": "d", "text": "ចាម ជ្វា និងភូមា… ។"},
    ]
    out, dropped = training_rows(rows, words, banned={"post"})
    assert [r["id"] for r in out] == ["a", "a-km", "b"]
    assert dropped == {"misread": 1, "unreadable": 1}
    with pytest.raises(ValueError, match="held-out"):
        training_rows([{"id": "e", "text": "មើល poster នេះ ។"}], words, banned={"post"})
    with pytest.raises(ValueError, match="duplicate"):
        training_rows([rows[1], rows[1]], words, banned=set())


def test_strict_unseen_drops_sentences_whose_words_occur_in_khmer_script():
    from samleng.bench import Item
    from samleng.data import khmer_script_leaks, strict_unseen

    spellings = {"facebook": "ហ្វេសប៊ុក", "upload": "អាប់ឡូត", "ok": "", "qr": "ឃ្យូ"}
    texts = ["គាត់ចូលចិត្តលេង ហ្វេស ប៊ុក រាល់ថ្ងៃ ។", "ថ្ងៃនេះមេឃស្រឡះ ។"]
    leaked = khmer_script_leaks(spellings, texts)
    # Spaces are ignored; a missing spelling counts as leaked; short spellings need 3+ letters.
    assert leaked == {"facebook", "ok"}
    items = [
        Item("mixed-01", "mixed", "សូម upload រូបភាព"),
        Item("mixed-02", "mixed", "បើក Facebook មើល"),
        Item("mixed-03", "mixed", "សូម upload ទៅ Facebook"),
        Item("mixed-04", "mixed", "សូម upload ឯកសារ"),
    ]
    split = {"unseen": ["mixed-01", "mixed-02", "mixed-03"], "seen": ["mixed-04"]}
    assert strict_unseen(items, split, leaked) == {"mixed-01"}
