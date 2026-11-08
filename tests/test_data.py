import numpy as np

from samleng.data import (
    TARGET_SR,
    LeakCheck,
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
