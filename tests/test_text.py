from samleng.text import verbalize


def test_khmer_digits_are_spelled_out():
    assert verbalize("២៥ នាក់") == "ម្ភៃប្រាំ នាក់"


def test_percent_is_spoken():
    assert "ភាគរយ" in verbalize("10%")


def test_no_tha_separator_leaks():
    assert "▁" not in verbalize("ឆ្នាំ 1998")
