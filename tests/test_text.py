from samleng.text import verbalize


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
