import math

import pytest

from samleng.bench import Item
from samleng.mos import (
    append_rating,
    build_clips,
    compare,
    load_ratings,
    merge_systems,
    order_for,
    summarize,
)

ITEMS = [Item(f"mixed-0{i}", "mixed", f"text {i}") for i in range(1, 4)]


def make_systems(tmp_path, names=("a", "b")):
    systems = {}
    for name in names:
        d = tmp_path / name
        d.mkdir()
        for it in ITEMS:
            (d / f"{it.id}.wav").write_bytes(b"RIFF")
        systems[name] = d
    return systems


def test_build_clips_hides_system(tmp_path):
    clips = build_clips(ITEMS, make_systems(tmp_path, ("mms_raw", "voxcpm2_raw")))
    assert len(clips) == 6
    assert len({c.clip for c in clips}) == 6
    assert not any("mms" in c.clip or "vox" in c.clip for c in clips)


def test_build_clips_missing_wav(tmp_path):
    systems = make_systems(tmp_path)
    (systems["a"] / "mixed-02.wav").unlink()
    with pytest.raises(FileNotFoundError):
        build_clips(ITEMS, systems)


def test_order_is_stable_per_rater(tmp_path):
    clips = build_clips(ITEMS, make_systems(tmp_path))
    first = [c.clip for c in order_for("heng", clips)]
    assert first == [c.clip for c in order_for("heng", list(reversed(clips)))]
    assert sorted(first) == sorted(c.clip for c in clips)
    assert first != [c.clip for c in order_for("dara", clips)]


def test_latest_rating_wins(tmp_path):
    clips = build_clips(ITEMS, make_systems(tmp_path))
    path = tmp_path / "ratings.jsonl"
    append_rating(path, "heng", clips[0], 2)
    append_rating(path, "heng", clips[0], 4)
    append_rating(path, "dara", clips[0], 3)
    rows = load_ratings(path)
    assert sorted((r["rater"], r["score"]) for r in rows) == [("dara", 3), ("heng", 4)]


def test_rating_out_of_range(tmp_path):
    clips = build_clips(ITEMS, make_systems(tmp_path))
    with pytest.raises(ValueError):
        append_rating(tmp_path / "r.jsonl", "heng", clips[0], 6)


def r(system, item, score, rater="x"):
    return {"rater": rater, "system": system, "item_id": item, "score": score}


def test_summarize_averages_per_sentence_first():
    ratings = [
        r("a", "s1", 5),
        r("a", "s1", 3, rater="y"),
        r("a", "s2", 2),
        r("b", "s1", 5),
    ]
    a, b = summarize(ratings)
    assert (a["system"], a["mos"], a["sentences"], a["ratings"], a["raters"]) == (
        "a",
        3.0,
        2,
        3,
        2,
    )
    # sentence means 4 and 2: sd = sqrt(2), t(0.975, df=1) = 12.706
    assert a["ci95"] == pytest.approx(12.706 * math.sqrt(2) / math.sqrt(2), abs=0.01)
    assert b["mos"] == 5.0 and math.isnan(b["ci95"])


def test_compare_is_paired_by_sentence():
    # s0 has no "b" score; the other 5 differ by 4, 3, 2, 1, 0.5
    ratings = [r("a", f"s{i}", 5) for i in range(5)]
    ratings += [r("a", "s5", 4), r("a", "s5", 5, rater="y")]
    ratings += [r("b", f"s{i}", i) for i in range(1, 5)] + [r("b", "s5", 4)]
    (row,) = compare(ratings)
    assert (row["a"], row["b"], row["sentences"]) == ("a", "b", 5)
    assert (row["a_higher"], row["ties"], row["b_higher"]) == (5, 0, 0)
    assert row["p"] == pytest.approx(0.0625)


def test_compare_all_ties():
    (row,) = compare([r("a", "s1", 3), r("b", "s1", 3)])
    assert row["ties"] == 1 and row["p"] == 1.0


def test_merge_systems():
    merged = merge_systems(
        [r("a1", "s1", 2), r("a2", "s1", 4), r("b", "s1", 1)], {"a": ["a1", "a2"]}
    )
    assert [m["system"] for m in merged] == ["a", "a", "b"]
    assert summarize(merged)[0]["mos"] == 3.0


def _active_db(y):
    import numpy as np

    return 20 * np.log10(np.sqrt(np.mean(y[np.abs(y) > 0] ** 2)))


def test_normalize_loudness_hits_target():
    import numpy as np

    from samleng.mos import normalize_loudness

    sr = 16000
    tone = 0.02 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    y = np.concatenate([np.zeros(sr), tone, np.zeros(sr)])
    out = normalize_loudness(y, sr, target_db=-26.0)
    assert _active_db(out[sr : 2 * sr]) == pytest.approx(-26.0, abs=0.5)


def test_normalize_loudness_never_clips():
    import numpy as np

    from samleng.mos import normalize_loudness

    sr = 16000
    y = np.full(sr, 0.001)
    y[100] = 0.9
    assert np.abs(normalize_loudness(y, sr, target_db=-10.0)).max() <= 0.99
