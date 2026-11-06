import pytest

from samleng.bench import load_bench
from samleng.paths import ROOT


def test_pilot_bench_loads_with_all_categories():
    items = load_bench(ROOT / "bench" / "pilot_v0.jsonl")
    assert {it.category for it in items} == {"pure", "numbers", "mixed", "long"}


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('{"id":"a","category":"pure","text":"x"}\n' * 2)
    with pytest.raises(ValueError):
        load_bench(p)


def test_pilot_v1_has_100_sentences_in_planned_split():
    from collections import Counter

    items = load_bench(ROOT / "bench" / "pilot_v1.jsonl")
    counts = Counter(it.category for it in items)
    assert counts == {"pure": 40, "long": 10, "numbers": 25, "mixed": 25}
