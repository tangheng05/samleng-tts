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


def test_unknown_category_is_rejected(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('{"id":"a","category":"name","text":"x"}\n')
    with pytest.raises(ValueError, match="name"):
        load_bench(p)


def test_bench_v1_has_320_reviewed_sentences_in_planned_split():
    import json
    from collections import Counter

    path = ROOT / "bench" / "khmertts_bench_v1.jsonl"
    items = load_bench(path)
    assert Counter(it.category for it in items) == {
        "pure": 80,
        "numbers": 60,
        "mixed": 70,
        "names": 50,
        "long": 40,
        "english": 20,
    }
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(r["reviewed"] for r in rows)
    pilot = {it.id: it.text for it in load_bench(ROOT / "bench" / "pilot_v1.jsonl")}
    v1 = {it.id: it.text for it in items}
    assert all(v1[k] == text for k, text in pilot.items())
