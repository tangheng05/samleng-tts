"""Benchmark files: one JSON line per sentence with id, category and text."""

import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = {"pure", "numbers", "mixed", "long", "names", "english"}


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    text: str


def load_bench(path: str | Path) -> list[Item]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    items = [Item(r["id"], r["category"], r["text"]) for r in rows]
    unknown = {it.category for it in items} - CATEGORIES
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)} in {path}")
    if len({it.id for it in items}) != len(items):
        raise ValueError(f"duplicate ids in {path}")
    return items
