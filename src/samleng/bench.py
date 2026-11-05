import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    text: str


def load_bench(path: str | Path) -> list[Item]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    items = [Item(r["id"], r["category"], r["text"]) for r in rows]
    if len({it.id for it in items}) != len(items):
        raise ValueError(f"duplicate ids in {path}")
    return items
