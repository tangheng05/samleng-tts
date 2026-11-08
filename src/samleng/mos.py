import hashlib
import itertools
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

from samleng.bench import Item
from samleng.runners import read_jsonl

PAGE = Path(__file__).with_name("mos_page") / "index.html"


@dataclass(frozen=True)
class Clip:
    clip: str
    system: str
    item_id: str
    text: str
    audio: Path


def build_clips(items: list[Item], systems: dict[str, Path]) -> list[Clip]:
    clips = []
    for system, folder in systems.items():
        for it in items:
            wav = Path(folder) / f"{it.id}.wav"
            if not wav.exists():
                raise FileNotFoundError(wav)
            # Opaque id so the page and URLs never reveal which system made a clip.
            cid = hashlib.sha1(f"{system}/{it.id}".encode()).hexdigest()[:10]
            clips.append(Clip(cid, system, it.id, it.text, wav))
    return clips


def order_for(rater: str, clips: list[Clip]) -> list[Clip]:
    seed = int(hashlib.sha256(rater.encode()).hexdigest(), 16)
    out = sorted(clips, key=lambda c: c.clip)
    random.Random(seed).shuffle(out)
    return out


def append_rating(path: Path, rater: str, clip: Clip, score: int) -> None:
    if score not in range(1, 6):
        raise ValueError(f"score must be 1-5, got {score}")
    row = {
        "rater": rater,
        "clip": clip.clip,
        "system": clip.system,
        "item_id": clip.item_id,
        "score": score,
        "time": round(time.time(), 3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_ratings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    latest = {(r["rater"], r["clip"]): r for r in read_jsonl(path)}
    return list(latest.values())


def merge_systems(ratings: list[dict], groups: dict[str, list[str]]) -> list[dict]:
    rename = {old: new for new, olds in groups.items() for old in olds}
    return [{**r, "system": rename.get(r["system"], r["system"])} for r in ratings]


def _sentence_means(ratings: list[dict]) -> dict[str, dict[str, float]]:
    scores = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        scores[r["system"]][r["item_id"]].append(r["score"])
    return {
        system: {item: statistics.mean(s) for item, s in items.items()}
        for system, items in scores.items()
    }


def summarize(ratings: list[dict]) -> list[dict]:
    """MOS per system; the CI is over sentence means, since scores share a sentence."""
    means = _sentence_means(ratings)
    rows = []
    for system, by_item in sorted(means.items()):
        m = list(by_item.values())
        mine = [r for r in ratings if r["system"] == system]
        ci = math.nan
        if len(m) > 1:
            ci = (
                stats.t.ppf(0.975, len(m) - 1) * statistics.stdev(m) / math.sqrt(len(m))
            )
        rows.append(
            {
                "system": system,
                "mos": round(statistics.mean(m), 2),
                "ci95": round(ci, 2),
                "sentences": len(m),
                "ratings": len(mine),
                "raters": len({r["rater"] for r in mine}),
            }
        )
    return rows


def compare(ratings: list[dict]) -> list[dict]:
    """Paired Wilcoxon signed-rank test per pair of systems, over shared sentences."""
    means = _sentence_means(ratings)
    rows = []
    for a, b in itertools.combinations(sorted(means), 2):
        shared = sorted(means[a].keys() & means[b].keys())
        if not shared:
            continue
        d = [means[a][i] - means[b][i] for i in shared]
        p = stats.wilcoxon(d).pvalue if any(d) else 1.0
        rows.append(
            {
                "a": a,
                "b": b,
                "sentences": len(shared),
                "a_higher": sum(x > 0 for x in d),
                "ties": sum(x == 0 for x in d),
                "b_higher": sum(x < 0 for x in d),
                "p": float(p),
            }
        )
    return rows
