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


def shuffle_for(rater: str, things: list, key) -> list:
    """A per-rater order: different for each rater, the same every time they return."""
    seed = int(hashlib.sha256(rater.encode()).hexdigest(), 16)
    out = sorted(things, key=key)
    random.Random(seed).shuffle(out)
    return out


def order_for(rater: str, clips: list[Clip]) -> list[Clip]:
    return shuffle_for(rater, clips, key=lambda c: c.clip)


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


def load_ratings(path: Path, key: str = "clip") -> list[dict]:
    """Ratings with only each rater's latest answer per clip (or per pair)."""
    if not path.exists():
        return []
    latest = {(r["rater"], r[key]): r for r in read_jsonl(path)}
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


# Paired comparison (CCR, ITU-T P.800): "the second sample compared to the first", -3..+3.
CCR = range(-3, 4)


@dataclass(frozen=True)
class Pair:
    pair: str
    item_id: str
    text: str
    a: str
    b: str
    first: Clip
    second: Clip
    check: bool = False


def _seeded(*parts: str) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16))


def build_pairs(
    items: list[Item], a: tuple[str, Path], b: tuple[str, Path], checks: int = 4
) -> list[Pair]:
    """One A-vs-B pair per sentence, sides shuffled, plus `checks` pairs of a clip with itself.

    A careful rater scores the identical pairs near 0 (attention check).
    """
    clips_a = {c.item_id: c for c in build_clips(items, {a[0]: a[1]})}
    clips_b = {c.item_id: c for c in build_clips(items, {b[0]: b[1]})}
    pairs = []
    for it in items:
        ca, cb = clips_a[it.id], clips_b[it.id]
        first, second = (
            (ca, cb) if _seeded(a[0], b[0], it.id).random() < 0.5 else (cb, ca)
        )
        pid = hashlib.sha1(f"{a[0]}|{b[0]}|{it.id}".encode()).hexdigest()[:10]
        pairs.append(Pair(pid, it.id, it.text, a[0], b[0], first, second))
    for it in _seeded("checks", a[0], b[0]).sample(items, min(checks, len(items))):
        c = clips_a[it.id]
        pid = hashlib.sha1(f"check|{a[0]}|{b[0]}|{it.id}".encode()).hexdigest()[:10]
        pairs.append(Pair(pid, it.id, it.text, a[0], a[0], c, c, check=True))
    return pairs


def append_ccr(path: Path, rater: str, pair: Pair, second_vs_first: int) -> None:
    if second_vs_first not in CCR:
        raise ValueError(f"score must be -3..3, got {second_vs_first}")
    a_vs_b = second_vs_first if pair.second.system == pair.a else -second_vs_first
    row = {
        "rater": rater,
        "pair": pair.pair,
        "a": pair.a,
        "b": pair.b,
        "item_id": pair.item_id,
        "check": pair.check,
        "raw": second_vs_first,
        "score": a_vs_b,
        "time": round(time.time(), 3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def careless_raters(ratings: list[dict], max_mean_abs: float = 1.0) -> set[str]:
    """Raters who heard a difference between identical clips (mean |score| above the limit)."""
    checks = defaultdict(list)
    for r in ratings:
        if r["check"]:
            checks[r["rater"]].append(abs(r["raw"]))
    return {rater for rater, s in checks.items() if statistics.mean(s) > max_mean_abs}


def summarize_ccr(ratings: list[dict]) -> list[dict]:
    """CMOS (A minus B) per system pair, averaged per sentence first; p tests CMOS = 0."""
    by_pair = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        if not r["check"]:
            by_pair[(r["a"], r["b"])][r["item_id"]].append(r["score"])
    rows = []
    for (a, b), items in sorted(by_pair.items()):
        m = [statistics.mean(s) for s in items.values()]
        ci = math.nan
        if len(m) > 1:
            ci = (
                stats.t.ppf(0.975, len(m) - 1) * statistics.stdev(m) / math.sqrt(len(m))
            )
        mine = [r for r in ratings if (r["a"], r["b"]) == (a, b) and not r["check"]]
        rows.append(
            {
                "a": a,
                "b": b,
                "cmos": round(statistics.mean(m), 2),
                "ci95": round(ci, 2),
                "sentences": len(m),
                "a_better": sum(x > 0 for x in m),
                "ties": sum(x == 0 for x in m),
                "b_better": sum(x < 0 for x in m),
                "p": float(stats.wilcoxon(m).pvalue) if any(m) else 1.0,
                "ratings": len(mine),
                "raters": len({r["rater"] for r in mine}),
            }
        )
    return rows
