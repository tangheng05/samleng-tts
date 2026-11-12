"""Learn where Khmer writers put phrase spaces, from Khmer Wikipedia (CC-BY-SA 3.0).

uv run python scripts/phrase_breaks.py            # train, test on held-out sentences, write the weights

A logistic regression over word boundaries: is there a space here, given the word before and
the word after? Writes texts/phrase_breaks.json for samleng.phrases and prints how well
`phrase_spaces` restores the spaces of held-out sentences, next to two simple baselines.
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from samleng.phrases import (
    MAX_RUN,
    MIN_SCORE,
    MODEL,
    OTHER,
    RUN,
    break_scores,
    context,
    split,
    words,
)

REPO, FILE = "wikimedia/wikipedia", "20231101.km/train-00000-of-00001.parquet"
REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SENTENCE = re.compile(r"(?<=[។៕?!])\s*|\n+")


def sentences(n: int, seed: int) -> list[str]:
    df = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset", revision=REV))
    seen, out = set(), []
    for text in df["text"]:
        for s in SENTENCE.split(text):
            s = unicodedata.normalize("NFC", s)
            s = re.sub("[\u200b\u200c\u200d\ufeff]", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            if 40 <= len(s) <= 300 and s.endswith("។") and s not in seen:
                seen.add(s)
                out.append(s)
    random.Random(seed).shuffle(out)
    return out[:n]


def runs(sentence: str) -> list[list[list[str]]]:
    """Stretches of Khmer words separated only by the writer's spaces: [[words of a piece]]."""
    out, cur = [], []
    for piece in sentence.split(" "):
        if RUN.fullmatch(piece):
            cur.append(words(piece))
        else:
            # Punctuation, digits or Latin end a stretch; a Khmer prefix or suffix is dropped.
            if len(cur) > 1:
                out.append(cur)
            cur = []
    if len(cur) > 1:
        out.append(cur)
    return out


def boundaries(run: list[list[str]]) -> tuple[list[str], list[int]]:
    """Flat words of a run and, per word boundary, whether the writer put a space there."""
    ws, labels = [], []
    for piece in run:
        labels += [0] * (len(piece) - 1) + [1]
        ws += piece
    return ws, labels[:-1]


def train(data: list[tuple[list[str], list[int]]], min_count: int, steps: int) -> dict:
    count = Counter(w for ws, _ in data for w in ws)
    vocab = [
        OTHER,
        *sorted(w for w, c in count.items() if c >= min_count),
        "<s>",
        "</s>",
    ]
    idx = {w: i for i, w in enumerate(vocab)}
    names = list(context(["a", "b"], 1))
    feats, y = {n: [] for n in names}, []
    for ws, labels in data:
        for i, label in enumerate(labels, 1):
            for n, w in context(ws, i).items():
                feats[n].append(idx.get(w, 0))
            y.append(label)
    feats = {n: torch.tensor(v) for n, v in feats.items()}
    y = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(0)
    w = {n: torch.zeros(len(vocab), requires_grad=True) for n in names}
    bias = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([*w.values(), bias], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        logits = bias + sum(w[n][feats[n]] for n in names)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        # A light pull toward zero, so rare words don't get extreme weights.
        (loss + 1e-5 * sum(t.pow(2).sum() for t in w.values())).backward()
        opt.step()
    print(
        f"{len(y)} boundaries, {int(y.sum())} spaces, {len(vocab)} words, loss {loss.item():.3f}"
    )

    def table(t):
        t = t.detach().tolist()
        kept = {v: round(x, 3) for v, x in zip(vocab, t, strict=True) if abs(x) >= 0.05}
        return kept | {OTHER: round(t[0], 3)}

    return {
        "source": f"{REPO} {FILE} @ {REV}",
        "bias": round(bias.item(), 3),
        "weights": {n: table(w[n]) for n in names},
    }


def positions(pieces: list[str]) -> set[int]:
    out, at = set(), 0
    for p in pieces[:-1]:
        at += len(p)
        out.add(at)
    return out


def test(data, model, max_run: int, min_score: float) -> str:
    rng = random.Random(0)
    systems = {
        "learned": lambda ws: break_scores(ws, model),
        "middle": None,
        "random": lambda ws: [rng.random() for _ in ws[1:]],
    }
    lines, n = [], 0
    for name, scorer in systems.items():
        tp = fp = fn = 0
        for ws, labels in data:
            total = sum(map(len, ws))
            if total <= max_run:
                continue
            n += name == "learned"
            if scorer is None:
                at, scores = 0, []
                for w in ws[:-1]:
                    at += len(w)
                    scores.append(-abs(at - total / 2))
            else:
                scores = scorer(ws)
            # The baselines have no notion of a believable break: they always split.
            floor = min_score if name == "learned" else -float("inf")
            pred = positions(split(ws, scores, max_run, min_score=floor))
            at, gold = 0, set()
            # One label per boundary: the last word has none.
            for w, label in zip(ws, labels, strict=False):
                at += len(w)
                if label:
                    gold.add(at)
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        lines.append(
            f"{name:8s} precision {p:.3f}  recall {r:.3f}  F1 {2 * p * r / max(p + r, 1e-9):.3f}"
            f"  ({tp + fp} spaces added, {tp + fn} written by the authors)"
        )
    return f"{n} held-out runs longer than {max_run} characters\n" + "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=33000, help="sentences (10% held out)")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="also try other settings")
    args = ap.parse_args()

    data = [boundaries(r) for s in sentences(args.n, args.seed) for r in runs(s)]
    cut = len(data) // 10
    held, rest = data[:cut], data[cut:]
    model = train(rest, args.min_count, args.steps)
    print(test(held, model, MAX_RUN, MIN_SCORE))
    if args.sweep:
        for max_run in (30, 40, 50):
            for floor in (-99, -2.5, -1.5, -0.5, 0.5):
                line = test(held, model, max_run, floor).splitlines()[1]
                print(f"max_run {max_run} min_score {floor:5}: {line}")
    MODEL.write_text(
        json.dumps(model, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    sizes = {n: len(t) for n, t in model["weights"].items()}
    print(f"word weights {sizes} -> {MODEL}")


if __name__ == "__main__":
    main()
