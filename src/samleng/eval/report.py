"""Summary tables: error rates per system and category, and paired comparisons."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from samleng.eval.cer import khmer_cer, normalize_khmer
from samleng.eval.english import english_wer

# The LLM-based ASR sometimes loops; flag transcripts far longer than the reference.
LOOP_RATIO = 1.5


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Corpus-level CER (WER for `english`) and real-time factor per system and category.

    Expects columns: system, category, ref, hyp, seconds, gen_seconds.
    ALL covers the Khmer categories only, so it stays comparable across bench versions.
    """
    rows = []
    for (system, category), g in df.groupby(["system", "category"]):
        rows.append(_row(system, category, g))
    for system, g in df[df["category"] != "english"].groupby("system"):
        rows.append(_row(system, "ALL", g))
    return pd.DataFrame(rows)


def merge_results(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Replace the systems in `new`, keep every other system's earlier rows."""
    kept = old[~old["system"].isin(new["system"].unique())]
    return pd.concat([kept, new], ignore_index=True)


def is_asr_loop(ref: str, hyp: str) -> bool:
    return len(normalize_khmer(hyp)) > LOOP_RATIO * len(normalize_khmer(ref))


def _row(system: str, category: str, g: pd.DataFrame) -> dict:
    refs, hyps = g["ref"].tolist(), g["hyp"].fillna("").tolist()
    timing = {
        "rtf": round(g["gen_seconds"].sum() / g["seconds"].sum(), 2),
        # Robust to slow clips when the machine is busy.
        "median_rtf": round(float((g["gen_seconds"] / g["seconds"]).median()), 2),
    }
    if category == "english":
        per_item = [english_wer(r, h) for r, h in zip(refs, hyps, strict=True)]
        return {
            "system": system,
            "category": category,
            "n": len(g),
            "wer_%": round(100 * english_wer(refs, hyps), 1),
            "median_wer_%": round(100 * float(pd.Series(per_item).median()), 1),
            # Whisper loops too, e.g. on garbled audio.
            "asr_loops": sum(
                len(h.split()) > LOOP_RATIO * len(r.split())
                for r, h in zip(refs, hyps, strict=True)
            ),
            **timing,
        }
    per_item = [khmer_cer(r, h) for r, h in zip(refs, hyps, strict=True)]
    return {
        "system": system,
        "category": category,
        "n": len(g),
        "cer_%": round(100 * khmer_cer(refs, hyps), 1),
        "median_cer_%": round(100 * float(pd.Series(per_item).median()), 1),
        "asr_loops": sum(is_asr_loop(r, h) for r, h in zip(refs, hyps, strict=True)),
        **timing,
    }


def compare_to_base(df: pd.DataFrame, base: str, metric=khmer_cer) -> pd.DataFrame:
    """Per-sentence CER of each system vs `base` on the same sentences, per category and ALL.

    Expects columns: system, id, category, ref, hyp. p is a paired Wilcoxon test.
    """
    df = df.assign(
        cer=[metric(r, h) for r, h in zip(df["ref"], df["hyp"].fillna(""), strict=True)]
    )
    ref = df[df["system"] == base].set_index("id")["cer"]
    rows = []
    for system, g in df[df["system"] != base].groupby("system"):
        g = g[g["id"].isin(ref.index)]
        groups = [*g.groupby("category"), ("ALL", g)]
        for category, c in groups:
            new, old = c.set_index("id")["cer"], ref[c["id"]]
            diff = (new - old).round(6)
            nonzero = diff[diff != 0]
            rows.append(
                {
                    "system": system,
                    "category": category,
                    "n": len(c),
                    "median_cer_base_pct": round(100 * float(old.median()), 1),
                    "median_cer_pct": round(100 * float(new.median()), 1),
                    "better": int((diff < 0).sum()),
                    "worse": int((diff > 0).sum()),
                    "same": int((diff == 0).sum()),
                    "p": float(wilcoxon(nonzero).pvalue) if len(nonzero) > 1 else 1.0,
                }
            )
    return pd.DataFrame(rows)


def spearman_ci(x, y, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Spearman rho with a 95% bootstrap CI over paired items."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), float("nan")
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) and np.ptp(y[i]):
            boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(rho, 3), round(float(lo), 3), round(float(hi), 3)


def seed_average(frames: list[pd.DataFrame], metric=khmer_cer) -> pd.DataFrame:
    """Per-sentence error averaged over runs of one system with different seeds.

    Each frame has id, category, ref, hyp. Returns id, category, cer (the mean over frames).
    One seed moves the all-Khmer median by about a point, so compare systems on this.
    """
    cers = [
        pd.Series(
            [metric(r, h) for r, h in zip(f["ref"], f["hyp"].fillna(""), strict=True)],
            index=f["id"],
        )
        for f in frames
    ]
    first = frames[0].set_index("id")
    mean = pd.concat(cers, axis=1).loc[first.index].mean(axis=1)
    return pd.DataFrame(
        {
            "id": first.index,
            "category": first["category"].to_numpy(),
            "cer": mean.to_numpy(),
        }
    )


def category_medians(avg: pd.DataFrame) -> dict[str, float]:
    """Median error (%) per category and for ALL, from a seed_average frame."""
    out = {c: g["cer"].median() for c, g in avg.groupby("category")}
    return {
        k: round(100 * float(v), 1)
        for k, v in (out | {"ALL": avg["cer"].median()}).items()
    }


def paired(a: pd.DataFrame, b: pd.DataFrame, category: str | None = None) -> dict:
    """Is system a better than b on the same sentences? (seed_average frames, Wilcoxon)."""
    if category:
        a, b = a[a["category"] == category], b[b["category"] == category]
    x, y = a.set_index("id")["cer"], b.set_index("id")["cer"]
    diff = (x - y[x.index]).round(6)
    nonzero = diff[diff != 0]
    return {
        "n": len(diff),
        "better": int((diff < 0).sum()),
        "worse": int((diff > 0).sum()),
        "p": float(wilcoxon(nonzero).pvalue) if len(nonzero) > 1 else 1.0,
    }
