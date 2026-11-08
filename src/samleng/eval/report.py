import pandas as pd

from samleng.eval.cer import khmer_cer, normalize_khmer

# The LLM-based ASR sometimes loops; flag transcripts far longer than the reference.
LOOP_RATIO = 1.5


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Corpus-level CER and real-time factor per system and category.

    Expects columns: system, category, ref, hyp, seconds, gen_seconds.
    """
    rows = []
    for (system, category), g in df.groupby(["system", "category"]):
        rows.append(_row(system, category, g))
    for system, g in df.groupby("system"):
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
    per_item = [khmer_cer(r, h) for r, h in zip(refs, hyps, strict=True)]
    return {
        "system": system,
        "category": category,
        "n": len(g),
        "cer_%": round(100 * khmer_cer(refs, hyps), 1),
        "median_cer_%": round(100 * float(pd.Series(per_item).median()), 1),
        "asr_loops": sum(is_asr_loop(r, h) for r, h in zip(refs, hyps, strict=True)),
        "rtf": round(g["gen_seconds"].sum() / g["seconds"].sum(), 2),
        # Robust to slow clips when the machine is busy.
        "median_rtf": round(float((g["gen_seconds"] / g["seconds"]).median()), 2),
    }
