"""Guardrail checks next to CER: a checkpoint must not buy lower CER with a drifting voice or speed."""

import numpy as np
import pandas as pd

# Frames quieter than this (dBFS) count as silence at the clip edges.
SILENCE_DB = -45.0
FRAME_S = 0.02
# Flag limits vs the base model (medians over the bench sentences).
MAX_SPK_DROP = 0.03
DUR_RATIO_RANGE = (0.9, 1.1)
MAX_CLIP_FRAC = 0.001


def signal_stats(y: np.ndarray, sr: int) -> dict:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = max(1, int(sr * FRAME_S))
    frames = y[: len(y) // n * n].reshape(-1, n) if len(y) >= n else y.reshape(1, -1)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)
    loud = np.flatnonzero(db > SILENCE_DB)
    lead = loud[0] if len(loud) else len(db)
    trail = len(db) - 1 - loud[-1] if len(loud) else len(db)
    return {
        "lead_silence_s": round(lead * FRAME_S, 3),
        "trail_silence_s": round(trail * FRAME_S, 3),
        "clip_frac": float((np.abs(y) >= 0.999).mean()),
    }


def guard_flags(step: dict, voice: dict, pace: float = 1.0) -> list[str]:
    """Which guardrails a checkpoint's summary breaks.

    `voice` is the summary of the target voice's own clips; `pace` is the wanted duration
    relative to base (1.0 = base's pace).
    """
    flags = []
    if step["spk_median"] < voice["spk_median"] - MAX_SPK_DROP:
        flags.append("voice drift")
    low, high = (pace * r for r in DUR_RATIO_RANGE)
    if not low <= step["dur_ratio_median"] <= high:
        flags.append("speed drift")
    if step["clip_frac_p95"] > MAX_CLIP_FRAC:
        flags.append("clipping")
    return flags


def speech_seconds(g: pd.DataFrame) -> pd.Series:
    """Seconds of speech per clip: its length without the silence at both ends."""
    return g["seconds"] - g["lead_silence_s"] - g["trail_silence_s"]


def guard_table(
    guards: dict[str, pd.DataFrame], voice: str = "base", pace: float = 1.0
) -> pd.DataFrame:
    """One row per system: voice similarity to `voice`, speech duration vs base, silence, clipping.

    Each frame has one row per clip: id, seconds, the signal_stats columns and a unit speaker
    embedding in columns e0..eN. `guards` must hold "base" and `voice`.
    """
    base = guards["base"].set_index("id")
    cols = [c for c in base.columns if c.startswith("e") and c[1:].isdigit()]
    profile = guards[voice][cols].to_numpy().mean(axis=0)
    profile /= np.linalg.norm(profile)
    rows = []
    for system, g in guards.items():
        g = g.set_index("id")
        ratio = speech_seconds(g) / speech_seconds(base).loc[g.index]
        rows.append(
            {
                "system": system,
                "spk_median": round(float(np.median(g[cols].to_numpy() @ profile)), 3),
                "dur_ratio_median": round(float(ratio.median()), 3),
                "lead_silence_s": round(float(g["lead_silence_s"].median()), 2),
                "trail_silence_s": round(float(g["trail_silence_s"].median()), 2),
                "clip_frac_p95": round(float(g["clip_frac"].quantile(0.95)), 5),
            }
        )
    out = pd.DataFrame(rows)
    target = out[out["system"] == voice].iloc[0].to_dict()
    out["flags"] = [
        "reference"
        if r["system"] in ("base", voice)
        else ", ".join(guard_flags(r, target, pace)) or "ok"
        for r in out.to_dict("records")
    ]
    return out
