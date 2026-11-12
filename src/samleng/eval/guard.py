"""Guardrail checks next to CER: a checkpoint must not buy lower CER with a drifting voice or speed."""

import numpy as np

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
