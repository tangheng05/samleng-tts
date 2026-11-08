import numpy as np


def normalize_loudness(y: np.ndarray, sr: int, target_db: float = -26.0) -> np.ndarray:
    """Scale speech to a target active level (ITU P.56 style), never past -0.1 dBFS peak."""
    frame = max(1, sr // 50)
    n = len(y) // frame * frame
    energy = (y[:n].reshape(-1, frame) ** 2).mean(axis=1)
    if not n or energy.max() <= 0:
        return y
    # Frames within 40 dB of the loudest one count as speech.
    active = energy[energy > energy.max() * 1e-4]
    level_db = 10 * np.log10(active.mean())
    gain = 10 ** ((target_db - level_db) / 20)
    peak = np.abs(y).max()
    return y * min(gain, 0.99 / peak)
