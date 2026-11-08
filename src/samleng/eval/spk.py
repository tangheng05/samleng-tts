"""Speaker embeddings: does a system keep one voice across sentences?"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MODEL_ID = "microsoft/wavlm-base-plus-sv"
REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
SR = 16000


def to_mono_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)


def consistency(embs: np.ndarray) -> float:
    """Mean pairwise cosine similarity between clips (1.0 = one voice)."""
    if len(embs) < 2:
        raise ValueError("need at least two clips")
    unit = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sims = unit @ unit.T
    return float(sims[np.triu_indices(len(embs), k=1)].mean())


class SpeakerEncoder:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch, self.device = torch, device
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = WavLMForXVector.from_pretrained(MODEL_ID, revision=REVISION)
        self.model.to(device).eval()

    def embed(self, path: str | Path) -> np.ndarray:
        wav, sr = sf.read(path)
        x = self.fe(to_mono_16k(wav, sr), sampling_rate=SR, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**x.to(self.device)).embeddings
        return out[0].cpu().numpy()
