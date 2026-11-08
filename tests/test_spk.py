import numpy as np
import pytest

from samleng.eval.spk import consistency, to_mono_16k


def test_consistency_of_identical_embeddings_is_one():
    e = np.tile([[0.6, 0.8]], (3, 1))
    assert consistency(e) == pytest.approx(1.0)


def test_consistency_is_mean_pairwise_cosine_ignoring_scale():
    e = np.array([[1.0, 0.0], [0.0, 5.0], [2.0, 0.0]])
    # pairs: (0,1)=0, (0,2)=1, (1,2)=0
    assert consistency(e) == pytest.approx(1 / 3)


def test_consistency_needs_two_clips():
    with pytest.raises(ValueError):
        consistency(np.ones((1, 4)))


def test_to_mono_16k_downmixes_and_resamples():
    stereo = np.ones((24000, 2), dtype=np.float32)
    out = to_mono_16k(stereo, 24000)
    assert out.ndim == 1 and len(out) == 16000
