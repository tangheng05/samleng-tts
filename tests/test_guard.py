import numpy as np

from samleng.eval.guard import guard_flags, signal_stats

SR = 16000


def tone(seconds, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * 220 * t)


def test_signal_stats_measures_silence_and_clipping():
    y = np.concatenate([np.zeros(SR // 2), tone(1.0), np.zeros(SR // 4)])
    s = signal_stats(y, SR)
    assert abs(s["lead_silence_s"] - 0.5) < 0.03
    assert abs(s["trail_silence_s"] - 0.25) < 0.03
    assert s["clip_frac"] == 0
    loud = np.clip(tone(1.0, amp=2.0), -1, 1)
    assert signal_stats(loud, SR)["clip_frac"] > 0.1


def test_guard_flags_catch_voice_and_speed_drift():
    ok = {"spk_median": 0.97, "dur_ratio_median": 1.02, "clip_frac_p95": 0.0}
    base = {"spk_median": 0.98}
    assert guard_flags(ok, base) == []
    assert guard_flags(ok | {"spk_median": 0.93}, base) == ["voice drift"]
    assert guard_flags(ok | {"dur_ratio_median": 1.2}, base) == ["speed drift"]
    assert guard_flags(ok | {"clip_frac_p95": 0.01}, base) == ["clipping"]
    # A target pace moves the allowed window: 0.88x is fine when 0.88 is wanted.
    fast = ok | {"dur_ratio_median": 0.88}
    assert guard_flags(fast, base) == ["speed drift"]
    assert guard_flags(fast, base, pace=0.88) == []
    assert guard_flags(ok, base, pace=0.88) == ["speed drift"]
