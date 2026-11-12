import numpy as np
import pandas as pd

from samleng.eval.guard import guard_flags, guard_table, signal_stats

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


def _clips(voice, seconds, lead=0.0, trail=0.5):
    """Two clips with a 2-d unit "speaker embedding" and the given timing."""
    return pd.DataFrame(
        {
            "id": ["a", "b"],
            "seconds": seconds,
            "lead_silence_s": lead,
            "trail_silence_s": trail,
            "clip_frac": 0.0,
            "e0": voice[0],
            "e1": voice[1],
        }
    )


def test_guard_table_measures_speech_only_pace_and_voice():
    guards = {
        "base": _clips((1.0, 0.0), [2.5, 4.5]),
        "teacher": _clips((0.0, 1.0), [1.5, 2.5], trail=0.1),
        # Same speech length as base times 0.87, but more trailing silence: pace must ignore it.
        "step2000": _clips((0.0, 1.0), [2.74, 4.48], trail=1.0),
    }
    table = guard_table(guards, voice="teacher", pace=0.87).set_index("system")
    assert table.loc["step2000", "dur_ratio_median"] == 0.87
    assert table.loc["step2000", "spk_median"] == 1.0
    assert table.loc["step2000", "flags"] == "ok"
    assert table.loc["base", "flags"] == table.loc["teacher", "flags"] == "reference"
    # Judged against base's voice and pace instead, the same clips drift on both.
    vs_base = guard_table(guards).set_index("system")
    assert vs_base.loc["step2000", "flags"] == "voice drift, speed drift"
