from samleng.notify import checkpoint_message


def rows(step, all_, unseen, numbers=2.6, pure=4.8):
    def r(cat, v):
        return {
            "system": f"step{step}",
            "category": cat,
            "median_cer_base_pct": {"ALL": 10.4, "mixed_unseen": 36.7}.get(cat, 8.0),
            "median_cer_pct": v,
        }

    return [
        r("ALL", all_),
        r("mixed_unseen", unseen),
        r("numbers", numbers),
        r("pure", pure),
    ]


def test_checkpoint_message_compares_with_best_so_far():
    table = rows(10000, 7.1, 24.0) + rows(12000, 6.5, 23.1)
    guards = {
        "step12000": {"spk_median": 0.97, "dur_ratio_median": 0.92, "flags": "ok"}
    }
    title, body = checkpoint_message("v1-A", 12000, table, guards)
    assert title == "v1-A step 12,000: better than the best so far"
    assert "6.5% (start 10.4%, best before 7.1% at step 10,000)" in body
    assert "Never-seen English: 23.1% (start 36.7%)" in body
    assert "speed 0.92×" in body


def test_checkpoint_message_flags_worse_and_guardrails():
    table = rows(10000, 7.1, 24.0) + rows(12000, 8.0, 26.0)
    guards = {
        "step12000": {
            "spk_median": 0.9,
            "dur_ratio_median": 0.8,
            "flags": "speed drift",
        }
    }
    title, body = checkpoint_message("v1-A", 12000, table, guards)
    assert "worse than the best so far" in title
    assert "⚠️ speed drift" in body


def test_checkpoint_message_first_step_has_no_best():
    title, body = checkpoint_message("v1-B", 1000, rows(1000, 9.0, 30.0), {})
    assert title == "v1-B step 1,000: first score"
    assert "best before" not in body
