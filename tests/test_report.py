import pandas as pd

from samleng.eval.report import summarize


def test_summarize_per_category_and_overall():
    df = pd.DataFrame(
        {
            "system": ["a", "a"],
            "category": ["pure", "mixed"],
            "ref": ["ab", "cd"],
            "hyp": ["ab", "cx"],
            "seconds": [2.0, 2.0],
            "gen_seconds": [1.0, 1.0],
        }
    )
    s = summarize(df).set_index("category")
    assert s.loc["pure", "cer_%"] == 0.0
    assert s.loc["mixed", "cer_%"] == 50.0
    assert s.loc["ALL", "cer_%"] == 25.0
    assert s.loc["ALL", "rtf"] == 0.5
    assert s.loc["ALL", "median_rtf"] == 0.5


def test_merge_keeps_other_systems_and_replaces_rerun():
    from samleng.eval.report import merge_results

    old = pd.DataFrame({"system": ["a", "a", "b"], "hyp": ["old", "old", "keep"]})
    new = pd.DataFrame({"system": ["a"], "hyp": ["new"]})
    m = merge_results(old, new)
    assert sorted(m["hyp"]) == ["keep", "new"]


def test_asr_loop_is_flagged():
    from samleng.eval.report import is_asr_loop

    assert is_asr_loop("ខ្ញុំសុំទោស", "ខ្ញុំសុំទោស ខ្ញុំសុំទោស")
    assert not is_asr_loop("ខ្ញុំសុំទោស", "ខ្ញុំសុំទោ")
