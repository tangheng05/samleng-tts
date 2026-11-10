import math

import pandas as pd

from samleng.eval.report import compare_to_base, spearman_ci, summarize


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


def test_compare_to_base_pairs_sentences():
    rows = [
        # base, then a checkpoint that fixes s1 and s2 and breaks nothing
        ("base", "s1", "pure", "abcd", "abxx"),
        ("base", "s2", "pure", "abcd", "abcx"),
        ("base", "s3", "pure", "abcd", "abcd"),
        ("step2", "s1", "pure", "abcd", "abcd"),
        ("step2", "s2", "pure", "abcd", "abcd"),
        ("step2", "s3", "pure", "abcd", "abcd"),
    ]
    df = pd.DataFrame(rows, columns=["system", "id", "category", "ref", "hyp"])
    out = compare_to_base(df, "base")
    row = out[out.category == "pure"].iloc[0]
    assert row.system == "step2"
    assert (row.n, row.better, row.worse, row.same) == (3, 2, 0, 1)
    assert row.median_cer_base_pct == 25.0
    assert row.median_cer_pct == 0.0
    assert set(out.category) == {"pure", "ALL"}


def test_spearman_ci_brackets_the_estimate():
    x = list(range(30))
    rho, lo, hi = spearman_ci(x, [-v for v in x])
    assert rho == -1.0 and lo == hi == -1.0
    noisy = [v + (7 if v % 3 == 0 else -7) for v in x]
    rho, lo, hi = spearman_ci(x, noisy)
    assert lo <= rho <= hi and 0 < rho < 1
    assert all(math.isnan(v) for v in spearman_ci([1, 1, 1], [1, 2, 3]))
