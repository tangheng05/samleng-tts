import pandas as pd
import pytest

from samleng.eval.english import english_wer
from samleng.eval.report import summarize


def test_wer_ignores_case_punctuation_and_number_format():
    assert (
        english_wer(
            "Great job! You got all five questions right.",
            "great job you got all 5 questions right",
        )
        == 0
    )


def test_wer_counts_word_errors():
    assert english_wer("the cat is sleeping", "the dog is sleeping") == pytest.approx(
        0.25
    )


def test_summary_scores_english_by_wer_and_keeps_it_out_of_khmer_all():
    df = pd.DataFrame(
        [
            {
                "system": "s",
                "category": "pure",
                "ref": "សួស្តី",
                "hyp": "សួស្តី",
                "seconds": 1.0,
                "gen_seconds": 1.0,
            },
            {
                "system": "s",
                "category": "english",
                "ref": "the cat",
                "hyp": "the dog",
                "seconds": 1.0,
                "gen_seconds": 1.0,
            },
        ]
    )
    rows = summarize(df).set_index("category")
    assert rows.loc["english", "wer_%"] == 50.0
    assert pd.isna(rows.loc["english", "cer_%"])
    assert rows.loc["ALL", "n"] == 1 and rows.loc["ALL", "cer_%"] == 0.0


def test_english_summary_reports_median_and_counts_whisper_loops():
    looped = " ".join(["them"] * 40)
    rows = [
        {"ref": "the cat sat", "hyp": "the cat sat"},
        {"ref": "a big dog", "hyp": "a big dog"},
        {"ref": "good morning", "hyp": looped},
    ]
    df = pd.DataFrame(
        [
            {"system": "s", "category": "english", "seconds": 1.0, "gen_seconds": 1.0}
            | r
            for r in rows
        ]
    )
    row = summarize(df).set_index("category").loc["english"]
    assert row["median_wer_%"] == 0.0
    assert row["asr_loops"] == 1
    assert row["wer_%"] > 100
