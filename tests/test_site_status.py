from datetime import UTC, datetime

import pandas as pd

from samleng.site_status import build_status, parse_train_log, step_scores

LOG = """\
\x1b[4m\x1b[1m > EPOCH: 3/29\x1b[0m
\x1b[1m   --> TIME: 2026-11-13 10:58:35 -- STEP: 299/598 -- GLOBAL_STEP: 2300\x1b[0m
     | > loss_mel: 19.5  (19.0)
     | > step_time: 4.0  (3.2)
\x1b[1m   --> TIME: 2026-11-13 11:00:10 -- STEP: 324/598 -- GLOBAL_STEP: 2325\x1b[0m
     | > loss_mel: 18.25  (18.9)
     | > step_time: 2.0  (3.1)
"""


def test_parse_train_log_reads_latest_position():
    p = parse_train_log(LOG)
    assert p == {
        "epoch": 3,
        "epochs": 30,
        "step_in_epoch": 324,
        "steps_per_epoch": 598,
        "step": 2325,
        "step_seconds": 3.0,
        "logged_at": "2026-11-13 11:00:10",
        "loss": [[2300, 19.5], [2325, 18.25]],
    }


def test_parse_train_log_empty():
    assert parse_train_log("no training yet") is None


def _results(path, rows):
    path.mkdir(parents=True)
    pd.DataFrame(rows, columns=["id", "category", "ref", "hyp"]).to_csv(
        path / "results.csv", index=False
    )


def test_step_scores_medians_per_category(tmp_path):
    _results(
        tmp_path / "2000",
        [
            ("a", "pure", "ក ខ គ ឃ", "ក ខ គ ឃ"),
            ("b", "pure", "ក ខ", "ក គ"),
            ("c", "numbers", "ក ខ", "ក ខ"),
            ("d", "english", "hello", "x"),
        ],
    )
    _results(tmp_path / "4000", [("a", "pure", "ក ខ", "ក ខ")])
    (tmp_path / "summary.md").write_text("not a step")
    scores = step_scores(tmp_path)
    assert [s["step"] for s in scores] == [2000, 4000]
    assert scores[0]["pure"] == 25.0
    assert scores[0]["numbers"] == 0.0
    # ALL leaves English out, like the bench summaries.
    assert scores[0]["all"] == 0.0
    assert scores[1] == {"step": 4000, "all": 0.0, "pure": 0.0}


def test_build_status_estimates_finish():
    now = datetime(2026, 11, 13, 4, 0, tzinfo=UTC)
    s = build_status(
        parse_train_log(LOG),
        scores=[{"step": 2000, "all": 11.4}],
        base={"all": 10.5},
        state="training",
        run="distill_v2_clear",
        now=now,
    )
    left = (598 - 324) + 26 * 598
    assert s["steps_left"] == left
    assert (
        s["finish_at"]
        == datetime.fromtimestamp(now.timestamp() + left * 3.0, UTC).isoformat()
    )
    assert s["progress"] == round((3 + 324 / 598) / 30, 4)
    assert s["best"] == {"step": 2000, "all": 11.4}
    assert s["updated_at"] == now.isoformat()


def test_build_status_without_training_has_no_finish():
    s = build_status(None, scores=[], base={}, state="idle", run=None)
    assert s["state"] == "idle"
    assert s["finish_at"] is None
    assert s["best"] is None


def test_recent_loss_keeps_newest_value_of_a_repeated_step():
    from samleng.site_status import recent_loss

    log = (
        "--> TIME: a b -- STEP: 1/9 -- GLOBAL_STEP: 100\n | > loss_mel: 30.0  (30)\n"
        "--> TIME: a b -- STEP: 2/9 -- GLOBAL_STEP: 125\n | > loss_mel: 29.0  (29)\n"
        "--> TIME: a b -- STEP: 1/9 -- GLOBAL_STEP: 100\n | > loss_mel: 28.0  (28)\n"
    )
    assert recent_loss(log) == [[100, 28.0], [125, 29.0]]


def test_job_info_reads_site_lines():
    from samleng.site_status import job_info

    script = "# Scores things.\n# site-kind: testing\n# site-en: Testing the model\n# site-progress: outputs/x/*/results.csv 17\nrun\n"
    assert job_info(script) == {
        "kind": "testing",
        "en": "Testing the model",
        "progress": "outputs/x/*/results.csv 17",
    }
    assert job_info("no headers") == {}
