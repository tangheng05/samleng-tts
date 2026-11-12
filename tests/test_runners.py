import pytest

from samleng.runners import same_settings


def test_same_settings_guards_a_cached_folder(tmp_path):
    out = tmp_path / "2000"
    same_settings(out, {"synth": ["--length-scale", "1.3"]})
    same_settings(out, {"synth": ["--length-scale", "1.3"]})
    with pytest.raises(SystemExit):
        same_settings(out, {"synth": ["--length-scale", "1.2"]})


def test_same_settings_accepts_folders_from_before_the_check(tmp_path):
    (tmp_path / "results.csv").write_text("id\n")
    same_settings(tmp_path, {"synth": []})
    assert not (tmp_path / "settings.json").exists()
