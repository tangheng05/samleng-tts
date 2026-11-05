from samleng.paths import DATA_DIR, OUTPUT_DIR, ROOT


def test_default_paths_live_under_repo():
    assert (ROOT / "pyproject.toml").exists()
    assert DATA_DIR.name == "data" or DATA_DIR.is_absolute()
    assert OUTPUT_DIR.name == "outputs" or OUTPUT_DIR.is_absolute()
