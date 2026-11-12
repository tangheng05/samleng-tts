"""Call model runners (isolated environments under runners/) through JSONL files."""

import json
import subprocess
from pathlib import Path

from samleng.paths import ROOT

RUNNERS = ROOT / "runners"


def run(name: str, script: str, *args: str | Path) -> None:
    """Run a script inside an isolated runner env (see runners/README.md)."""
    project = RUNNERS / name
    cmd = ["uv", "run", "--project", str(project), "python", str(project / script)]
    subprocess.run([*cmd, *map(str, args)], check=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def same_settings(folder: Path, settings: dict) -> None:
    """Record how a cached output folder is produced; refuse to reuse it for other settings.

    Folders from before this check (no settings.json) are accepted as they are.
    """
    f = folder / "settings.json"
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old != settings:
            raise SystemExit(
                f"{folder} was made with {old}, not {settings}: delete it or use another --tag"
            )
    elif not any(folder.glob("results.*")):
        folder.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
