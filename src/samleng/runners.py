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
