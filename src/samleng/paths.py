"""Where data and outputs live; override with SAMLENG_DATA / SAMLENG_OUTPUT."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Override on the training machine to point at an external SSD.
DATA_DIR = Path(os.environ.get("SAMLENG_DATA", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("SAMLENG_OUTPUT", ROOT / "outputs"))
