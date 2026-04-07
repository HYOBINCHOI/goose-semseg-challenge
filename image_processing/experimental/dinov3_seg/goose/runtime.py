from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PROCESSING_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = IMAGE_PROCESSING_ROOT / "scripts"
TOOLS_DIR = IMAGE_PROCESSING_ROOT / "tools"


def ensure_project_paths() -> None:
    for path in (EXPERIMENT_ROOT, IMAGE_PROCESSING_ROOT, SCRIPTS_DIR, TOOLS_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
