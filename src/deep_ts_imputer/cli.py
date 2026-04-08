"""Console-script entry points exposed via ``pyproject.toml``.

These are thin wrappers around the ``scripts/*.py`` modules so that after
``pip install -e .`` users can run ``dti-train``, ``dti-tune`` and
``dti-reconstruct`` from anywhere.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _run(script_name: str) -> None:
    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")


def train_main() -> None:
    _run("train.py")


def tune_main() -> None:
    _run("tune.py")


def reconstruct_main() -> None:
    _run("reconstruct.py")


def analyze_main() -> None:
    _run("analyze.py")


def grid_main() -> None:
    _run("run_grid.py")


def progressive_main() -> None:
    _run("run_progressive.py")
