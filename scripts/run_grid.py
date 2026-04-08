"""Phase 1 — run the modelling-task grid and populate the results database.

Example::

    python scripts/run_grid.py --grid configs/seine_grid.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_ts_imputer.experiments import run_grid
from deep_ts_imputer.utils.logging import get_logger

LOGGER = get_logger("run_grid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True, help="Grid spec YAML")
    args = parser.parse_args()

    db = run_grid(args.grid)
    LOGGER.info("Database now has %d records", len(db))


if __name__ == "__main__":
    main()
