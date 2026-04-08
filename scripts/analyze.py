"""Run the full exploratory data analysis on a dataset.

Reads a config file (or a raw CSV path), loads the series **without**
interpolating NaNs (so the missing-value plots show the truth), and writes
all EDA artifacts under ``<output_dir>/eda/``.

Examples::

    # From a project config
    python scripts/analyze.py --config configs/seine_water_quality.yaml

    # Or directly on any CSV
    python scripts/analyze.py --input data/synthetic.csv --out outputs/eda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deep_ts_imputer.data.dataset import load_timeseries
from deep_ts_imputer.eda import run_full_eda
from deep_ts_imputer.utils.config import load_config
from deep_ts_imputer.utils.logging import get_logger

LOGGER = get_logger("analyze")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML config (preferred)")
    parser.add_argument("--input", type=Path, help="Raw CSV path (overrides --config)")
    parser.add_argument("--date-column", default="Dates")
    parser.add_argument("--out", type=Path, help="Output directory (defaults to <output_dir>/eda)")
    args = parser.parse_args()

    if args.config is None and args.input is None:
        parser.error("provide --config or --input")

    if args.config is not None:
        cfg = load_config(args.config)
        input_path = args.input or Path(cfg.data.path)
        date_column = cfg.data.date_column
        out_dir = args.out or (Path(cfg.output_dir) / "eda")
        date_begin, date_end = cfg.data.date_begin, cfg.data.date_end
    else:
        input_path = args.input
        date_column = args.date_column
        out_dir = args.out or Path("outputs/eda")
        date_begin = date_end = None

    LOGGER.info("Loading %s", input_path)
    df = load_timeseries(
        input_path,
        date_column=date_column,
        date_begin=date_begin,
        date_end=date_end,
        interpolate_missing=False,  # we want to see the gaps
    )
    LOGGER.info("Loaded %d rows × %d columns", *df.shape)

    LOGGER.info("Writing EDA artifacts to %s", out_dir)
    artifacts = run_full_eda(df, out_dir)
    (Path(out_dir) / "manifest.json").write_text(
        json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2),
    )
    for name, path in artifacts.items():
        LOGGER.info("  %-22s -> %s", name, path)


if __name__ == "__main__":
    main()
