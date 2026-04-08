"""Generate a synthetic multivariate time series for demos and tests.

The synthetic series mimics the *structure* of the Seine water-quality
dataset (correlated tide-driven signals + noise + a few seasonal cycles)
without distributing the original data. This makes the repo runnable
end-to-end on a fresh clone.

Usage::

    python scripts/generate_synthetic_demo.py --out data/synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_hours: int = 24 * 365 * 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    # Tide-like dominant cycle (~12.4 h M2 tide) + diurnal + annual.
    tide = np.sin(2 * np.pi * t / 12.4)
    diurnal = 0.3 * np.sin(2 * np.pi * t / 24)
    annual = 0.5 * np.sin(2 * np.pi * t / (24 * 365))

    water_level_upstream = 2.5 + 1.2 * tide + diurnal + 0.4 * annual + 0.1 * rng.standard_normal(n_hours)
    water_level_downstream = water_level_upstream * 0.85 + 0.3 + 0.1 * rng.standard_normal(n_hours)

    conductivity = 800 + 400 * (1 - tide) + 50 * rng.standard_normal(n_hours)
    dissolved_oxygen = 8.5 - 0.4 * tide + 0.3 * np.sin(2 * np.pi * t / 24) + 0.2 * rng.standard_normal(n_hours)
    turbidity = np.clip(20 + 15 * np.abs(tide) + 5 * rng.standard_normal(n_hours), 0, None)

    dates = pd.date_range("2018-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame(
        {
            "Dates": dates,
            "water_level_upstream": water_level_upstream,
            "water_level_downstream": water_level_downstream,
            "conductivity": conductivity,
            "dissolved_oxygen": dissolved_oxygen,
            "turbidity": turbidity,
        },
    )

    # Knock out ~5% of the target columns to mimic real sensor gaps.
    for col in ("conductivity", "dissolved_oxygen", "turbidity"):
        idx = rng.choice(n_hours, size=int(0.05 * n_hours), replace=False)
        df.loc[idx, col] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--n-hours", type=int, default=24 * 365 * 3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n_hours, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
