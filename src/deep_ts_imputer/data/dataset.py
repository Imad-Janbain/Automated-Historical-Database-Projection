"""Tabular time-series loading and cleaning.

The original Paper-2 codebase loaded a single hard-coded CSV with reversed
row order, manually re-indexed by date, and clipped a hard-coded window.
This module generalises that into a small, reusable function that works
with any CSV containing a date column and one or more numeric columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from deep_ts_imputer.utils.units import apply_column_aliases


def load_timeseries(
    path: str | Path,
    date_column: str = "Dates",
    date_begin: str | None = None,
    date_end: str | None = None,
    interpolate_missing: bool = True,
    delimiter: str = ",",
    column_aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load a CSV/TSV time-series file and return a date-indexed DataFrame.

    Parameters
    ----------
    path:
        Path to the input file.
    date_column:
        Name of the column to use as the temporal index.
    date_begin, date_end:
        Optional ISO date strings used to clip the series.
    interpolate_missing:
        If True, linearly interpolate NaNs (limit_direction='both').
    delimiter:
        Field delimiter passed to ``pandas.read_csv``.
    column_aliases:
        Optional ``raw_name -> clean_name`` mapping applied immediately
        after loading. Use it to strip unit suffixes from column headers
        (e.g. ``"Conductivity_Tancarville_Surface (μS·cm⁻¹)"`` →
        ``"Conductivity_Tancarville_Surface"``) so configs can refer to
        clean identifiers regardless of how the source CSV is formatted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_csv(path, delimiter=delimiter)
    df = apply_column_aliases(df, column_aliases)

    if date_column not in df.columns:
        raise KeyError(
            f"Date column '{date_column}' not found. Available: {list(df.columns)}",
        )

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).set_index(date_column)

    if date_begin is not None:
        df = df.loc[pd.to_datetime(date_begin):]
    if date_end is not None:
        df = df.loc[: pd.to_datetime(date_end)]

    if interpolate_missing:
        df = df.interpolate(method="linear", limit_direction="both")

    return df


def select_features(
    df: pd.DataFrame,
    input_features: list[str],
    target_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and slice the requested input/target columns."""
    missing = [c for c in (*input_features, *target_features) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataset: {missing}")
    return df[input_features].copy(), df[target_features].copy()
