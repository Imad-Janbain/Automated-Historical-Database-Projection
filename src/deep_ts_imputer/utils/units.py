"""Unit tracking and column-name normalisation.

The pipeline itself is mathematically unit-agnostic — MinMax / Standard /
Robust scalers don't care what units a column is in, and the inverse
transform recovers the original scale. **But** plot labels, metric reports
and audit trails do care: ``rmse: 38.1`` is meaningless without
``μS·cm⁻¹`` next to it.

This module provides a tiny, library-wide convention:

* a ``units`` dict (column name → unit string) lives on
  :class:`~deep_ts_imputer.utils.config.DataConfig`
* :func:`label_with_unit` formats a column name as ``"name (unit)"`` for
  plotting, falling back gracefully when the unit is unknown
* :func:`apply_column_aliases` renames messy real-world column names
  (e.g. ``Conductivity_Tancarville_Surface (μS·cm⁻¹)``) to the clean
  identifiers used in configs, keeping the original CSV untouched

The pattern is borrowed from how scientific code traditionally
documents units in column headers, generalised so the convention is
explicit and machine-readable instead of hidden in column-name strings.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def label_with_unit(column: str, units: Mapping[str, str] | None = None) -> str:
    """Return ``"column (unit)"`` if a unit is known, else just ``column``."""
    if not units:
        return column
    unit = units.get(column)
    if unit:
        return f"{column} ({unit})"
    return column


def apply_column_aliases(
    df: pd.DataFrame,
    aliases: Mapping[str, str] | None,
) -> pd.DataFrame:
    """Rename DataFrame columns according to ``aliases`` (raw → clean).

    The function is forgiving: keys that don't appear in ``df.columns``
    are silently ignored, so a single alias map can cover several
    related datasets.
    """
    if not aliases:
        return df
    rename_map = {raw: clean for raw, clean in aliases.items() if raw in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def units_for(columns: list[str], units: Mapping[str, str] | None) -> dict[str, str]:
    """Filter a units dict to just the columns we care about.

    Returns an empty dict instead of None so downstream code can use it
    unconditionally without ``if units is not None`` guards.
    """
    if not units:
        return {}
    return {c: units[c] for c in columns if c in units}
