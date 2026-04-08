"""Tests for the units / column-alias helpers."""

from __future__ import annotations

import pandas as pd

from deep_ts_imputer.utils.units import (
    apply_column_aliases,
    label_with_unit,
    units_for,
)


def test_label_with_unit_with_known_unit():
    assert label_with_unit("cond", {"cond": "μS·cm⁻¹"}) == "cond (μS·cm⁻¹)"


def test_label_with_unit_with_unknown_unit():
    assert label_with_unit("cond", {"other": "x"}) == "cond"


def test_label_with_unit_with_none_units():
    assert label_with_unit("cond", None) == "cond"


def test_apply_column_aliases_renames_known_columns():
    df = pd.DataFrame({"Conductivity (μS·cm⁻¹)": [1, 2, 3], "Other": [4, 5, 6]})
    out = apply_column_aliases(df, {"Conductivity (μS·cm⁻¹)": "Conductivity"})
    assert "Conductivity" in out.columns
    assert "Conductivity (μS·cm⁻¹)" not in out.columns
    assert "Other" in out.columns


def test_apply_column_aliases_silently_ignores_missing():
    df = pd.DataFrame({"a": [1]})
    out = apply_column_aliases(df, {"missing": "renamed"})
    assert list(out.columns) == ["a"]


def test_apply_column_aliases_with_empty_map():
    df = pd.DataFrame({"a": [1]})
    out = apply_column_aliases(df, None)
    assert list(out.columns) == ["a"]


def test_units_for_filters_to_requested_columns():
    units = {"a": "m", "b": "s", "c": "kg"}
    assert units_for(["a", "c"], units) == {"a": "m", "c": "kg"}


def test_units_for_with_no_units_returns_empty_dict():
    assert units_for(["a", "b"], None) == {}
