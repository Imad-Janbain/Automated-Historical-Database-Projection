"""Tests for the depth-aware column-name parser used by the spatial EDA helpers."""

from __future__ import annotations

import pytest

from deep_ts_imputer.eda.eda import DEFAULT_DEPTHS, parse_column_name

STATIONS = [
    "Honfleur", "Tancarville", "Caudebec", "Duclair",
    "Rouen", "Fatouville", "Valdesleux",
]
PARAMETERS = ["Water_level", "Conductivity", "Dissolved_Oxygen", "Turbidity"]


@pytest.mark.parametrize(
    "column,expected",
    [
        ("Conductivity_Tancarville_Surface", ("Conductivity", "Tancarville", "Surface")),
        ("Conductivity_Tancarville_Bottom",  ("Conductivity", "Tancarville", "Bottom")),
        ("Conductivity_Fatouville_Surface",  ("Conductivity", "Fatouville", "Surface")),
        ("Conductivity_Fatouville_Bottom",   ("Conductivity", "Fatouville", "Bottom")),
        ("Conductivity_Rouen_Surface",       ("Conductivity", "Rouen", "Surface")),
        ("Conductivity_Valdesleux_Surface",  ("Conductivity", "Valdesleux", "Surface")),
        ("Turbidity_Tancarville_Bottom",     ("Turbidity", "Tancarville", "Bottom")),
        ("Turbidity_Fatouville_Bottom",      ("Turbidity", "Fatouville", "Bottom")),
        ("Dissolved_Oxygen_Tancarville_Surface", ("Dissolved_Oxygen", "Tancarville", "Surface")),
        ("Dissolved_Oxygen_Tancarville_Bottom",  ("Dissolved_Oxygen", "Tancarville", "Bottom")),
        ("Dissolved_Oxygen_Rouen_Surface",       ("Dissolved_Oxygen", "Rouen", "Surface")),
    ],
)
def test_parse_known_seine_columns(column, expected):
    assert parse_column_name(column, STATIONS, PARAMETERS) == expected


def test_parse_no_depth_suffix():
    """Stations with a single probe omit the depth — parser returns None."""
    assert parse_column_name("Water_level_Honfleur", STATIONS, PARAMETERS) == (
        "Water_level", "Honfleur", None,
    )


def test_parse_unknown_station_returns_none_for_station():
    assert parse_column_name(
        "Conductivity_Marseille_Surface", STATIONS, PARAMETERS,
    ) == ("Conductivity", None, "Surface")


def test_parse_unknown_parameter_returns_none_for_parameter():
    assert parse_column_name(
        "Salinity_Tancarville_Surface", STATIONS, PARAMETERS,
    ) == (None, "Tancarville", "Surface")


def test_parse_is_case_insensitive():
    assert parse_column_name(
        "conductivity_tancarville_surface", STATIONS, PARAMETERS,
    ) == ("Conductivity", "Tancarville", "Surface")


def test_parse_custom_depths():
    """Non-default depth labels (e.g. mid-column) are honoured if passed."""
    assert parse_column_name(
        "Conductivity_Tancarville_Mid",
        STATIONS, PARAMETERS,
        depths=("Surface", "Mid", "Bottom"),
    ) == ("Conductivity", "Tancarville", "Mid")


def test_default_depths_constant():
    assert DEFAULT_DEPTHS == ("Surface", "Bottom")
