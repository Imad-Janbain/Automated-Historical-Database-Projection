"""Tests for the Phase 1 / Phase 2 results database."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from deep_ts_imputer.experiments.database import ResultsDatabase, TrialRecord


def _make_record(target, inputs, model, r2):
    return TrialRecord(
        target=target,
        station="estuary",
        input_features=tuple(inputs),
        model_name=model,
        metrics={"r2": r2, "rmse": 1.0 - r2, "mae": 0.0, "nse": r2, "kge": r2},
        model_path=f"/tmp/{model}.keras",
        x_scaler_path="/tmp/x.joblib",
        y_scaler_path="/tmp/y.joblib",
    )


@pytest.fixture
def populated_db(tmp_path: Path) -> ResultsDatabase:
    db = ResultsDatabase(tmp_path / "results.csv")
    db.add(_make_record("conductivity", ["wl_t"], "bilstm", r2=0.85))
    db.add(_make_record("conductivity", ["wl_h", "wl_t"], "cnn_bilstm", r2=0.92))
    db.add(_make_record("dissolved_oxygen", ["wl_h"], "gru", r2=0.55))
    db.add(_make_record("dissolved_oxygen", ["wl_t", "conductivity"], "bilstm", r2=0.78))
    db.add(_make_record("turbidity", ["wl_t", "conductivity", "dissolved_oxygen"], "cnn_bilstm", r2=0.62))
    return db


def test_persistence_round_trip(populated_db: ResultsDatabase, tmp_path: Path):
    db2 = ResultsDatabase(populated_db.path)
    assert len(db2) == 5


def test_feasibility_filters_by_required_inputs(populated_db: ResultsDatabase):
    feasible = populated_db.feasible("dissolved_oxygen", available_features=["wl_h", "wl_t"])
    # Only the GRU is feasible with water levels alone.
    assert len(feasible) == 1
    assert feasible.iloc[0]["model_name"] == "gru"


def test_best_feasible_picks_highest_r2(populated_db: ResultsDatabase):
    rec = populated_db.best_feasible("conductivity", ["wl_h", "wl_t"])
    assert rec is not None
    assert rec.model_name == "cnn_bilstm"


def test_best_feasible_returns_none_when_unsatisfiable(populated_db: ResultsDatabase):
    rec = populated_db.best_feasible("turbidity", ["wl_h"])
    assert rec is None


def test_progressive_unlock_pattern(populated_db: ResultsDatabase):
    """Reproduce the Phase 2 logic by hand and check the unlock order."""
    available = {"wl_h", "wl_t"}
    remaining = {"conductivity", "dissolved_oxygen", "turbidity"}
    order = []
    while remaining:
        # Pick the globally best feasible across all remaining targets.
        best = None
        for t in remaining:
            rec = populated_db.best_feasible(t, available)
            if rec is None:
                continue
            if best is None or rec.metrics["r2"] > best[1].metrics["r2"]:
                best = (t, rec)
        if best is None:
            break
        order.append(best[0])
        available.add(best[0])
        remaining.remove(best[0])
    # Conductivity has the strongest WL-only correlation, so it goes first.
    # DO requires conductivity (its best model), so it goes second.
    # Turbidity needs both DO and conductivity, so it goes last.
    assert order == ["conductivity", "dissolved_oxygen", "turbidity"]
