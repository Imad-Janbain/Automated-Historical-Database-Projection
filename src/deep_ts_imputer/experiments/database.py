"""Persistent database of modelling-task results.

This is the artifact that links Phase 1 (modelling) and Phase 2
(progressive reconstruction) in the methodology of Janbain et al. 2023.

Each record represents one trained model and stores:

* the **target** column it predicts
* the **station** the target belongs to (free-form string, used for
  filtering and reporting only)
* the **input_features** it consumes (a frozen, sorted tuple — this is
  the feasibility key)
* the **model name** (`bilstm`, `cnn_bilstm`, …)
* the test **metrics** (rmse, mae, r2, nse, kge)
* the path to the persisted Keras model and its scalers

The on-disk format is a single CSV. Two reasons: it diffs nicely in git,
and there is no concurrency story to worry about — the grid runner is
sequential.

Public API
----------
* :class:`ResultsDatabase` — load / append / query / pick the best feasible
  record for a target given the currently-available feature set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class TrialRecord:
    target: str
    station: str
    input_features: tuple[str, ...]
    model_name: str
    metrics: dict[str, float]
    model_path: str
    x_scaler_path: str
    y_scaler_path: str
    config_hash: str = ""

    def to_row(self) -> dict:
        return {
            "target": self.target,
            "station": self.station,
            "input_features": json.dumps(list(self.input_features)),
            "n_inputs": len(self.input_features),
            "model_name": self.model_name,
            "rmse": self.metrics.get("rmse"),
            "mae": self.metrics.get("mae"),
            "r2": self.metrics.get("r2"),
            "nse": self.metrics.get("nse"),
            "kge": self.metrics.get("kge"),
            "model_path": self.model_path,
            "x_scaler_path": self.x_scaler_path,
            "y_scaler_path": self.y_scaler_path,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_row(cls, row: pd.Series) -> "TrialRecord":
        return cls(
            target=row["target"],
            station=row["station"],
            input_features=tuple(json.loads(row["input_features"])),
            model_name=row["model_name"],
            metrics={
                k: float(row[k]) for k in ("rmse", "mae", "r2", "nse", "kge")
                if k in row and pd.notna(row[k])
            },
            model_path=row["model_path"],
            x_scaler_path=row["x_scaler_path"],
            y_scaler_path=row["y_scaler_path"],
            config_hash=row.get("config_hash", ""),
        )


class ResultsDatabase:
    """CSV-backed store of modelling-task trials.

    The database is the bridge between Phase 1 and Phase 2 of the paper:
    Phase 1 ``add()``\\ s every trained model; Phase 2 calls
    :meth:`best_feasible` to pick the next reconstruction step.
    """

    COLUMNS = [
        "target", "station", "input_features", "n_inputs", "model_name",
        "rmse", "mae", "r2", "nse", "kge",
        "model_path", "x_scaler_path", "y_scaler_path", "config_hash",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)

    # ------------------------------------------------------------------
    # write side
    # ------------------------------------------------------------------
    def add(self, record: TrialRecord) -> None:
        row = record.to_row()
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.path, index=False)

    # ------------------------------------------------------------------
    # read side
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def for_target(self, target: str) -> pd.DataFrame:
        return self.df[self.df["target"] == target]

    def feasible(
        self,
        target: str,
        available_features: Iterable[str],
    ) -> pd.DataFrame:
        """Return all trials for ``target`` whose inputs are a subset of
        the currently available features."""
        avail = set(available_features)
        rows = self.for_target(target)
        if rows.empty:
            return rows
        mask = rows["input_features"].apply(
            lambda s: set(json.loads(s)).issubset(avail),
        )
        return rows[mask]

    def best_feasible(
        self,
        target: str,
        available_features: Iterable[str],
        metric: str = "r2",
        higher_is_better: bool | None = None,
    ) -> TrialRecord | None:
        """Return the highest-scoring feasible trial for ``target``.

        Returns ``None`` if no feasible trial exists. ``higher_is_better``
        is auto-detected from common metric names if not provided.
        """
        feasible = self.feasible(target, available_features)
        if feasible.empty:
            return None
        if higher_is_better is None:
            higher_is_better = metric.lower() in {"r2", "nse", "kge"}
        ascending = not higher_is_better
        sorted_df = feasible.sort_values(metric, ascending=ascending)
        return TrialRecord.from_row(sorted_df.iloc[0])

    def best_for_each_target(
        self,
        targets: Iterable[str],
        available_features: Iterable[str],
        metric: str = "r2",
    ) -> dict[str, TrialRecord]:
        out: dict[str, TrialRecord] = {}
        for t in targets:
            rec = self.best_feasible(t, available_features, metric=metric)
            if rec is not None:
                out[t] = rec
        return out
