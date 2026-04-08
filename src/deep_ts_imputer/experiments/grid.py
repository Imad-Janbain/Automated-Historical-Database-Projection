"""Phase 1 — modelling-task grid runner.

For every combination of ``(target, input_combination, model_architecture)``
defined in a grid spec, train a model on the period where every variable
exists, evaluate on a held-out test split and append the result to the
:class:`~deep_ts_imputer.experiments.database.ResultsDatabase`.

This is the deep-learning equivalent of the ``GridSearchCV`` step in a
classical ML pipeline. Its output is the database that drives Phase 2.

Grid spec (YAML, see ``configs/seine_grid.yaml`` for a full example):

.. code-block:: yaml

    base_config: configs/seine_water_quality.yaml
    db_path: outputs/seine_grid/results.csv
    artifacts_dir: outputs/seine_grid/artifacts
    targets:
      - { name: "Conductivity_Tancarville_Surface", station: "Tancarville" }
      - { name: "Dissolved_Oxygen_Tancarville_Surface", station: "Tancarville" }
    input_combinations:
      - ["Water_level_Honfleur"]
      - ["Water_level_Honfleur", "Water_level_Tancarville"]
      - ["Water_level_Honfleur", "Water_level_Tancarville", "Water_level_Caudebec"]
    models:
      - { name: bilstm, units: 64, num_layers: 2, dropout: 0.1 }
      - { name: cnn_bilstm, units: 64, num_layers: 2, dropout: 0.2, use_attention: true }
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import yaml

from deep_ts_imputer.data.dataset import load_timeseries, select_features
from deep_ts_imputer.data.preprocessing import chronological_split
from deep_ts_imputer.data.windowing import sliding_window
from deep_ts_imputer.evaluation.metrics import compute_all
from deep_ts_imputer.experiments.database import ResultsDatabase, TrialRecord
from deep_ts_imputer.models.factory import build_model
from deep_ts_imputer.training.trainer import fit_model
from deep_ts_imputer.utils.config import Config, ModelConfig, load_config
from deep_ts_imputer.utils.logging import get_logger
from deep_ts_imputer.utils.seed import set_global_seed

LOGGER = get_logger("grid")


def _hash_trial(target: str, inputs: list[str], model_kwargs: dict) -> str:
    blob = json.dumps(
        {"target": target, "inputs": sorted(inputs), "model": model_kwargs},
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def load_grid_spec(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _override_cfg(
    base: Config,
    target: str,
    inputs: list[str],
    model_kwargs: dict,
) -> Config:
    cfg = deepcopy(base)
    cfg.data.input_features = list(inputs)
    cfg.data.target_features = [target]
    cfg.model = ModelConfig(**{**asdict(cfg.model), **model_kwargs})
    return cfg


def _train_one(
    cfg: Config,
    target: str,
    station: str,
    inputs: list[str],
    model_kwargs: dict,
    artifacts_dir: Path,
) -> TrialRecord:
    df = load_timeseries(
        cfg.data.path,
        date_column=cfg.data.date_column,
        date_begin=cfg.data.date_begin,
        date_end=cfg.data.date_end,
        interpolate_missing=cfg.data.interpolate_missing,
    )
    inputs_df, targets_df = select_features(df, inputs, [target])
    splits = chronological_split(
        inputs_df, targets_df,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        scaler_name=cfg.data.scaler,
    )
    x_tr, y_tr = sliding_window(
        splits.x_train, splits.y_train, cfg.window.look_back, cfg.window.horizon,
    )
    x_va, y_va = sliding_window(
        splits.x_val, splits.y_val, cfg.window.look_back, cfg.window.horizon,
    )
    x_te, y_te = sliding_window(
        splits.x_test, splits.y_test, cfg.window.look_back, cfg.window.horizon,
    )

    model = build_model(
        cfg.model, cfg.train,
        look_back=cfg.window.look_back,
        n_features_in=x_tr.shape[2],
        n_outputs=y_tr.shape[1],
    )

    trial_hash = _hash_trial(target, inputs, model_kwargs)
    trial_dir = artifacts_dir / trial_hash
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_path = trial_dir / "model.keras"
    x_scaler_path = trial_dir / "x_scaler.joblib"
    y_scaler_path = trial_dir / "y_scaler.joblib"

    fit_model(
        model, x_tr, y_tr, x_va, y_va,
        cfg.train,
        checkpoint_path=model_path,
        verbose=0,
    )
    joblib.dump(splits.x_scaler, x_scaler_path)
    joblib.dump(splits.y_scaler, y_scaler_path)

    y_pred = splits.y_scaler.inverse_transform(model.predict(x_te, verbose=0))
    y_true = splits.y_scaler.inverse_transform(y_te)
    metrics = compute_all(y_true, y_pred)

    return TrialRecord(
        target=target,
        station=station,
        input_features=tuple(sorted(inputs)),
        model_name=cfg.model.name,
        metrics=metrics,
        model_path=str(model_path),
        x_scaler_path=str(x_scaler_path),
        y_scaler_path=str(y_scaler_path),
        config_hash=trial_hash,
    )


def run_grid(grid_spec_path: str | Path) -> ResultsDatabase:
    spec = load_grid_spec(grid_spec_path)
    base_cfg = load_config(spec["base_config"])
    set_global_seed(base_cfg.seed)

    db_path = Path(spec["db_path"])
    artifacts_dir = Path(spec["artifacts_dir"])
    db = ResultsDatabase(db_path)

    targets = spec["targets"]
    input_combinations = spec["input_combinations"]
    models = spec["models"]

    total = len(targets) * len(input_combinations) * len(models)
    LOGGER.info("Grid: %d targets × %d input combos × %d models = %d trials",
                len(targets), len(input_combinations), len(models), total)

    counter = 0
    for target_spec, inputs, model_kwargs in product(targets, input_combinations, models):
        counter += 1
        target = target_spec["name"] if isinstance(target_spec, dict) else target_spec
        station = target_spec.get("station", "") if isinstance(target_spec, dict) else ""
        trial_hash = _hash_trial(target, inputs, model_kwargs)
        if not db.df.empty and (db.df["config_hash"] == trial_hash).any():
            LOGGER.info("[%d/%d] cached %s", counter, total, trial_hash)
            continue

        LOGGER.info(
            "[%d/%d] target=%s inputs=%d model=%s",
            counter, total, target, len(inputs), model_kwargs.get("name"),
        )
        try:
            cfg = _override_cfg(base_cfg, target, list(inputs), model_kwargs)
            record = _train_one(cfg, target, station, list(inputs), model_kwargs, artifacts_dir)
            db.add(record)
            LOGGER.info("    -> r2=%.3f rmse=%.4f", record.metrics["r2"], record.metrics["rmse"])
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("    !! trial failed: %s", exc)

    LOGGER.info("Grid done. %d records in %s", len(db), db_path)
    return db
