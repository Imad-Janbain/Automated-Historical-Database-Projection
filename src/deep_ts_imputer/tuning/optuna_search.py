"""Hyper-parameter search with Optuna.

The search space covers the four knobs that mattered most in the original
Paper-2 experiments:

* recurrent ``units``         — log-uniform 16…256
* ``num_layers``              — int 1…4
* ``dropout``                 — uniform 0.0…0.5
* ``learning_rate``           — log-uniform 1e-4…1e-2

Plus, when the model is ``cnn_bilstm``, ``cnn_filters`` (16…128).

Each trial trains a *short* model (``epochs // 2``) with early stopping
and median pruning, and reports validation RMSE. The best parameters are
returned and can be merged back into the main config for a full
re-training.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import optuna
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback

from deep_ts_imputer.models.factory import build_model
from deep_ts_imputer.training.trainer import fit_model
from deep_ts_imputer.utils.config import Config
from deep_ts_imputer.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _suggest(trial: optuna.Trial, base_cfg: Config) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.model.units = trial.suggest_int("units", 16, 256, log=True)
    cfg.model.num_layers = trial.suggest_int("num_layers", 1, 4)
    cfg.model.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    cfg.train.learning_rate = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True,
    )
    if cfg.model.name.lower() == "cnn_bilstm":
        cfg.model.cnn_filters = trial.suggest_int("cnn_filters", 16, 128, log=True)
    return cfg


def _make_sampler(name: str) -> optuna.samplers.BaseSampler:
    name = name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=42)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=42)
    raise ValueError(f"Unknown sampler '{name}'")


def _make_pruner(name: str) -> optuna.pruners.BasePruner:
    name = name.lower()
    if name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=5)
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Unknown pruner '{name}'")


def run_search(
    base_cfg: Config,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, Any], optuna.Study]:
    """Run an Optuna study and return ``(best_params, study)``."""
    look_back = x_train.shape[1]
    n_features_in = x_train.shape[2]
    n_outputs = y_train.shape[1]

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = _suggest(trial, base_cfg)
        # Cap epochs during search; final model will train longer.
        trial_cfg.train.epochs = max(10, base_cfg.train.epochs // 2)

        tf.keras.backend.clear_session()
        model = build_model(
            trial_cfg.model,
            trial_cfg.train,
            look_back=look_back,
            n_features_in=n_features_in,
            n_outputs=n_outputs,
        )
        history = fit_model(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            trial_cfg.train,
            verbose=0,
            extra_callbacks=[TFKerasPruningCallback(trial, "val_loss")],
        )
        val_loss = float(np.min(history.history["val_loss"]))
        return val_loss

    study = optuna.create_study(
        direction=base_cfg.tune.direction,
        sampler=_make_sampler(base_cfg.tune.sampler),
        pruner=_make_pruner(base_cfg.tune.pruner),
        study_name=base_cfg.tune.study_name,
        storage=base_cfg.tune.storage,
        load_if_exists=base_cfg.tune.storage is not None,
    )
    LOGGER.info(
        "Starting Optuna study '%s' (%d trials)",
        base_cfg.tune.study_name,
        base_cfg.tune.n_trials,
    )
    study.optimize(
        objective,
        n_trials=base_cfg.tune.n_trials,
        timeout=base_cfg.tune.timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    LOGGER.info("Best val_loss=%.5f params=%s", study.best_value, study.best_params)
    return study.best_params, study


def merge_best_params(cfg: Config, best_params: dict[str, Any]) -> Config:
    """Return a copy of ``cfg`` with the Optuna best parameters applied."""
    out = deepcopy(cfg)
    if "units" in best_params:
        out.model.units = best_params["units"]
    if "num_layers" in best_params:
        out.model.num_layers = best_params["num_layers"]
    if "dropout" in best_params:
        out.model.dropout = best_params["dropout"]
    if "learning_rate" in best_params:
        out.train.learning_rate = best_params["learning_rate"]
    if "cnn_filters" in best_params:
        out.model.cnn_filters = best_params["cnn_filters"]
    return out
