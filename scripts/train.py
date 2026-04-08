"""Train a single model from a YAML config and save artifacts.

Example::

    python scripts/train.py --config configs/seine_water_quality.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from deep_ts_imputer.data.dataset import load_timeseries, select_features
from deep_ts_imputer.data.preprocessing import chronological_split
from deep_ts_imputer.data.windowing import sliding_window
from deep_ts_imputer.evaluation.metrics import compute_all
from deep_ts_imputer.evaluation.plots import plot_history, plot_predictions, plot_scatter
from deep_ts_imputer.models.factory import build_model
from deep_ts_imputer.training.trainer import fit_model
from deep_ts_imputer.utils.config import load_config
from deep_ts_imputer.utils.logging import get_logger
from deep_ts_imputer.utils.seed import set_global_seed

LOGGER = get_logger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataset from %s", cfg.data.path)
    df = load_timeseries(
        cfg.data.path,
        date_column=cfg.data.date_column,
        date_begin=cfg.data.date_begin,
        date_end=cfg.data.date_end,
        interpolate_missing=cfg.data.interpolate_missing,
        column_aliases=cfg.data.column_aliases,
    )
    inputs, targets = select_features(df, cfg.data.input_features, cfg.data.target_features)
    LOGGER.info("Dataset shape: inputs=%s, targets=%s", inputs.shape, targets.shape)

    splits = chronological_split(
        inputs,
        targets,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        scaler_name=cfg.data.scaler,
    )

    x_train, y_train = sliding_window(
        splits.x_train, splits.y_train, cfg.window.look_back, cfg.window.horizon,
    )
    x_val, y_val = sliding_window(
        splits.x_val, splits.y_val, cfg.window.look_back, cfg.window.horizon,
    )
    x_test, y_test = sliding_window(
        splits.x_test, splits.y_test, cfg.window.look_back, cfg.window.horizon,
    )
    LOGGER.info("Windows: train=%s val=%s test=%s", x_train.shape, x_val.shape, x_test.shape)

    n_features_in = x_train.shape[2]
    n_outputs = y_train.shape[1]

    model = build_model(
        cfg.model, cfg.train,
        look_back=cfg.window.look_back,
        n_features_in=n_features_in,
        n_outputs=n_outputs,
    )
    model.summary(print_fn=LOGGER.info)

    history = fit_model(
        model, x_train, y_train, x_val, y_val,
        cfg.train,
        checkpoint_path=out_dir / "model.keras",
    )

    # Test-time evaluation in the original (unscaled) units.
    y_pred_scaled = model.predict(x_test, verbose=0)
    y_pred = splits.y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, len(cfg.data.target_features)),
    ).reshape(y_pred_scaled.shape[0], -1, len(cfg.data.target_features))[:, 0, :]
    y_true = splits.y_scaler.inverse_transform(
        y_test.reshape(-1, len(cfg.data.target_features)),
    ).reshape(y_test.shape[0], -1, len(cfg.data.target_features))[:, 0, :]

    metrics = compute_all(y_true, y_pred)
    LOGGER.info("Test metrics: %s", metrics)

    # Persist artifacts.
    target_units = {t: cfg.data.units.get(t, "") for t in cfg.data.target_features}
    (out_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "units": target_units}, indent=2),
    )
    (out_dir / "units.json").write_text(
        json.dumps({**cfg.data.units}, indent=2, ensure_ascii=False),
    )
    joblib.dump(splits.x_scaler, out_dir / "x_scaler.joblib")
    joblib.dump(splits.y_scaler, out_dir / "y_scaler.joblib")
    plot_history(history.history, out_dir / "training_history.png")
    plot_predictions(
        splits.test_index[cfg.window.look_back : cfg.window.look_back + len(y_true)],
        y_true, y_pred, cfg.data.target_features, out_dir / "predictions.png",
        units=cfg.data.units,
    )
    plot_scatter(
        y_true, y_pred, cfg.data.target_features, out_dir / "scatter.png",
        units=cfg.data.units,
    )
    LOGGER.info("Artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
