"""Run an Optuna hyper-parameter search, then retrain at full budget.

Example::

    python scripts/tune.py --config configs/seine_water_quality.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from deep_ts_imputer.data.dataset import load_timeseries, select_features
from deep_ts_imputer.data.preprocessing import chronological_split
from deep_ts_imputer.data.windowing import sliding_window
from deep_ts_imputer.evaluation.metrics import compute_all
from deep_ts_imputer.evaluation.plots import plot_history, plot_predictions
from deep_ts_imputer.models.factory import build_model
from deep_ts_imputer.training.trainer import fit_model
from deep_ts_imputer.tuning.optuna_search import merge_best_params, run_search
from deep_ts_imputer.utils.config import load_config
from deep_ts_imputer.utils.logging import get_logger
from deep_ts_imputer.utils.seed import set_global_seed

LOGGER = get_logger("tune")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_timeseries(
        cfg.data.path,
        date_column=cfg.data.date_column,
        date_begin=cfg.data.date_begin,
        date_end=cfg.data.date_end,
        interpolate_missing=cfg.data.interpolate_missing,
    )
    inputs, targets = select_features(df, cfg.data.input_features, cfg.data.target_features)
    splits = chronological_split(
        inputs, targets,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        scaler_name=cfg.data.scaler,
    )
    x_train, y_train = sliding_window(splits.x_train, splits.y_train, cfg.window.look_back, cfg.window.horizon)
    x_val, y_val = sliding_window(splits.x_val, splits.y_val, cfg.window.look_back, cfg.window.horizon)
    x_test, y_test = sliding_window(splits.x_test, splits.y_test, cfg.window.look_back, cfg.window.horizon)

    LOGGER.info("Launching Optuna search (%d trials)", cfg.tune.n_trials)
    best_params, study = run_search(cfg, x_train, y_train, x_val, y_val)
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    # Persist the study results that don't depend on plotly.
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(out_dir / "optuna_trials.csv", index=False)
    except Exception as exc:  # pragma: no cover - cosmetic
        LOGGER.warning("Could not export trials dataframe: %s", exc)

    LOGGER.info("Re-training best config at full budget")
    final_cfg = merge_best_params(cfg, best_params)
    model = build_model(
        final_cfg.model, final_cfg.train,
        look_back=cfg.window.look_back,
        n_features_in=x_train.shape[2],
        n_outputs=y_train.shape[1],
    )
    history = fit_model(
        model, x_train, y_train, x_val, y_val,
        final_cfg.train,
        checkpoint_path=out_dir / "model.keras",
    )
    plot_history(history.history, out_dir / "training_history.png")

    n_targets = len(cfg.data.target_features)
    y_pred_scaled = model.predict(x_test, verbose=0)
    y_pred = splits.y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, n_targets),
    ).reshape(y_pred_scaled.shape[0], -1, n_targets)[:, 0, :]
    y_true = splits.y_scaler.inverse_transform(
        y_test.reshape(-1, n_targets),
    ).reshape(y_test.shape[0], -1, n_targets)[:, 0, :]

    metrics = compute_all(y_true, y_pred)
    LOGGER.info("Final test metrics: %s", metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(splits.x_scaler, out_dir / "x_scaler.joblib")
    joblib.dump(splits.y_scaler, out_dir / "y_scaler.joblib")
    plot_predictions(
        splits.test_index[cfg.window.look_back : cfg.window.look_back + len(y_true)],
        y_true, y_pred, cfg.data.target_features, out_dir / "predictions.png",
    )
    LOGGER.info("Done. Artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
