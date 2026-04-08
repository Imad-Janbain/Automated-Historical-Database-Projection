"""Apply a trained model to reconstruct missing values in a CSV.

Example::

    python scripts/reconstruct.py \\
        --config configs/seine_water_quality.yaml \\
        --model outputs/seine_water_quality/model.keras \\
        --input data/seine_with_gaps.csv \\
        --output outputs/seine_water_quality/reconstructed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import tensorflow as tf

from deep_ts_imputer.data.dataset import load_timeseries, select_features
from deep_ts_imputer.imputation.reconstructor import reconstruct_series
from deep_ts_imputer.utils.config import load_config
from deep_ts_imputer.utils.logging import get_logger

LOGGER = get_logger("reconstruct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Path to .keras model")
    parser.add_argument("--input", type=Path, required=True, help="CSV with gaps to fill")
    parser.add_argument("--output", type=Path, required=True, help="Where to write reconstructed CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifact_dir = args.model.parent

    LOGGER.info("Loading model from %s", args.model)
    model = tf.keras.models.load_model(args.model)
    x_scaler = joblib.load(artifact_dir / "x_scaler.joblib")
    y_scaler = joblib.load(artifact_dir / "y_scaler.joblib")

    LOGGER.info("Loading input series from %s", args.input)
    df = load_timeseries(
        args.input,
        date_column=cfg.data.date_column,
        interpolate_missing=False,  # we want to *see* the gaps
    )
    inputs, targets = select_features(df, cfg.data.input_features, cfg.data.target_features)

    n_missing_before = int(targets.isna().sum().sum())
    LOGGER.info("Missing target cells before reconstruction: %d", n_missing_before)

    filled = reconstruct_series(
        model=model,
        inputs=inputs.interpolate(method="linear", limit_direction="both"),
        targets=targets,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        look_back=cfg.window.look_back,
        horizon=cfg.window.horizon,
    )
    n_missing_after = int(filled.isna().sum().sum())
    LOGGER.info("Missing target cells after reconstruction: %d", n_missing_after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(args.output)
    LOGGER.info("Wrote reconstructed series to %s", args.output)


if __name__ == "__main__":
    main()
