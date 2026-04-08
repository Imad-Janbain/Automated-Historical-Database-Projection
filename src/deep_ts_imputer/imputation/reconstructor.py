"""Apply a trained model to reconstruct missing values in a time series.

This is the production-side counterpart to the training pipeline. Given a
trained Keras model, the fitted scalers and a series with NaNs, it slides
the model along the series and fills the gaps with the inverse-scaled
predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf


def reconstruct_series(
    model: tf.keras.Model,
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    x_scaler,
    y_scaler,
    look_back: int,
    horizon: int = 1,
) -> pd.DataFrame:
    """Predict every step where ``targets`` has a NaN.

    The function does not modify ``targets``; it returns a new DataFrame
    with the same index and column names where missing values have been
    replaced by the model's prediction.
    """
    if not (inputs.index.equals(targets.index)):
        raise ValueError("inputs and targets must share the same index")

    inputs_scaled = x_scaler.transform(inputs)
    n = len(inputs_scaled)
    target_cols = list(targets.columns)
    n_features_out = len(target_cols)

    # Predict in batches over all valid positions.
    n_samples = n - look_back - horizon + 1
    if n_samples <= 0:
        return targets.copy()

    X = np.stack(
        [inputs_scaled[i : i + look_back] for i in range(n_samples)],
    ).astype(np.float32)
    y_scaled = model.predict(X, verbose=0)
    # Reshape per-step predictions back to (samples, horizon, n_features_out).
    y_scaled = y_scaled.reshape(n_samples, horizon, n_features_out)
    # Use only the first horizon step for filling.
    y_step1 = y_scaled[:, 0, :]
    y_inv = y_scaler.inverse_transform(y_step1)

    filled = targets.copy()
    pred_index = targets.index[look_back : look_back + n_samples]
    pred_df = pd.DataFrame(y_inv, index=pred_index, columns=target_cols)

    mask = filled.isna()
    filled = filled.fillna(pred_df)
    # Track which cells were imputed for downstream auditing.
    filled.attrs["imputed_mask"] = mask.loc[pred_index]
    return filled
