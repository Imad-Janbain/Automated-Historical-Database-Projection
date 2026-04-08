"""Convert flat time series into supervised (X, y) tensors."""

from __future__ import annotations

import numpy as np


def sliding_window(
    inputs: np.ndarray,
    targets: np.ndarray,
    look_back: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a supervised dataset using a sliding window.

    For each step ``t`` we take ``inputs[t : t + look_back]`` as the
    predictor sequence and ``targets[t + look_back : t + look_back + horizon]``
    as the target.

    Parameters
    ----------
    inputs:
        2-D array of shape ``(time, n_features_in)``.
    targets:
        2-D array of shape ``(time, n_features_out)``.
    look_back:
        Number of past steps fed to the model.
    horizon:
        Number of future steps to predict (>= 1).

    Returns
    -------
    X : ndarray of shape (samples, look_back, n_features_in)
    y : ndarray of shape (samples, horizon * n_features_out)
    """
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("inputs and targets must be 2-D arrays")
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same length")
    if look_back < 1 or horizon < 1:
        raise ValueError("look_back and horizon must be >= 1")

    n_samples = len(inputs) - look_back - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Series too short ({len(inputs)}) for look_back={look_back} "
            f"and horizon={horizon}",
        )

    n_features_in = inputs.shape[1]
    n_features_out = targets.shape[1]

    X = np.empty((n_samples, look_back, n_features_in), dtype=np.float32)
    y = np.empty((n_samples, horizon * n_features_out), dtype=np.float32)

    for i in range(n_samples):
        X[i] = inputs[i : i + look_back]
        y[i] = targets[i + look_back : i + look_back + horizon].reshape(-1)

    return X, y
