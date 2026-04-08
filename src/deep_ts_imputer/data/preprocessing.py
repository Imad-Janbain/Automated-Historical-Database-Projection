"""Scaling and chronological splits.

Two principles drive this module:

1. **Fit only on training data.** Validation and test sets must never see
   the scaler during ``fit`` — otherwise we leak future information into
   the model.
2. **Splits are chronological.** Random shuffling is wrong for time
   series; the model must be evaluated on data strictly *after* what it
   trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

_SCALER_REGISTRY = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
}


def make_scaler(name: str):
    name = name.lower()
    if name not in _SCALER_REGISTRY:
        raise ValueError(
            f"Unknown scaler '{name}'. Available: {list(_SCALER_REGISTRY)}",
        )
    return _SCALER_REGISTRY[name]()


@dataclass
class ScaledSplits:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_index: pd.Index
    val_index: pd.Index
    test_index: pd.Index
    x_scaler: object
    y_scaler: object


def chronological_split(
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    train_split: float = 0.7,
    val_split: float = 0.8,
    scaler_name: str = "minmax",
) -> ScaledSplits:
    """Split a dataframe into train / val / test chronologically and scale it.

    ``train_split`` is the fraction of the *whole* series used for training
    plus validation. ``val_split`` is the fraction of *that* block used for
    actual training (the remainder becomes the validation set). The final
    ``1 - train_split`` of the series is held out as test.

    Example with defaults: 56% train / 14% val / 30% test.
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be in (0, 1)")
    if not 0 < train_split < 1:
        raise ValueError("train_split must be in (0, 1)")

    n = len(inputs)
    train_end = int(train_split * n)
    val_end = int(val_split * train_end)

    x_scaler = make_scaler(scaler_name)
    y_scaler = make_scaler(scaler_name)

    x_train_raw = inputs.iloc[:val_end]
    x_val_raw = inputs.iloc[val_end:train_end]
    x_test_raw = inputs.iloc[train_end:]

    y_train_raw = targets.iloc[:val_end]
    y_val_raw = targets.iloc[val_end:train_end]
    y_test_raw = targets.iloc[train_end:]

    x_train = x_scaler.fit_transform(x_train_raw)
    x_val = x_scaler.transform(x_val_raw)
    x_test = x_scaler.transform(x_test_raw)

    y_train = y_scaler.fit_transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    return ScaledSplits(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_index=x_train_raw.index,
        val_index=x_val_raw.index,
        test_index=x_test_raw.index,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
