"""High-level training loop.

Wraps Keras ``model.fit`` with the callbacks we always want (early
stopping, checkpointing, optional MLflow logging) so scripts and the
Optuna objective stay short.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from deep_ts_imputer.utils.config import TrainConfig
from deep_ts_imputer.utils.logging import get_logger

LOGGER = get_logger(__name__)


def make_callbacks(
    train_cfg: TrainConfig,
    checkpoint_path: Path | None = None,
) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            ),
        )
    return callbacks


def fit_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_cfg: TrainConfig,
    checkpoint_path: Path | None = None,
    verbose: int = 2,
    extra_callbacks: list[tf.keras.callbacks.Callback] | None = None,
) -> tf.keras.callbacks.History:
    """Train a model and return the Keras history object."""
    callbacks = make_callbacks(train_cfg, checkpoint_path)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    LOGGER.info(
        "Training %s for up to %d epochs (batch=%d, lr=%.2g)",
        model.__class__.__name__,
        train_cfg.epochs,
        train_cfg.batch_size,
        train_cfg.learning_rate,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,  # chronological data → never shuffle
    )
    return history
