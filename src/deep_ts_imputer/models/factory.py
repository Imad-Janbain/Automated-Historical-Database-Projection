"""Model factory: turn a `ModelConfig` into a compiled Keras model."""

from __future__ import annotations

import tensorflow as tf

from deep_ts_imputer.models.architectures import (
    build_bigru,
    build_bilstm,
    build_cnn_bilstm,
    build_gru,
    build_lstm,
)
from deep_ts_imputer.utils.config import ModelConfig, TrainConfig

_MODEL_REGISTRY = {
    "lstm": build_lstm,
    "bilstm": build_bilstm,
    "gru": build_gru,
    "bigru": build_bigru,
    "cnn_bilstm": build_cnn_bilstm,
}


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def build_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    look_back: int,
    n_features_in: int,
    n_outputs: int,
) -> tf.keras.Model:
    """Build, compile and return a Keras model from typed configs."""
    name = model_cfg.name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_cfg.name}'. Available: {available_models()}",
        )

    builder = _MODEL_REGISTRY[name]
    kwargs = dict(
        look_back=look_back,
        n_features_in=n_features_in,
        n_outputs=n_outputs,
        units=model_cfg.units,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
    )
    if name == "cnn_bilstm":
        kwargs["cnn_filters"] = model_cfg.cnn_filters
        kwargs["use_attention"] = model_cfg.use_attention

    model = builder(**kwargs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg.learning_rate),
        loss=train_cfg.loss,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model
