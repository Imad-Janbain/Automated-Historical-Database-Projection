"""Smoke tests: every registered model must build, compile and predict."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from deep_ts_imputer.models.factory import available_models, build_model
from deep_ts_imputer.utils.config import ModelConfig, TrainConfig


@pytest.mark.parametrize("name", available_models())
def test_model_builds_and_predicts(name):
    model_cfg = ModelConfig(name=name, units=8, num_layers=2, dropout=0.1, cnn_filters=8, use_attention=False)
    train_cfg = TrainConfig(learning_rate=1e-3, loss="mse")
    model = build_model(
        model_cfg, train_cfg,
        look_back=12, n_features_in=3, n_outputs=2,
    )
    x = np.random.random((4, 12, 3)).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (4, 2)


def test_cnn_bilstm_with_attention():
    model_cfg = ModelConfig(name="cnn_bilstm", units=8, num_layers=2, dropout=0.1, cnn_filters=8, use_attention=True)
    train_cfg = TrainConfig(learning_rate=1e-3, loss="mse")
    model = build_model(model_cfg, train_cfg, look_back=12, n_features_in=3, n_outputs=2)
    x = np.random.random((2, 12, 3)).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 2)
