import numpy as np
import pytest

from deep_ts_imputer.data.windowing import sliding_window


def test_sliding_window_shapes():
    inputs = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    targets = np.arange(100 * 2, dtype=np.float32).reshape(100, 2)
    X, y = sliding_window(inputs, targets, look_back=10, horizon=1)
    assert X.shape == (90, 10, 3)
    assert y.shape == (90, 2)


def test_sliding_window_horizon_gt1():
    inputs = np.zeros((50, 1), dtype=np.float32)
    targets = np.zeros((50, 1), dtype=np.float32)
    X, y = sliding_window(inputs, targets, look_back=5, horizon=3)
    assert X.shape == (43, 5, 1)
    assert y.shape == (43, 3)


def test_sliding_window_rejects_short_series():
    inputs = np.zeros((5, 1), dtype=np.float32)
    targets = np.zeros((5, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        sliding_window(inputs, targets, look_back=10, horizon=1)


def test_sliding_window_first_sample_alignment():
    inputs = np.arange(20).reshape(20, 1).astype(np.float32)
    targets = (np.arange(20).reshape(20, 1) * 10).astype(np.float32)
    X, y = sliding_window(inputs, targets, look_back=4, horizon=1)
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(y[0], [40])
