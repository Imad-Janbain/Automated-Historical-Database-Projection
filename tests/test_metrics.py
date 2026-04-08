import numpy as np

from deep_ts_imputer.evaluation.metrics import compute_all, kge, nse, r2_score, rmse


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = compute_all(y, y)
    assert out["rmse"] == 0.0
    assert out["mae"] == 0.0
    assert abs(out["r2"] - 1.0) < 1e-9
    assert abs(out["nse"] - 1.0) < 1e-9
    assert abs(out["kge"] - 1.0) < 1e-9


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert rmse(y_true, y_pred) == 1.0


def test_r2_baseline_is_zero():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert abs(r2_score(y_true, y_pred)) < 1e-9
