"""Regression metrics commonly used in hydrology and forecasting.

The Seine Paper-2 results are reported in terms of RMSE, MAE, R^2, and
Nash–Sutcliffe Efficiency (NSE), so we expose all of them here plus
Kling-Gupta Efficiency (KGE), which is a more robust hydrological score.
"""

from __future__ import annotations

import numpy as np


def _flatten(y: np.ndarray) -> np.ndarray:
    return np.asarray(y).reshape(-1)


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _flatten(y_true), _flatten(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _flatten(y_true), _flatten(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = _flatten(y_true), _flatten(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def nse(y_true, y_pred) -> float:
    """Nash–Sutcliffe Efficiency (algebraically equal to R^2 here)."""
    return r2_score(y_true, y_pred)


def kge(y_true, y_pred) -> float:
    """Kling–Gupta Efficiency (Gupta et al., 2009)."""
    y_true, y_pred = _flatten(y_true), _flatten(y_pred)
    if np.std(y_true) == 0 or np.mean(y_true) == 0:
        return float("nan")
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(np.std(y_pred) / np.std(y_true))
    beta = float(np.mean(y_pred) / np.mean(y_true))
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def compute_all(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "nse": nse(y_true, y_pred),
        "kge": kge(y_true, y_pred),
    }
