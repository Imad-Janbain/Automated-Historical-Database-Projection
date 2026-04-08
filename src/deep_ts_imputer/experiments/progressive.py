"""Phase 2 — progressive historical reconstruction.

Implements the loop sketched in Figure 6 of Janbain et al. (Water 2023).
Starting from the set of features that exist over the long historical
window (typically water levels alone), the algorithm:

1. Queries the :class:`ResultsDatabase` for every target whose required
   inputs are a subset of the currently-available features.
2. Picks the (target, model, input-combination) with the best Phase 1
   metric (default: highest R²).
3. Loads that model and its scalers, runs sliding-window inference over
   the long historical record, and writes the predictions into the
   target column.
4. Adds the newly-reconstructed target to the available-feature set.
5. Repeats until every requested target has been reconstructed (or no
   feasible candidate remains, which means the user-supplied database
   does not cover some target with the inputs available).

The order in which targets get reconstructed is therefore *learned from
the data*, not hard-coded — exactly as in the paper, where conductivity
and dissolved oxygen typically come before turbidity because they
correlate more strongly with water level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from deep_ts_imputer.experiments.database import ResultsDatabase, TrialRecord
from deep_ts_imputer.utils.logging import get_logger

LOGGER = get_logger("progressive")


@dataclass
class ReconstructionStep:
    step: int
    target: str
    chosen_record: TrialRecord
    metric_value: float
    n_filled: int


@dataclass
class ProgressiveResult:
    reconstructed: pd.DataFrame
    order: list[ReconstructionStep] = field(default_factory=list)

    def to_summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "step": s.step,
                    "target": s.target,
                    "model": s.chosen_record.model_name,
                    "n_inputs": len(s.chosen_record.input_features),
                    "inputs": ", ".join(s.chosen_record.input_features),
                    "r2_phase1": s.chosen_record.metrics.get("r2"),
                    "rmse_phase1": s.chosen_record.metrics.get("rmse"),
                    "n_filled": s.n_filled,
                }
                for s in self.order
            ],
        )


def _predict_full_series(
    model: tf.keras.Model,
    inputs_df: pd.DataFrame,
    x_scaler,
    y_scaler,
    look_back: int,
) -> pd.Series:
    """Run sliding-window inference along the entire ``inputs_df``."""
    x = x_scaler.transform(inputs_df.to_numpy())
    n = len(x)
    n_samples = n - look_back
    if n_samples <= 0:
        raise ValueError(
            f"Series too short ({n}) for look_back={look_back}",
        )
    X = np.stack([x[i : i + look_back] for i in range(n_samples)]).astype(np.float32)
    y_scaled = model.predict(X, verbose=0)
    y_inv = y_scaler.inverse_transform(y_scaled).reshape(-1)

    # Predictions align with timestamps starting at look_back.
    out = pd.Series(np.nan, index=inputs_df.index, name="prediction")
    out.iloc[look_back : look_back + n_samples] = y_inv
    return out


def run_progressive_reconstruction(
    db: ResultsDatabase,
    series: pd.DataFrame,
    initial_available: list[str],
    targets_to_reconstruct: list[str],
    look_back: int,
    metric: str = "r2",
    overwrite_observed: bool = False,
) -> ProgressiveResult:
    """Iteratively reconstruct ``targets_to_reconstruct``.

    Parameters
    ----------
    db:
        Phase-1 results database.
    series:
        DataFrame indexed by date covering the **historical** window
        (e.g. 1990–2022 for the Seine). Columns named in
        ``initial_available`` must be fully populated. Target columns may
        be entirely missing — they will be added.
    initial_available:
        Feature columns whose data covers the full historical window.
    targets_to_reconstruct:
        The water-quality columns to fill in. Order is *not* respected;
        the algorithm picks an order based on what's feasible at each step.
    look_back:
        Sliding-window length used during Phase 1.
    metric:
        Metric column in the database used to rank candidates.
    overwrite_observed:
        If False (default), real measurements that already exist in the
        target column are preserved; only NaNs are filled. If True, the
        model output replaces every value.
    """
    available = set(initial_available)
    remaining = set(targets_to_reconstruct)
    reconstructed = series.copy()

    # Make sure target columns exist as NaN columns if they're not present.
    for t in remaining:
        if t not in reconstructed.columns:
            reconstructed[t] = np.nan

    order: list[ReconstructionStep] = []
    step = 0
    higher_is_better = metric.lower() in {"r2", "nse", "kge"}

    while remaining:
        step += 1
        # Find the best feasible record across all remaining targets.
        candidates: list[tuple[str, TrialRecord]] = []
        for target in remaining:
            rec = db.best_feasible(target, available, metric=metric)
            if rec is not None:
                candidates.append((target, rec))

        if not candidates:
            LOGGER.warning(
                "Step %d: no feasible candidate for any of %s. Stopping.",
                step, sorted(remaining),
            )
            break

        # Global best across targets.
        candidates.sort(
            key=lambda tr: tr[1].metrics.get(metric, -np.inf),
            reverse=higher_is_better,
        )
        chosen_target, chosen_record = candidates[0]
        chosen_inputs = list(chosen_record.input_features)
        LOGGER.info(
            "Step %d: target=%s model=%s inputs=%d %s=%.3f",
            step, chosen_target, chosen_record.model_name, len(chosen_inputs),
            metric, chosen_record.metrics[metric],
        )

        # Load model + scalers and run inference over the historical window.
        model = tf.keras.models.load_model(chosen_record.model_path)
        x_scaler = joblib.load(chosen_record.x_scaler_path)
        y_scaler = joblib.load(chosen_record.y_scaler_path)
        prediction = _predict_full_series(
            model,
            reconstructed[chosen_inputs],
            x_scaler, y_scaler,
            look_back=look_back,
        )

        target_col = reconstructed[chosen_target]
        if overwrite_observed:
            mask = prediction.notna()
        else:
            mask = target_col.isna() & prediction.notna()
        n_filled = int(mask.sum())
        reconstructed.loc[mask, chosen_target] = prediction[mask]

        order.append(
            ReconstructionStep(
                step=step,
                target=chosen_target,
                chosen_record=chosen_record,
                metric_value=chosen_record.metrics[metric],
                n_filled=n_filled,
            ),
        )
        available.add(chosen_target)
        remaining.remove(chosen_target)

    return ProgressiveResult(reconstructed=reconstructed, order=order)
