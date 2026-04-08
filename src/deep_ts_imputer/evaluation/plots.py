"""Plot helpers used by the evaluation and reconstruction scripts.

We deliberately keep these very small and matplotlib-only — the goal is
reproducible figures for a paper or a portfolio repo, not a dashboarding
toolkit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "lightgray",
        "font.size": 11,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    },
)


def plot_history(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training history")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_predictions(
    index: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    out_path: Path,
    title: str = "Predictions vs observations",
    units: dict[str, str] | None = None,
) -> None:
    from deep_ts_imputer.utils.units import label_with_unit

    n_targets = len(target_names)
    fig, axes = plt.subplots(n_targets, 1, figsize=(10, 2.5 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]
    n = min(len(index), len(y_true))
    for i, name in enumerate(target_names):
        axes[i].plot(index[:n], y_true[:n, i], label="observed", linewidth=1.0)
        axes[i].plot(index[:n], y_pred[:n, i], label="predicted", linewidth=1.0, alpha=0.8)
        axes[i].set_ylabel(label_with_unit(name, units))
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("date")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    out_path: Path,
    units: dict[str, str] | None = None,
) -> None:
    from deep_ts_imputer.utils.units import label_with_unit

    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4), squeeze=False)
    for i, name in enumerate(target_names):
        ax = axes[0, i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=4, alpha=0.4)
        lims = [
            min(y_true[:, i].min(), y_pred[:, i].min()),
            max(y_true[:, i].max(), y_pred[:, i].max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1)
        labelled = label_with_unit(name, units)
        ax.set_xlabel(f"observed {labelled}")
        ax.set_ylabel(f"predicted {labelled}")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_progressive_waterfall(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Visualise the order of a progressive reconstruction run.

    ``summary_df`` is the output of
    :meth:`ProgressiveResult.to_summary_df`. The figure has one bar per
    reconstruction step, ordered chronologically, coloured by Phase-1 R²,
    with the chosen model name annotated above each bar.
    """
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(summary_df)), 4.5))
    x = np.arange(len(summary_df))
    r2 = summary_df["r2_phase1"].astype(float).values
    bars = ax.bar(x, r2, color="steelblue", edgecolor="navy")

    for bar, model_name in zip(bars, summary_df["model"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            str(model_name),
            ha="center", va="bottom", fontsize=8,
        )

    labels = [f"{row.step}. {row.target}" for row in summary_df.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Phase-1 R²")
    ax.set_ylim(0, 1.05)
    ax.set_title("Progressive reconstruction — order and model choice")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
