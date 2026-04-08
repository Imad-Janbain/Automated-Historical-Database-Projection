"""Exploratory data analysis for multivariate time series.

This module is the library counterpart of the one-off EDA scripts that
typically litter a research project. Every function takes a DataFrame and
an output directory and writes a PNG — no global state, no notebook side
effects, fully scriptable from CI.

Functions
---------
* :func:`plot_distributions` — per-column histogram + KDE grid
* :func:`plot_correlation_heatmap` — annotated Pearson correlation matrix
* :func:`plot_correlation_clustermap` — hierarchically-clustered
  correlation matrix (the dendrogram view from the original Paper-2 EDA)
* :func:`plot_missing_data` — missing-value matrix and per-column bar
* :func:`plot_timeseries_overview` — every column on its own subplot
* :func:`run_full_eda` — convenience wrapper that calls all of the above
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", font_scale=0.9)


def _ensure(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_distributions(df: pd.DataFrame, out_dir: Path, n_cols: int = 3) -> Path:
    """Histogram + KDE grid, one panel per numeric column."""
    out_dir = _ensure(out_dir)
    numeric = df.select_dtypes(include=np.number)
    n = len(numeric.columns)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, numeric.columns):
        sns.histplot(numeric[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Variable distributions", fontsize=12)
    fig.tight_layout()
    path = out_dir / "distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Annotated Pearson correlation matrix (RdBu, [-1, 1])."""
    out_dir = _ensure(out_dir)
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(corr), 0.8 + 0.5 * len(corr)))
    sns.heatmap(
        corr.round(2),
        cmap="RdBu_r",
        annot=True,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Pearson correlation matrix")
    fig.tight_layout()
    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_clustermap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Hierarchically-clustered correlation heatmap (dendrogram view)."""
    out_dir = _ensure(out_dir)
    corr = df.select_dtypes(include=np.number).corr()
    g = sns.clustermap(
        corr,
        method="complete",
        cmap="RdBu_r",
        annot=True,
        annot_kws={"size": 7},
        vmin=-1,
        vmax=1,
        figsize=(1.0 + 0.7 * len(corr), 1.0 + 0.7 * len(corr)),
    )
    g.fig.suptitle("Hierarchically-clustered correlations", y=1.02)
    path = out_dir / "correlation_clustermap.png"
    g.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    return path


def plot_missing_data(df: pd.DataFrame, out_dir: Path) -> Path:
    """Two-panel figure: missing-value matrix and per-column bar."""
    out_dir = _ensure(out_dir)
    numeric = df.select_dtypes(include=np.number)
    missing_mask = numeric.isna().to_numpy().astype(float)
    pct = numeric.isna().mean().sort_values(ascending=False) * 100

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    axes[0].imshow(missing_mask.T, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[0].set_yticks(range(len(numeric.columns)))
    axes[0].set_yticklabels(numeric.columns, fontsize=7)
    axes[0].set_xlabel("time index")
    axes[0].set_title("Missing-value matrix (black = NaN)")

    axes[1].barh(pct.index[::-1], pct.values[::-1], color="firebrick")
    axes[1].set_xlabel("% missing")
    axes[1].set_title("Per-column missingness")
    axes[1].tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    path = out_dir / "missing_data.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_timeseries_overview(df: pd.DataFrame, out_dir: Path) -> Path:
    """One subplot per column, sharing the time axis."""
    out_dir = _ensure(out_dir)
    numeric = df.select_dtypes(include=np.number)
    n = len(numeric.columns)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, numeric.columns):
        ax.plot(numeric.index, numeric[col], linewidth=0.6, color="navy")
        ax.set_ylabel(col, fontsize=7, rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("date")
    fig.suptitle("Time-series overview")
    fig.tight_layout()
    path = out_dir / "timeseries_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _group_columns_by_station(
    columns: list[str],
    stations: list[str],
) -> dict[str, list[str]]:
    """Group columns by which station name they contain.

    Matching is case-insensitive substring. A column belongs to the
    first station whose name appears in it.
    """
    out: dict[str, list[str]] = {s: [] for s in stations}
    for col in columns:
        lc = col.lower()
        for s in stations:
            if s.lower() in lc:
                out[s].append(col)
                break
    return out


# ---------------------------------------------------------------------------
# Depth-aware column parsing
# ---------------------------------------------------------------------------
# Many environmental archives — including the Lower Seine record used in
# Janbain et al. 2023 — measure the same parameter at TWO depths at the
# same station: a Surface probe and a Bottom probe. The two depths are
# physically and chemically distinct (the bottom layer can be saltier and
# more turbid than the surface during a flood tide) and any spatial
# analysis that ignores the distinction will mis-correlate the data.
#
# We treat depth as a first-class dimension throughout the spatial
# helpers. Column names follow the convention used in the paper:
#
#     {Parameter}_{Station}_{Depth}
#
# where ``{Depth}`` is one of ``Surface`` or ``Bottom``. Stations with a
# single probe simply omit the depth suffix.

DEFAULT_DEPTHS = ("Surface", "Bottom")


def parse_column_name(
    column: str,
    stations: list[str],
    parameters: list[str],
    depths: tuple[str, ...] = DEFAULT_DEPTHS,
) -> tuple[str | None, str | None, str | None]:
    """Best-effort split of ``column`` into ``(parameter, station, depth)``.

    Matching is case-insensitive substring. Any component that cannot be
    identified is returned as ``None`` instead of raising — callers
    typically just skip such columns.
    """
    lc = column.lower()
    parameter = next((p for p in parameters if p.lower() in lc), None)
    station = next((s for s in stations if s.lower() in lc), None)
    depth = next((d for d in depths if d.lower() in lc), None)
    return parameter, station, depth


def plot_station_correlation_grid(
    df: pd.DataFrame,
    stations: list[str],
    parameters: list[str],
    out_dir: Path,
    depths: tuple[str, ...] = DEFAULT_DEPTHS,
) -> Path:
    """Inter-station correlation per parameter, depth-aware.

    Produces a small-multiple figure: one heatmap per *parameter*. Rows
    and columns of each heatmap are ``(station, depth)`` pairs, so a
    station with both a Surface and a Bottom probe contributes two rows.
    Stations or depths that don't exist for a given parameter are
    silently dropped.

    This is the depth-aware spatial counterpart of
    :func:`plot_correlation_heatmap`. Use it whenever your archive has
    surface-vs-bottom probes — without it the analysis will silently
    smear the two layers together.
    """
    out_dir = _ensure(out_dir)
    n_params = len(parameters)
    n_cols = min(2, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for ax, param in zip(axes.ravel(), parameters):
        # Find every (station, depth) pair that matches this parameter.
        records: list[tuple[str, str, str]] = []  # (label, column_name, sort_key)
        for col in df.columns:
            p, s, d = parse_column_name(col, stations, [param], depths)
            if p is None or s is None:
                continue
            label = f"{s} ({d[0]})" if d else s
            sort_key = f"{stations.index(s):02d}_{d or ''}"
            records.append((label, col, sort_key))

        if len(records) < 2:
            ax.set_visible(False)
            continue

        records.sort(key=lambda r: r[2])
        labels = [r[0] for r in records]
        cols = [r[1] for r in records]

        sub = df[cols].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue
        corr = sub.corr()
        corr.index = corr.columns = labels

        sns.heatmap(
            corr.round(2),
            cmap="vlag",
            annot=True,
            annot_kws={"size": 8},
            vmin=-1, vmax=1, center=0,
            square=True,
            linewidths=0.5, linecolor="white",
            cbar=True,
            cbar_kws={"shrink": 0.7},
            ax=ax,
        )
        ax.set_title(param, fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    for ax in axes.ravel()[n_params:]:
        ax.set_visible(False)

    fig.suptitle("Inter-station correlation, per parameter (S = Surface, B = Bottom)", fontsize=12)
    fig.tight_layout()
    path = out_dir / "station_correlation_grid.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_surface_vs_bottom(
    df: pd.DataFrame,
    stations: list[str],
    parameters: list[str],
    out_dir: Path,
    n_samples: int | None = 3000,
) -> Path:
    """Compare surface and bottom probes for every (station, parameter) pair.

    For each station that has both a Surface and a Bottom column for a
    given parameter, plot the two series on a shared axis. The figure
    has one row per parameter and one column per station; cells where
    only one depth exists fall back to that single trace.

    This is the figure that answers "do the two probes agree, and where
    do they diverge?" — essential for any estuarine archive where the
    salt wedge can decouple the two layers.
    """
    out_dir = _ensure(out_dir)
    fig, axes = plt.subplots(
        len(parameters), len(stations),
        figsize=(2.8 * len(stations), 2.0 * len(parameters)),
        sharex=True,
        squeeze=False,
    )

    if n_samples is not None and len(df) > n_samples:
        df = df.iloc[:: max(1, len(df) // n_samples)]

    for i, param in enumerate(parameters):
        for j, station in enumerate(stations):
            ax = axes[i, j]
            surface_col = next(
                (c for c in df.columns
                 if param.lower() in c.lower()
                 and station.lower() in c.lower()
                 and "surface" in c.lower()),
                None,
            )
            bottom_col = next(
                (c for c in df.columns
                 if param.lower() in c.lower()
                 and station.lower() in c.lower()
                 and "bottom" in c.lower()),
                None,
            )

            plotted_any = False
            if surface_col is not None:
                ax.plot(df.index, df[surface_col], linewidth=0.6,
                        color="#1f77b4", label="Surface")
                plotted_any = True
            if bottom_col is not None:
                ax.plot(df.index, df[bottom_col], linewidth=0.6,
                        color="#d62728", label="Bottom", alpha=0.8)
                plotted_any = True

            if not plotted_any:
                ax.set_visible(False)
                continue

            if i == 0:
                ax.set_title(station, fontsize=9)
            if j == 0:
                ax.set_ylabel(param, fontsize=8)
            if i == 0 and j == len(stations) - 1:
                ax.legend(loc="upper right", fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Surface vs bottom probes, per (parameter, station)", fontsize=11)
    fig.tight_layout()
    path = out_dir / "surface_vs_bottom.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parameter_per_station(
    df: pd.DataFrame,
    stations: list[str],
    parameters: list[str],
    out_dir: Path,
    depth: str = "Surface",
    n_samples: int | None = 2000,
) -> Path:
    """Small-multiples grid: one row per parameter, one column per station.

    For estuarine datasets with multiple depths, ``depth`` selects which
    probe to plot (default ``"Surface"``). Use
    :func:`plot_surface_vs_bottom` instead when you want to see both
    depths overlaid.
    """
    out_dir = _ensure(out_dir)
    fig, axes = plt.subplots(
        len(parameters), len(stations),
        figsize=(2.6 * len(stations), 1.8 * len(parameters)),
        sharex=True,
        squeeze=False,
    )

    if n_samples is not None and len(df) > n_samples:
        df = df.iloc[:: max(1, len(df) // n_samples)]

    for i, param in enumerate(parameters):
        for j, station in enumerate(stations):
            ax = axes[i, j]
            match = next(
                (c for c in df.columns
                 if param.lower() in c.lower()
                 and station.lower() in c.lower()
                 and depth.lower() in c.lower()),
                None,
            )
            if match is None:
                # Fall back to any matching column ignoring depth.
                match = next(
                    (c for c in df.columns
                     if param.lower() in c.lower() and station.lower() in c.lower()),
                    None,
                )
            if match is None:
                ax.set_visible(False)
                continue
            ax.plot(df.index, df[match], linewidth=0.5, color="navy")
            if i == 0:
                ax.set_title(station, fontsize=9)
            if j == 0:
                ax.set_ylabel(param, fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Parameter dynamics per station ({depth} probes)", fontsize=11)
    fig.tight_layout()
    path = out_dir / "parameter_per_station.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def summarise(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write a CSV summary (count / mean / std / min / quantiles / max / %NaN)."""
    out_dir = _ensure(out_dir)
    numeric = df.select_dtypes(include=np.number)
    desc = numeric.describe().T
    desc["pct_missing"] = numeric.isna().mean() * 100
    path = out_dir / "summary_statistics.csv"
    desc.to_csv(path)
    return path


def run_full_eda(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Run every analysis above and return a name → path mapping."""
    out_dir = _ensure(out_dir)
    return {
        "summary": summarise(df, out_dir),
        "distributions": plot_distributions(df, out_dir),
        "correlation_heatmap": plot_correlation_heatmap(df, out_dir),
        "correlation_clustermap": plot_correlation_clustermap(df, out_dir),
        "missing_data": plot_missing_data(df, out_dir),
        "timeseries_overview": plot_timeseries_overview(df, out_dir),
    }
