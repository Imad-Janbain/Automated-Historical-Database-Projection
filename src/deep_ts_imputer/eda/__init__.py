"""Exploratory data analysis subpackage."""

from deep_ts_imputer.eda.eda import (
    DEFAULT_DEPTHS,
    parse_column_name,
    plot_correlation_clustermap,
    plot_correlation_heatmap,
    plot_distributions,
    plot_missing_data,
    plot_parameter_per_station,
    plot_station_correlation_grid,
    plot_surface_vs_bottom,
    plot_timeseries_overview,
    run_full_eda,
    summarise,
)

__all__ = [
    "DEFAULT_DEPTHS",
    "parse_column_name",
    "plot_correlation_clustermap",
    "plot_correlation_heatmap",
    "plot_distributions",
    "plot_missing_data",
    "plot_parameter_per_station",
    "plot_station_correlation_grid",
    "plot_surface_vs_bottom",
    "plot_timeseries_overview",
    "run_full_eda",
    "summarise",
]
