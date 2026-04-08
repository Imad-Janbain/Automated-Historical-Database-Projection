"""Phase 2 — progressive historical reconstruction.

Loads a previously-built results database (Phase 1 output), takes a
historical CSV that contains the *initially-available* features over the
full time window, and iteratively reconstructs every requested target
column by repeatedly picking the highest-scoring feasible model from the
database.

Example::

    python scripts/run_progressive.py \\
        --db outputs/seine_grid/results.csv \\
        --series data/seine_water_levels_1990_2022.csv \\
        --date-column Dates \\
        --look-back 48 \\
        --available Water_level_Honfleur Water_level_Tancarville Water_level_Caudebec Water_level_Duclair Water_level_Rouen \\
        --targets Conductivity_Tancarville_Surface Dissolved_Oxygen_Tancarville_Surface Turbidity_Tancarville_Bottom \\
        --output outputs/seine_grid/historical_reconstruction.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_ts_imputer.data.dataset import load_timeseries
from deep_ts_imputer.experiments import (
    ResultsDatabase,
    run_progressive_reconstruction,
)
from deep_ts_imputer.utils.logging import get_logger

LOGGER = get_logger("run_progressive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Phase-1 results CSV")
    parser.add_argument("--series", type=Path, required=True, help="Historical CSV (contains initial features)")
    parser.add_argument("--date-column", default="Dates")
    parser.add_argument("--look-back", type=int, required=True)
    parser.add_argument("--available", nargs="+", required=True, help="Initial available features")
    parser.add_argument("--targets", nargs="+", required=True, help="Targets to reconstruct progressively")
    parser.add_argument("--metric", default="r2", help="DB column used to rank candidates")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the reconstructed CSV")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Where to write a CSV summary of the reconstruction order (default: alongside --output)",
    )
    args = parser.parse_args()

    db = ResultsDatabase(args.db)
    LOGGER.info("Loaded database with %d records", len(db))

    series = load_timeseries(
        args.series,
        date_column=args.date_column,
        interpolate_missing=False,
    )
    LOGGER.info("Loaded historical series: %d rows × %d cols", *series.shape)

    result = run_progressive_reconstruction(
        db=db,
        series=series,
        initial_available=args.available,
        targets_to_reconstruct=args.targets,
        look_back=args.look_back,
        metric=args.metric,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.reconstructed.to_csv(args.output)
    LOGGER.info("Wrote reconstructed series to %s", args.output)

    summary_path = args.summary or args.output.with_name(args.output.stem + "_order.csv")
    summary_df = result.to_summary_df()
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Wrote reconstruction order to %s", summary_path)

    # Render the waterfall figure alongside the audit trail.
    try:
        from deep_ts_imputer.evaluation.plots import plot_progressive_waterfall
        waterfall_path = args.output.with_name(args.output.stem + "_waterfall.png")
        plot_progressive_waterfall(summary_df, waterfall_path)
        LOGGER.info("Wrote waterfall figure to %s", waterfall_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not render waterfall: %s", exc)

    LOGGER.info("Reconstruction order:")
    for s in result.order:
        LOGGER.info(
            "  %d. %-40s model=%-12s %s=%.3f filled=%d",
            s.step, s.target, s.chosen_record.model_name,
            args.metric, s.metric_value, s.n_filled,
        )


if __name__ == "__main__":
    main()
