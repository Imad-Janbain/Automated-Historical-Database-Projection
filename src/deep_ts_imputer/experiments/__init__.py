"""Phase 1 grid search and Phase 2 progressive reconstruction.

The submodules are imported lazily so that the database can be used in
contexts where TensorFlow is not available (tests, analysis notebooks,
CI lint passes).
"""

from deep_ts_imputer.experiments.database import ResultsDatabase, TrialRecord

__all__ = [
    "ResultsDatabase",
    "TrialRecord",
    "run_grid",
    "load_grid_spec",
    "run_progressive_reconstruction",
    "ProgressiveResult",
    "ReconstructionStep",
]


def __getattr__(name):
    if name in {"run_grid", "load_grid_spec"}:
        from deep_ts_imputer.experiments.grid import load_grid_spec, run_grid
        return {"run_grid": run_grid, "load_grid_spec": load_grid_spec}[name]
    if name in {"run_progressive_reconstruction", "ProgressiveResult", "ReconstructionStep"}:
        from deep_ts_imputer.experiments.progressive import (
            ProgressiveResult,
            ReconstructionStep,
            run_progressive_reconstruction,
        )
        return {
            "run_progressive_reconstruction": run_progressive_reconstruction,
            "ProgressiveResult": ProgressiveResult,
            "ReconstructionStep": ReconstructionStep,
        }[name]
    raise AttributeError(f"module 'deep_ts_imputer.experiments' has no attribute {name!r}")
