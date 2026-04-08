"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG we touch.

    TensorFlow is seeded lazily because importing it is slow and not
    every entry-point needs it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: WPS433 (lazy import)

        tf.random.set_seed(seed)
        # Best-effort determinism. Will be ignored on TF builds without it.
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    except ImportError:
        pass
