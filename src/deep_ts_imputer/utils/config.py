"""Typed configuration loader.

Configs are plain YAML files. They are parsed into nested dataclasses so
that downstream code gets autocompletion and clear error messages instead
of dictionary key-errors deep in the stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Where the data lives and how to slice it."""

    path: str
    date_column: str = "Dates"
    input_features: list[str] = field(default_factory=list)
    target_features: list[str] = field(default_factory=list)
    date_begin: str | None = None
    date_end: str | None = None
    interpolate_missing: bool = True
    train_split: float = 0.7
    val_split: float = 0.8  # fraction of train used as train (rest -> val)
    scaler: str = "minmax"  # minmax | standard | robust
    units: dict[str, str] = field(default_factory=dict)
    column_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class WindowConfig:
    look_back: int = 24
    horizon: int = 1


@dataclass
class ModelConfig:
    name: str = "bilstm"  # see deep_ts_imputer.models.factory
    units: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    cnn_filters: int = 32  # only used by cnn_bilstm
    use_attention: bool = False  # only used by cnn_bilstm


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 15
    optimizer: str = "adam"
    loss: str = "mse"


@dataclass
class TuneConfig:
    enabled: bool = False
    n_trials: int = 30
    timeout: int | None = None
    sampler: str = "tpe"
    pruner: str = "median"
    direction: str = "minimize"
    study_name: str = "deep_ts_imputer"
    storage: str | None = None  # e.g. sqlite:///optuna.db


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=lambda: DataConfig(path=""))
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)


def _build(cls, payload: dict[str, Any]):
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    if payload is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in field_names:
            continue
        field_type = cls.__dataclass_fields__[key].type  # type: ignore[attr-defined]
        if hasattr(field_type, "__dataclass_fields__"):
            kwargs[key] = _build(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a typed `Config`."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    # Resolve nested dataclasses manually because of forward-ref types.
    cfg = Config()
    if "seed" in raw:
        cfg.seed = raw["seed"]
    if "output_dir" in raw:
        cfg.output_dir = raw["output_dir"]
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])
    if "window" in raw:
        cfg.window = WindowConfig(**raw["window"])
    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "train" in raw:
        cfg.train = TrainConfig(**raw["train"])
    if "tune" in raw:
        cfg.tune = TuneConfig(**raw["tune"])
    return cfg
