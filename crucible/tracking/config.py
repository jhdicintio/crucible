"""Config for experiment tracking backend selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentTrackingConfig:
    backend: str = "sqlite"  # "sqlite" | "none" | future: "mlflow" | "wandb"
    run_name: str | None = None
    sqlite_path: str = "experiments.db"
