"""Experiment tracker protocol — pluggable backends (SQLite, MLflow, WandB)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    """Interface for experiment tracking. Implementations: SQLite, MLflow, WandB."""

    @property
    def run_id(self) -> str | None:
        """Current run id, or None if no active run."""
        ...

    def start_run(
        self,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Start a new run. Returns run_id."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters (e.g. full config)."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics, optionally at a given step."""
        ...

    def log_artifact(self, path: str | Path) -> None:
        """Log a file or directory as an artifact."""
        ...

    def end_run(self) -> None:
        """End the current run."""
        ...
