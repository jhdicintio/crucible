"""Factory for creating the configured experiment tracker."""

from __future__ import annotations

from crucible.tracking.backends.sqlite_backend import SQLiteTracker
from crucible.tracking.config import ExperimentTrackingConfig
from crucible.tracking.protocol import ExperimentTracker


def get_tracker(config: ExperimentTrackingConfig) -> ExperimentTracker | None:
    """Return an ExperimentTracker for the given config, or None if backend is 'none'."""
    if config.backend == "none":
        return None
    if config.backend == "sqlite":
        return SQLiteTracker(db_path=config.sqlite_path)
    raise ValueError(f"Unknown tracking backend: {config.backend!r}")
