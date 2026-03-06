"""MetricProtocol — structural interface for evaluation metrics."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricProtocol(Protocol):
    """Every evaluation metric must expose a *name* and a *compute* method.

    ``compute`` receives parallel lists of model predictions and ground-truth
    references, and returns a dict mapping sub-metric names to float values
    (e.g. ``{"rouge1": 0.45, "rougeL": 0.38}``).
    """

    @property
    def name(self) -> str: ...

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Return sub-metric name -> value. Some metrics (e.g. refusal) use metadata."""
        ...
