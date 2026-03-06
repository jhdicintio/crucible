"""ModelProtocol — structural interface every trainable model must satisfy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Self

from datasets import Dataset


class ModelProtocol(Protocol):
    """Minimal contract for models that can be trained, persisted, and served."""

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Fine-tune the model. Returns a dict of training metrics."""
        ...

    def predict(self, inputs: list[str], max_new_tokens: int = 256) -> list[str]:
        """Generate text completions for a batch of inputs."""
        ...

    def save(self, path: str | Path) -> None:
        """Persist the model and tokenizer to *path*."""
        ...

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a model from a previously saved directory."""
        ...
