"""Top-level config that combines data processing, fine-tuning, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from crucible.data.config import DataProcessingConfig
from crucible.evaluation.config import EvaluationConfig
from crucible.training.config import FinetuningConfig


@dataclass
class CrucibleConfig:
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CrucibleConfig:
        """Load a combined config from a single YAML file."""
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, file_cfg)
        obj = OmegaConf.to_object(merged)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
