"""Top-level config that combines data processing, fine-tuning, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from crucible.data.config import DataProcessingConfig
from crucible.evaluation.config import EvaluationConfig
from crucible.templates import get_template_for_model
from crucible.tracking.config import ExperimentTrackingConfig
from crucible.training.config import FinetuningConfig


@dataclass
class CrucibleConfig:
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: ExperimentTrackingConfig = field(default_factory=ExperimentTrackingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CrucibleConfig:
        """Load a combined config from a single YAML file.

        If finetuning.model.name is set and data_processing.formatting.use_model_template
        is True, the data processing and evaluation templates are set from the
        model-template registry so they align with the chosen model.
        """
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, file_cfg)
        obj = OmegaConf.to_object(merged)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        _apply_model_templates(obj)
        return obj

    @classmethod
    def from_dict(cls, d: dict[Any, Any]) -> CrucibleConfig:
        """Build config from a dict (e.g. default config) and apply model templates."""
        schema = OmegaConf.structured(cls)
        merged = OmegaConf.merge(schema, OmegaConf.create(d))
        obj = OmegaConf.to_object(merged)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        _apply_model_templates(obj)
        return obj


def _apply_model_templates(config: CrucibleConfig) -> None:
    """When use_model_template is True and a model is set, align templates from registry."""
    if not config.finetuning.model.name:
        return
    if not getattr(config.data_processing.formatting, "use_model_template", True):
        return
    templates = get_template_for_model(config.finetuning.model.name)
    if templates is None:
        return
    config.data_processing.formatting.template = templates["formatting_template"]
    config.evaluation.input_template = templates["eval_input_template"]
