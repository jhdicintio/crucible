"""Structured configs for the data processing pipeline.

These dataclasses serve double duty — they're valid OmegaConf structured configs
(for Hydra YAML parsing) and valid Flyte task/workflow input types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class DatasetConfig:
    name: str = "takala/financial_phrasebank"
    split: str = "train"
    sample_size: int | None = None
    sample_strategy: str = "random"
    sample_seed: int = 42
    # For stratified: column to stratify by (e.g. label, output) so subset preserves distribution
    sample_stratify_column: str | None = None
    # For diversity / quality: columns to concatenate for embedding (e.g. ["instruction", "input"])
    sample_text_columns: list[str] | None = None
    sample_embedding_model: str = "all-MiniLM-L6-v2"
    # Quality: drop examples outside [min_chars, max_chars]; drop near-duplicates (cos sim > thresh)
    sample_quality_min_chars: int = 20
    sample_quality_max_chars: int = 4000
    sample_quality_dedup_threshold: float = 0.95


@dataclass
class CleaningConfig:
    drop_nans: bool = True
    drop_duplicates: bool = True
    strip_whitespace: bool = True
    lowercase: bool = False
    columns: list[str] = field(default_factory=list)


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    stratify_column: str | None = None


@dataclass
class FormattingConfig:
    template: str = (
        "### Instruction:\n"
        "Classify the sentiment of the following financial text "
        "as positive, negative, or neutral.\n\n"
        "### Input:\n"
        "{sentence}\n\n"
        "### Response:\n"
        "{label}"
    )
    use_model_template: bool = True
    system_prompt: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    output_column: str = "formatted_text"


@dataclass
class DataProcessingConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataProcessingConfig:
        """Load config from a YAML file, merging with schema defaults."""
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, file_cfg)
        obj = OmegaConf.to_object(merged)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
