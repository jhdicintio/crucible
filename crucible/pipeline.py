"""End-to-end pipeline: data processing -> fine-tuning."""

from __future__ import annotations

from pathlib import Path

from flytekit import workflow

from crucible.config import CrucibleConfig
from crucible.data.pipeline import data_processing_pipeline
from crucible.training.pipeline import FinetuneResult, finetune


@workflow
def full_pipeline(config: CrucibleConfig) -> FinetuneResult:
    """Load & clean data, then fine-tune a model in a single workflow."""
    splits = data_processing_pipeline(config=config.data_processing)
    return finetune(  # type: ignore[no-any-return]
        train_df=splits.train,
        val_df=splits.val,
        config=config.finetuning,
    )


def run_full_pipeline(
    config: CrucibleConfig | None = None,
    config_path: str | Path | None = None,
) -> FinetuneResult:
    """Convenience entry point — accepts a config or YAML path."""
    if config is not None and config_path is not None:
        raise ValueError("Provide config or config_path, not both")
    if config_path is not None:
        config = CrucibleConfig.from_yaml(config_path)
    if config is None:
        config = CrucibleConfig()
    return full_pipeline(config=config)  # type: ignore[no-any-return]
