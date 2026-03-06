"""Flyte workflow that composes the full data-processing pipeline."""

from __future__ import annotations

from pathlib import Path

from flytekit import workflow

from crucible.data.cleaning import clean_data
from crucible.data.config import DataProcessingConfig
from crucible.data.formatting import format_prompts
from crucible.data.loading import load_hf_dataset
from crucible.data.splitting import SplitResult, split_data


@workflow
def data_processing_pipeline(config: DataProcessingConfig) -> SplitResult:
    """Load → clean → split → format.

    Returns train / val / test DataFrames with a ``formatted_text`` column.
    """
    raw = load_hf_dataset(config=config.dataset)
    cleaned = clean_data(df=raw, config=config.cleaning)
    splits = split_data(df=cleaned, config=config.split)
    train = format_prompts(df=splits.train, config=config.formatting)
    val = format_prompts(df=splits.val, config=config.formatting)
    test = format_prompts(df=splits.test, config=config.formatting)
    return SplitResult(train=train, val=val, test=test)


def run_data_processing(
    config: DataProcessingConfig | None = None,
    config_path: str | Path | None = None,
) -> SplitResult:
    """Convenience entry point — accepts a config object *or* a YAML file path.

    * If both are ``None``, runs with default settings.
    * If ``config_path`` is given, loads and merges the YAML with schema defaults.
    * Passing both ``config`` and ``config_path`` is an error.
    """
    if config is not None and config_path is not None:
        raise ValueError("Provide config or config_path, not both")
    if config_path is not None:
        config = DataProcessingConfig.from_yaml(config_path)
    if config is None:
        config = DataProcessingConfig()
    return data_processing_pipeline(config=config)  # type: ignore[no-any-return]
