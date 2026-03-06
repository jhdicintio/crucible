"""Flyte tasks and workflow for model fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from flytekit import task, workflow

from crucible.training.config import FinetuningConfig
from crucible.training.model import CausalLMModel


@dataclass
class FinetuneResult:
    model_path: str
    train_loss: float
    eval_loss: float
    log_file: str


@task
def finetune(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: FinetuningConfig,
) -> FinetuneResult:
    """Set up a model, fine-tune it, save to disk, and return summary metrics."""
    model = CausalLMModel(config)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    metrics: dict[str, Any] = model.train(train_ds, val_ds)
    model.save(config.training.output_dir)

    return FinetuneResult(
        model_path=config.training.output_dir,
        train_loss=float(metrics.get("train_loss", 0.0)),
        eval_loss=float(metrics.get("eval_loss", 0.0)),
        log_file=config.training.log_file,
    )


@workflow
def finetuning_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: FinetuningConfig,
) -> FinetuneResult:
    """Thin workflow wrapper — lets Flyte orchestrate the fine-tune task."""
    return finetune(train_df=train_df, val_df=val_df, config=config)  # type: ignore[no-any-return]


def run_finetuning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: FinetuningConfig | None = None,
    config_path: str | Path | None = None,
) -> FinetuneResult:
    """Convenience entry point — accepts a config object *or* a YAML path."""
    if config is not None and config_path is not None:
        raise ValueError("Provide config or config_path, not both")
    if config_path is not None:
        config = FinetuningConfig.from_yaml(config_path)
    if config is None:
        config = FinetuningConfig()
    return finetuning_pipeline(train_df=train_df, val_df=val_df, config=config)  # type: ignore[no-any-return]
