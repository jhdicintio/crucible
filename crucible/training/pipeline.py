"""Flyte tasks and workflow for model fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from flytekit import task, workflow

from crucible.config import CrucibleConfig
from crucible.tracking.config import ExperimentTrackingConfig
from crucible.tracking.factory import get_tracker
from crucible.tracking.protocol import ExperimentTracker
from crucible.training.config import FinetuningConfig
from crucible.training.model import CausalLMModel


@dataclass
class FinetuneResult:
    model_path: str
    train_loss: float
    eval_loss: float
    log_file: str
    run_id: str | None = None


@task
def finetune(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: FinetuningConfig,
    tracking_config: ExperimentTrackingConfig | None = None,
    full_config: CrucibleConfig | None = None,
) -> FinetuneResult:
    """Set up a model, fine-tune it, save to disk, and return summary metrics.

    If tracking_config is provided and backend is not 'none', logs the run
    (including full_config_dict) and training metrics to the configured tracker.
    """
    model = CausalLMModel(config)
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    tracker: ExperimentTracker | None = None
    run_id: str | None = None
    if tracking_config is not None and tracking_config.backend != "none":
        tracker = get_tracker(tracking_config)
        if tracker is not None:
            run_name = tracking_config.run_name or config.model.name
            cfg_dict: dict[str, Any] | None = (
                asdict(full_config) if full_config is not None else None
            )
            run_id = tracker.start_run(run_name=run_name, config=cfg_dict)
            assert run_id is not None

    metrics: dict[str, Any]
    try:
        metrics = model.train(train_ds, val_ds, tracker=tracker)
        model.save(config.training.output_dir)
        if tracker is not None:
            tracker.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
            )
            tracker.log_artifact(config.training.output_dir)
    finally:
        if tracker is not None:
            tracker.end_run()

    return FinetuneResult(
        model_path=config.training.output_dir,
        train_loss=float(metrics.get("train_loss", 0.0)),
        eval_loss=float(metrics.get("eval_loss", 0.0)),
        log_file=config.training.log_file,
        run_id=run_id,
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
