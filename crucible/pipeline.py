"""End-to-end pipeline: data processing -> fine-tuning -> evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from flytekit import dynamic, task

from crucible.config import CrucibleConfig
from crucible.data.pipeline import data_processing_pipeline
from crucible.evaluation.pipeline import EvaluationResult, evaluate
from crucible.training.pipeline import FinetuneResult, finetune


@task
def _model_path_from_finetune(ft_result: FinetuneResult) -> str:
    """Extract model_path so evaluate sees a resolved value."""
    return ft_result.model_path


@dataclass
class PipelineResult:
    finetune: FinetuneResult
    evaluation: EvaluationResult = field(default_factory=lambda: EvaluationResult(output_file=""))


@task
def _make_pipeline_result(
    ft_result: FinetuneResult, eval_result: EvaluationResult
) -> PipelineResult:
    """Build PipelineResult from resolved task outputs."""
    return PipelineResult(finetune=ft_result, evaluation=eval_result)


@task
def load_config(config_path: str) -> CrucibleConfig:
    """Load CrucibleConfig from a YAML file (Flyte task so workflow can use it)."""
    return CrucibleConfig.from_yaml(config_path)


@dynamic
def full_pipeline(config_path: str) -> PipelineResult:
    """Load config from YAML, then run data → fine-tune → evaluation.

    Use this as the entrypoint: pass a path to an experiment YAML.
    """
    config = load_config(config_path=config_path)
    splits = data_processing_pipeline(config=config.data_processing)
    ft_result = finetune(
        train_df=splits.train,
        val_df=splits.val,
        config=config.finetuning,
        tracking_config=config.tracking,
        full_config=config,
    )
    model_path = _model_path_from_finetune(ft_result=ft_result)
    eval_result = evaluate(
        test_df=splits.test,
        model_path=model_path,
        config=config.evaluation,
    )
    return cast(PipelineResult, _make_pipeline_result(ft_result=ft_result, eval_result=eval_result))


def run_full_pipeline(
    config_path: str | Path | None = None,
) -> PipelineResult:
    """Run the full pipeline with the workflow as entrypoint.

    Loads the given YAML path inside the workflow. Pass the path to your
    experiment config (e.g. conf/experiments/experiment_0_baseline.yaml).
    """
    if config_path is None:
        raise ValueError("config_path is required")
    return cast(PipelineResult, full_pipeline(config_path=str(config_path)))


if __name__ == "__main__":
    result = run_full_pipeline(config_path="conf/experiments/experiment_0_baseline.yaml")
    print(result)
