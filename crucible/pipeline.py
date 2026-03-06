"""End-to-end pipeline: data processing -> fine-tuning -> evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from flytekit import workflow

from crucible.config import CrucibleConfig
from crucible.data.pipeline import data_processing_pipeline
from crucible.evaluation.pipeline import EvaluationResult, evaluate
from crucible.training.pipeline import FinetuneResult, finetune


@dataclass
class PipelineResult:
    finetune: FinetuneResult
    evaluation: EvaluationResult = field(default_factory=lambda: EvaluationResult(output_file=""))


@workflow
def full_pipeline(config: CrucibleConfig) -> PipelineResult:
    """Load & clean data, fine-tune a model, then evaluate on the test set."""
    splits = data_processing_pipeline(config=config.data_processing)
    full_config_dict = asdict(config)
    ft_result = finetune(
        train_df=splits.train,
        val_df=splits.val,
        config=config.finetuning,
        tracking_config=config.tracking,
        full_config_dict=full_config_dict,
    )
    eval_result = evaluate(
        test_df=splits.test,
        model_path=ft_result.model_path,
        config=config.evaluation,
    )
    return PipelineResult(finetune=ft_result, evaluation=eval_result)


def run_full_pipeline(
    config: CrucibleConfig | None = None,
    config_path: str | Path | None = None,
) -> PipelineResult:
    """Convenience entry point — accepts a config or YAML path."""
    if config is not None and config_path is not None:
        raise ValueError("Provide config or config_path, not both")
    if config_path is not None:
        config = CrucibleConfig.from_yaml(config_path)
    if config is None:
        config = CrucibleConfig()
    return full_pipeline(config=config)  # type: ignore[no-any-return]
