"""Flyte tasks and workflow for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from flytekit import task, workflow

from crucible.evaluation.config import EvaluationConfig
from crucible.evaluation.evaluator import Evaluator


@dataclass
class EvaluationResult:
    output_file: str
    quantitative: dict[str, dict[str, float]] = field(default_factory=dict)


@task
def evaluate(
    test_df: pd.DataFrame,
    model_path: str,
    config: EvaluationConfig,
    qualitative_df: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Load a trained model, compute metrics on the test set, and optionally
    generate qualitative outputs."""
    resolved = EvaluationConfig(
        model_path=model_path,
        metrics=config.metrics,
        max_new_tokens=config.max_new_tokens,
        input_template=config.input_template,
        reference_column=config.reference_column,
        column_mapping=config.column_mapping,
        bertscore_model=config.bertscore_model,
        similarity_model=config.similarity_model,
        output_file=config.output_file,
        batch_size=config.batch_size,
    )

    evaluator = Evaluator(resolved)
    results: dict[str, Any] = evaluator.run(test_df, qualitative_df=qualitative_df)

    return EvaluationResult(
        output_file=resolved.output_file,
        quantitative=results.get("quantitative", {}),
    )


@workflow
def evaluation_pipeline(
    test_df: pd.DataFrame,
    model_path: str,
    config: EvaluationConfig,
    qualitative_df: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Thin workflow wrapper for Flyte orchestration."""
    return evaluate(  # type: ignore[no-any-return]
        test_df=test_df,
        model_path=model_path,
        config=config,
        qualitative_df=qualitative_df,
    )


def run_evaluation(
    test_df: pd.DataFrame,
    model_path: str,
    config: EvaluationConfig | None = None,
    config_path: str | Path | None = None,
    qualitative_df: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Convenience entry point — accepts a config object *or* a YAML path."""
    if config is not None and config_path is not None:
        raise ValueError("Provide config or config_path, not both")
    if config_path is not None:
        config = EvaluationConfig.from_yaml(config_path)
    if config is None:
        config = EvaluationConfig()
    return evaluation_pipeline(  # type: ignore[no-any-return]
        test_df=test_df,
        model_path=model_path,
        config=config,
        qualitative_df=qualitative_df,
    )
