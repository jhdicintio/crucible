"""Evaluator — orchestrates quantitative metrics and qualitative generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from crucible.evaluation.config import EvaluationConfig
from crucible.evaluation.metrics.registry import build_metrics
from crucible.evaluation.protocol import MetricProtocol
from crucible.training.model import CausalLMModel

logger = logging.getLogger(__name__)


class Evaluator:
    """Run quantitative metrics on a test set and qualitative generation on user prompts.

    Parameters
    ----------
    config:
        Evaluation configuration (metrics, templates, model path, etc.).
    extra_metrics:
        Additional ``MetricProtocol``-conforming instances to run alongside
        the built-in metrics listed in *config.metrics*.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        extra_metrics: list[MetricProtocol] | None = None,
    ) -> None:
        self.config = config
        self.model = CausalLMModel.load(config.model_path)
        self._metrics: list[MetricProtocol] = build_metrics(config)
        if extra_metrics:
            self._metrics.extend(extra_metrics)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_quantitative(self, test_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Generate predictions for the test set and compute all metrics."""
        prompts = self._build_prompts(test_df)
        references = test_df[self.config.reference_column].tolist()

        logger.info("Generating predictions for %d test examples …", len(prompts))
        raw_outputs = self._predict_batched(prompts)
        predictions = [
            self._strip_prompt(output, prompt)
            for output, prompt in zip(raw_outputs, prompts, strict=True)
        ]

        results: dict[str, dict[str, float]] = {}
        for metric in self._metrics:
            logger.info("Computing %s …", metric.name)
            results[metric.name] = metric.compute(predictions, references)
        return results

    def evaluate_qualitative(
        self, prompts_df: pd.DataFrame, prompt_column: str = "prompt"
    ) -> list[dict[str, str]]:
        """Generate model outputs for user-provided prompts.

        Returns a list of ``{"input": ..., "output": ...}`` dicts.
        """
        prompts = prompts_df[prompt_column].tolist()
        raw_outputs = self._predict_batched(prompts)
        predictions = [
            self._strip_prompt(output, prompt)
            for output, prompt in zip(raw_outputs, prompts, strict=True)
        ]
        return [
            {"input": prompt, "output": pred}
            for prompt, pred in zip(prompts, predictions, strict=True)
        ]

    def run(
        self,
        test_df: pd.DataFrame,
        qualitative_df: pd.DataFrame | None = None,
        prompt_column: str = "prompt",
    ) -> dict[str, Any]:
        """Run full evaluation and persist results to *config.output_file*."""
        results: dict[str, Any] = {
            "quantitative": self.evaluate_quantitative(test_df),
        }
        if qualitative_df is not None:
            results["qualitative"] = self.evaluate_qualitative(
                qualitative_df, prompt_column=prompt_column
            )

        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        logger.info("Evaluation results written to %s", output_path)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompts(self, df: pd.DataFrame) -> list[str]:
        """Format each row of *df* using the configured input template."""
        template = self.config.input_template
        mapping = self.config.column_mapping
        prompts: list[str] = []
        for _, row in df.iterrows():
            row_dict = dict(row)
            for alias, col in mapping.items():
                row_dict[alias] = row_dict[col]
            prompts.append(template.format(**row_dict))
        return prompts

    def _predict_batched(self, prompts: list[str]) -> list[str]:
        """Run model.predict in batches to manage memory."""
        bs = self.config.batch_size
        all_outputs: list[str] = []
        for i in range(0, len(prompts), bs):
            batch = prompts[i : i + bs]
            all_outputs.extend(self.model.predict(batch, max_new_tokens=self.config.max_new_tokens))
        return all_outputs

    @staticmethod
    def _strip_prompt(output: str, prompt: str) -> str:
        """Remove the echoed prompt prefix from the generated output."""
        if output.startswith(prompt):
            return output[len(prompt) :].strip()
        return output.strip()
