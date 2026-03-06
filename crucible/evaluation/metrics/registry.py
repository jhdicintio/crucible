"""Built-in metric registry for YAML-driven metric lookup."""

from __future__ import annotations

from typing import Any

from crucible.evaluation.config import EvaluationConfig
from crucible.evaluation.metrics.bertscore import BertScoreMetric
from crucible.evaluation.metrics.rouge import RougeMetric
from crucible.evaluation.metrics.similarity import SemanticSimilarityMetric
from crucible.evaluation.protocol import MetricProtocol

BUILTIN_METRICS: dict[str, type] = {
    "rouge": RougeMetric,
    "bertscore": BertScoreMetric,
    "semantic_similarity": SemanticSimilarityMetric,
}


def build_metrics(config: EvaluationConfig) -> list[MetricProtocol]:
    """Instantiate the metrics listed in *config.metrics*."""
    instances: list[Any] = []
    for name in config.metrics:
        cls = BUILTIN_METRICS.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown metric {name!r}. Available: {sorted(BUILTIN_METRICS.keys())}"
            )
        if cls is BertScoreMetric:
            instances.append(cls(model_type=config.bertscore_model))
        elif cls is SemanticSimilarityMetric:
            instances.append(cls(model_name=config.similarity_model))
        else:
            instances.append(cls())
    return instances
