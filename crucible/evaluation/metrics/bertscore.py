"""BERTScore metric implementation."""

from __future__ import annotations

from typing import Any

from bert_score import score as bert_score_fn


class BertScoreMetric:
    """Computes BERTScore precision, recall, and F1."""

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli") -> None:
        self._model_type = model_type

    @property
    def name(self) -> str:
        return "bertscore"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        if not predictions:
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
            }

        precision, recall, f1 = bert_score_fn(
            predictions,
            references,
            model_type=self._model_type,
            verbose=False,
        )

        return {
            "bertscore_precision": round(precision.mean().item(), 6),
            "bertscore_recall": round(recall.mean().item(), 6),
            "bertscore_f1": round(f1.mean().item(), 6),
        }
