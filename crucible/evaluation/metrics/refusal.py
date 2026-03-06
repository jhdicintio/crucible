"""Refusal metric: precision/recall on refusing unanswerable or out-of-scope questions."""

from __future__ import annotations

from typing import Any

from crucible.guardrails.constants import REFUSAL_MESSAGE


def _is_refusal(pred: str) -> bool:
    return pred.strip() == REFUSAL_MESSAGE.strip()


class RefusalMetric:
    """
    Evaluates refusal behavior on unanswerable/out-of-scope questions.
    Expects metadata with an 'answerable' boolean per example.
    """

    @property
    def name(self) -> str:
        return "refusal"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        if not metadata or len(metadata) != len(predictions):
            return {
                "refusal_precision": 0.0,
                "refusal_recall": 0.0,
                "false_refusal_rate": 0.0,
            }

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred, meta in zip(predictions, metadata, strict=True):
            refused = _is_refusal(pred)
            answerable = meta.get("answerable", True)

            if not answerable and refused:
                true_positives += 1
            elif answerable and refused:
                false_positives += 1
            elif not answerable and not refused:
                false_negatives += 1

        total_unanswerable = sum(1 for m in metadata if not m.get("answerable", True))
        n = len(predictions)

        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (total_unanswerable + 1e-9)
        false_refusal_rate = false_positives / n if n else 0.0

        return {
            "refusal_precision": round(precision, 6),
            "refusal_recall": round(recall, 6),
            "false_refusal_rate": round(false_refusal_rate, 6),
        }
