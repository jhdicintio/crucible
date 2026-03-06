"""ROUGE metric implementation."""

from __future__ import annotations

from rouge_score import rouge_scorer


class RougeMetric:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""

    def __init__(self) -> None:
        self._scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    @property
    def name(self) -> str:
        return "rouge"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        totals: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        n = len(predictions)
        if n == 0:
            return totals

        for pred, ref in zip(predictions, references, strict=True):
            scores = self._scorer.score(ref, pred)
            for key in totals:
                totals[key] += scores[key].fmeasure

        return {k: round(v / n, 6) for k, v in totals.items()}
