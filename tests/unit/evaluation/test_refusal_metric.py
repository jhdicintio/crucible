"""Unit tests for RefusalMetric."""

from crucible.evaluation.metrics.refusal import RefusalMetric
from crucible.guardrails.constants import REFUSAL_MESSAGE


class TestRefusalMetric:
    def test_correctly_refused_unanswerable(self) -> None:
        m = RefusalMetric()
        predictions = [REFUSAL_MESSAGE, "Some answer", REFUSAL_MESSAGE]
        references = ["", "ref", ""]
        metadata = [
            {"answerable": False},
            {"answerable": True},
            {"answerable": False},
        ]
        out = m.compute(predictions, references, metadata=metadata)
        assert out["refusal_precision"] == 1.0
        assert out["refusal_recall"] == 1.0
        assert out["false_refusal_rate"] == 0.0

    def test_false_positive_refusal(self) -> None:
        m = RefusalMetric()
        predictions = [REFUSAL_MESSAGE, "Answer"]
        references = ["ref1", "ref2"]
        metadata = [{"answerable": True}, {"answerable": True}]
        out = m.compute(predictions, references, metadata=metadata)
        assert out["false_refusal_rate"] == 0.5

    def test_no_metadata_returns_zeros(self) -> None:
        m = RefusalMetric()
        out = m.compute(["a", "b"], ["x", "y"], metadata=None)
        assert out["refusal_precision"] == 0.0
        assert out["refusal_recall"] == 0.0
        assert out["false_refusal_rate"] == 0.0
