"""Unit tests for built-in metric implementations."""

from crucible.evaluation.metrics.rouge import RougeMetric


class TestRougeMetric:
    def test_identical_texts_score_high(self) -> None:
        m = RougeMetric()
        preds = ["The revenue increased by ten percent."]
        refs = ["The revenue increased by ten percent."]
        scores = m.compute(preds, refs)
        assert scores["rouge1"] > 0.9
        assert scores["rouge2"] > 0.9
        assert scores["rougeL"] > 0.9

    def test_completely_different_texts_score_low(self) -> None:
        m = RougeMetric()
        preds = ["alpha beta gamma delta"]
        refs = ["one two three four"]
        scores = m.compute(preds, refs)
        assert scores["rouge1"] < 0.1

    def test_empty_inputs(self) -> None:
        m = RougeMetric()
        scores = m.compute([], [])
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_multiple_examples_averaged(self) -> None:
        m = RougeMetric()
        preds = ["The cat sat on the mat.", "Dogs are great pets."]
        refs = ["The cat sat on the mat.", "Cats are great pets."]
        scores = m.compute(preds, refs)
        # First pair is perfect, second is partial — average should be moderate-high
        assert 0.5 < scores["rouge1"] <= 1.0

    def test_name_property(self) -> None:
        assert RougeMetric().name == "rouge"
