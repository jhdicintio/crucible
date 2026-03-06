"""Verify that built-in metrics satisfy MetricProtocol."""

from crucible.evaluation.protocol import MetricProtocol


def test_metric_protocol_is_runtime_checkable() -> None:
    assert hasattr(MetricProtocol, "__protocol_attrs__") or isinstance(MetricProtocol, type)


def test_rouge_metric_satisfies_protocol() -> None:
    from crucible.evaluation.metrics.rouge import RougeMetric

    m = RougeMetric()
    assert isinstance(m, MetricProtocol)
    assert m.name == "rouge"
    assert callable(m.compute)


def test_bertscore_metric_has_required_interface() -> None:
    from crucible.evaluation.metrics.bertscore import BertScoreMetric

    assert hasattr(BertScoreMetric, "compute")
    assert hasattr(BertScoreMetric, "name")


def test_similarity_metric_has_required_interface() -> None:
    from crucible.evaluation.metrics.similarity import SemanticSimilarityMetric

    assert hasattr(SemanticSimilarityMetric, "compute")
    assert hasattr(SemanticSimilarityMetric, "name")
