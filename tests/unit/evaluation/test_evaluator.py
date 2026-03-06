"""Unit tests for the Evaluator class with mocked model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from crucible.evaluation.config import EvaluationConfig
from crucible.evaluation.evaluator import Evaluator


def _make_test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instruction": ["Summarise the text", "Explain the concept"],
            "input": ["Revenue rose 10%.", "Compound interest"],
            "output": ["Revenue increased.", "Interest on interest."],
        }
    )


def _make_qualitative_df() -> pd.DataFrame:
    return pd.DataFrame({"prompt": ["What is a bond?", "Define equity."]})


def _mock_model() -> MagicMock:
    """A mock that echoes the prompt plus a fixed suffix."""
    model = MagicMock()

    def fake_predict(inputs: list[str], max_new_tokens: int = 256) -> list[str]:
        return [inp + "Generated answer." for inp in inputs]

    model.predict = MagicMock(side_effect=fake_predict)
    return model


class TestEvaluator:
    @patch("crucible.evaluation.evaluator.CausalLMModel.load")
    @patch("crucible.evaluation.evaluator.build_metrics")
    def test_evaluate_quantitative(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _mock_model()

        fake_metric = MagicMock()
        fake_metric.name = "test_metric"
        fake_metric.compute.return_value = {"score": 0.75}
        mock_build.return_value = [fake_metric]

        config = EvaluationConfig(
            model_path="fake/path",
            metrics=["rouge"],
            input_template=(
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            reference_column="output",
        )
        evaluator = Evaluator(config)
        results = evaluator.evaluate_quantitative(_make_test_df())

        assert "test_metric" in results
        assert results["test_metric"]["score"] == 0.75
        fake_metric.compute.assert_called_once()
        call_args = fake_metric.compute.call_args
        preds, refs = call_args[0][0], call_args[0][1]
        assert len(preds) == 2
        assert len(refs) == 2

    @patch("crucible.evaluation.evaluator.CausalLMModel.load")
    @patch("crucible.evaluation.evaluator.build_metrics")
    def test_evaluate_qualitative(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _mock_model()
        mock_build.return_value = []

        config = EvaluationConfig(model_path="fake/path", metrics=[])
        evaluator = Evaluator(config)
        outputs = evaluator.evaluate_qualitative(_make_qualitative_df())

        assert len(outputs) == 2
        assert outputs[0]["input"] == "What is a bond?"
        assert "Generated answer." in outputs[0]["output"]

    @patch("crucible.evaluation.evaluator.CausalLMModel.load")
    @patch("crucible.evaluation.evaluator.build_metrics")
    def test_run_writes_output_file(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _mock_model()

        fake_metric = MagicMock()
        fake_metric.name = "m"
        fake_metric.compute.return_value = {"v": 1.0}
        mock_build.return_value = [fake_metric]

        out = tmp_path / "results.json"
        config = EvaluationConfig(
            model_path="fake/path",
            metrics=["rouge"],
            output_file=str(out),
        )
        evaluator = Evaluator(config)
        result: dict[str, Any] = evaluator.run(
            _make_test_df(), qualitative_df=_make_qualitative_df()
        )

        assert out.exists()
        assert "quantitative" in result
        assert "qualitative" in result

    @patch("crucible.evaluation.evaluator.CausalLMModel.load")
    @patch("crucible.evaluation.evaluator.build_metrics")
    def test_strip_prompt(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _mock_model()
        mock_build.return_value = []

        config = EvaluationConfig(model_path="fake/path", metrics=[])
        evaluator = Evaluator(config)

        assert evaluator._strip_prompt("Hello World", "Hello ") == "World"
        assert evaluator._strip_prompt("No match here", "XYZ") == "No match here"
