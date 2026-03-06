"""Integration test — evaluate a tiny model on CPU.

Run with:  pytest -m integration -k evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset

from crucible.evaluation.config import EvaluationConfig
from crucible.evaluation.evaluator import Evaluator
from crucible.training.config import FinetuningConfig, ModelConfig, TrainingConfig
from crucible.training.model import CausalLMModel

pytestmark = pytest.mark.integration

TINY_TEXTS = [
    "Revenue increased by 10%.",
    "The company posted a net loss.",
    "Operating margins held steady.",
    "Earnings exceeded expectations.",
] * 5


def _make_train_df() -> pd.DataFrame:
    return pd.DataFrame({"formatted_text": TINY_TEXTS})


def _make_test_df() -> pd.DataFrame:
    """Test set with columns the input_template and reference_column expect."""
    return pd.DataFrame(
        {
            "instruction": [
                "Summarise the financial statement",
                "Explain the result",
            ],
            "input": [
                "Revenue rose by 10%.",
                "Net loss reported.",
            ],
            "output": [
                "Revenue increased.",
                "The company lost money.",
            ],
        }
    )


def _make_qualitative_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt": [
                "What is compound interest?",
                "Explain stock dividends.",
            ]
        }
    )


class TestEvaluationIntegration:
    """Fine-tune a tiny model, save it, then run the full evaluation."""

    @pytest.fixture()
    def trained_model_path(self, tmp_path: Path) -> str:
        model_dir = str(tmp_path / "model")
        config = FinetuningConfig(
            model=ModelConfig(name="sshleifer/tiny-gpt2", approach="full"),
            training=TrainingConfig(
                output_dir=model_dir,
                num_epochs=1,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=1,
                learning_rate=5e-4,
                logging_steps=1,
                eval_steps=999,
                save_steps=999,
                eval_strategy="no",
                max_seq_length=32,
                seed=42,
                log_file=str(tmp_path / "log.jsonl"),
                text_column="formatted_text",
            ),
        )
        model = CausalLMModel(config)
        train_ds = Dataset.from_pandas(_make_train_df())
        model.train(train_ds)
        model.save(model_dir)
        return model_dir

    def test_evaluation_round_trip(self, trained_model_path: str, tmp_path: Path) -> None:
        output_file = str(tmp_path / "eval_results.json")
        eval_config = EvaluationConfig(
            model_path=trained_model_path,
            metrics=["rouge"],
            max_new_tokens=10,
            batch_size=2,
            input_template=(
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            reference_column="output",
            output_file=output_file,
        )

        evaluator = Evaluator(eval_config)

        # Quantitative
        quant = evaluator.evaluate_quantitative(_make_test_df())
        assert "rouge" in quant
        assert "rouge1" in quant["rouge"]
        assert isinstance(quant["rouge"]["rouge1"], float)

        # Qualitative
        qual = evaluator.evaluate_qualitative(_make_qualitative_df())
        assert len(qual) == 2
        assert "input" in qual[0]
        assert "output" in qual[0]

        # Full run writes file
        evaluator.run(_make_test_df(), qualitative_df=_make_qualitative_df())
        assert Path(output_file).exists()
        data = json.loads(Path(output_file).read_text())
        assert "quantitative" in data
        assert "qualitative" in data
