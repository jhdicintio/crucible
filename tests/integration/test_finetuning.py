"""Integration test — fine-tune a tiny model on CPU.

Run with:  pytest -m integration -k finetuning
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from crucible.training.config import FinetuningConfig, ModelConfig, TrainingConfig
from crucible.training.model import CausalLMModel

pytestmark = pytest.mark.integration

TINY_TEXTS = [
    "Revenue increased by 10%.",
    "The company posted a net loss.",
    "Operating margins held steady.",
    "Earnings exceeded expectations.",
] * 5  # 20 examples — enough for one training pass


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({"formatted_text": TINY_TEXTS})


class TestCausalLMModelTraining:
    """Full round-trip: init → train → save → load → predict."""

    @pytest.fixture()
    def config(self, tmp_path: pytest.TempPathFactory) -> FinetuningConfig:
        return FinetuningConfig(
            model=ModelConfig(name="sshleifer/tiny-gpt2", approach="full"),
            training=TrainingConfig(
                output_dir=str(tmp_path / "model"),
                num_epochs=1,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=1,
                learning_rate=5e-4,
                logging_steps=1,
                eval_steps=5,
                save_steps=999,
                eval_strategy="steps",
                max_seq_length=32,
                seed=42,
                log_file=str(tmp_path / "log.jsonl"),
                text_column="formatted_text",
            ),
        )

    def test_full_finetune_round_trip(self, config: FinetuningConfig) -> None:
        train_df = _make_df()
        val_df = _make_df()

        model = CausalLMModel(config)
        metrics = model.train(
            train_dataset=__import__("datasets").Dataset.from_pandas(train_df),
            val_dataset=__import__("datasets").Dataset.from_pandas(val_df),
        )

        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0

        log_path = config.training.log_file
        lines = Path(log_path).read_text().strip().split("\n")
        assert len(lines) > 0
        first = json.loads(lines[0])
        assert "loss" in first or "train_loss" in first

        model.save(config.training.output_dir)

        loaded = CausalLMModel.load(config.training.output_dir)
        preds = loaded.predict(["Revenue rose"], max_new_tokens=5)
        assert len(preds) == 1
        assert isinstance(preds[0], str)
