"""Integration tests that hit the HuggingFace Hub.

Run with:  pytest -m integration
"""

import pandas as pd
import pytest

from crucible.data.cleaning import clean_data
from crucible.data.config import (
    CleaningConfig,
    DataProcessingConfig,
    DatasetConfig,
    FormattingConfig,
    SplitConfig,
)
from crucible.data.formatting import format_prompts
from crucible.data.loading import load_hf_dataset
from crucible.data.pipeline import data_processing_pipeline, run_data_processing
from crucible.data.splitting import split_data

pytestmark = pytest.mark.integration


class TestLoadHFDataset:
    def test_loads_financial_phrasebank(self) -> None:
        config = DatasetConfig(
            name="takala/financial_phrasebank",
            split="train",
        )
        df = load_hf_dataset(config=config)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "sentence" in df.columns
        assert "label" in df.columns


class TestEndToEndPipeline:
    def test_full_pipeline_with_config(self) -> None:
        config = DataProcessingConfig(
            split=SplitConfig(
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42,
                stratify_column="label",
            ),
        )
        result = data_processing_pipeline(config=config)

        assert len(result.train) > 0
        assert len(result.val) > 0
        assert len(result.test) > 0

        for split_df in (result.train, result.val, result.test):
            assert "formatted_text" in split_df.columns
            assert split_df["formatted_text"].isna().sum() == 0

    def test_run_with_yaml_path(self, tmp_path: pytest.TempPathFactory) -> None:
        yaml_content = """\
dataset:
  name: takala/financial_phrasebank
  subset: sentences_allagree
  split: train
split:
  stratify_column: label
"""
        yaml_file = tmp_path / "test_cfg.yaml"
        yaml_file.write_text(yaml_content)

        result = run_data_processing(config_path=str(yaml_file))
        assert len(result.train) > 0
        assert "formatted_text" in result.train.columns

    def test_pipeline_steps_individually(self) -> None:
        """Run each step in sequence so failures are easier to diagnose."""
        ds_config = DatasetConfig()
        raw = load_hf_dataset(config=ds_config)
        assert len(raw) > 0

        cleaned = clean_data(df=raw, config=CleaningConfig())
        assert cleaned["sentence"].isna().sum() == 0

        splits = split_data(
            df=cleaned,
            config=SplitConfig(seed=42, stratify_column="label"),
        )
        total = len(splits.train) + len(splits.val) + len(splits.test)
        assert total == len(cleaned)

        formatted = format_prompts(df=splits.train, config=FormattingConfig())
        assert "formatted_text" in formatted.columns
        sample = formatted["formatted_text"].iloc[0]
        assert "### Instruction:" in sample
        assert "### Input:" in sample
        assert "### Response:" in sample
