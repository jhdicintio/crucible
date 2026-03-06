"""Unit tests for dataset loading and downsampling."""

import pandas as pd
import pytest

from crucible.data.config import DatasetConfig
from crucible.data.loading import _downsample


class TestDownsample:
    def test_no_sample_size_returns_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        config = DatasetConfig(sample_size=None)
        result = _downsample(df, config)
        pd.testing.assert_frame_equal(result, df)

    def test_sample_size_larger_than_df_returns_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        config = DatasetConfig(sample_size=10)
        result = _downsample(df, config)
        pd.testing.assert_frame_equal(result, df)

    def test_random_strategy_returns_n_rows_and_reproducible(self) -> None:
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        config = DatasetConfig(sample_size=20, sample_strategy="random", sample_seed=42)
        result = _downsample(df, config)
        assert len(result) == 20
        result2 = _downsample(df, config)
        pd.testing.assert_frame_equal(result, result2)

    def test_first_strategy_returns_first_n_rows(self) -> None:
        df = pd.DataFrame({"a": [10, 20, 30, 40, 50], "b": [1, 2, 3, 4, 5]})
        config = DatasetConfig(sample_size=3, sample_strategy="first")
        result = _downsample(df, config)
        assert len(result) == 3
        assert result["a"].tolist() == [10, 20, 30]

    def test_invalid_strategy_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        config = DatasetConfig(sample_size=2, sample_strategy="invalid")
        with pytest.raises(ValueError, match="Unknown sample_strategy"):
            _downsample(df, config)

    def test_stratified_preserves_distribution(self) -> None:
        df = pd.DataFrame({"x": range(100), "label": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})
        config = DatasetConfig(
            sample_size=20,
            sample_strategy="stratified",
            sample_stratify_column="label",
            sample_seed=42,
        )
        result = _downsample(df, config)
        assert len(result) == 20
        # Proportional: expect roughly 10 A, 6 B, 4 C
        counts = result["label"].value_counts()
        assert counts["A"] >= 8 and counts["A"] <= 12
        assert counts["B"] >= 4 and counts["B"] <= 8
        assert counts["C"] >= 2 and counts["C"] <= 6

    def test_stratified_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        config = DatasetConfig(
            sample_size=2,
            sample_strategy="stratified",
            sample_stratify_column="nonexistent",
        )
        with pytest.raises(ValueError, match="sample_stratify_column"):
            _downsample(df, config)

    def test_diversity_returns_n_rows(self) -> None:
        df = pd.DataFrame(
            {
                "instruction": ["summarize", "classify", "translate"] * 10,
                "input": ["text a", "text b", "text c"] * 10,
            }
        )
        config = DatasetConfig(
            sample_size=5,
            sample_strategy="diversity",
            sample_text_columns=["instruction", "input"],
            sample_embedding_model="all-MiniLM-L6-v2",
            sample_seed=42,
        )
        result = _downsample(df, config)
        assert len(result) == 5

    def test_diversity_defaults_to_instruction_input_when_possible(self) -> None:
        df = pd.DataFrame(
            {
                "instruction": ["instr"] * 5,
                "input": ["inp"] * 5,
            }
        )
        config = DatasetConfig(
            sample_size=2,
            sample_strategy="diversity",
            sample_seed=42,
        )
        result = _downsample(df, config)
        assert len(result) == 2

    def test_quality_drops_short_and_long_then_dedups(self) -> None:
        # One too short, one too long; rest valid. Quality drops bad length, then dedups.
        short = "a" * 5
        long = "b" * 5000
        ok1 = "First distinct instruction with enough length for the filter."
        ok2 = "Second different instruction to pass length and stay after dedup."
        df = pd.DataFrame(
            {
                "instruction": [short, long, ok1, ok2],
                "input": ["x", "y", "z", "w"],
            }
        )
        config = DatasetConfig(
            sample_size=2,
            sample_strategy="quality",
            sample_text_columns=["instruction", "input"],
            sample_quality_min_chars=20,
            sample_quality_max_chars=4000,
            sample_quality_dedup_threshold=0.95,
            sample_seed=42,
        )
        result = _downsample(df, config)
        assert len(result) == 2
        combined = (result["instruction"] + " " + result["input"]).str.len()
        assert (combined >= 20).all() and (combined <= 4000).all()

    def test_quality_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": ["only column"] * 5})
        config = DatasetConfig(
            sample_size=2,
            sample_strategy="quality",
            sample_text_columns=["instruction"],
            sample_seed=42,
        )
        with pytest.raises(ValueError, match="sample_text_columns"):
            _downsample(df, config)
