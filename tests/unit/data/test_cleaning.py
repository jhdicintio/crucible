import pandas as pd

from crucible.data.cleaning import clean_data
from crucible.data.config import CleaningConfig


class TestCleanData:
    def test_drops_nan_rows(self, sample_financial_df: pd.DataFrame) -> None:
        config = CleaningConfig(drop_nans=True, drop_duplicates=False)
        result = clean_data(df=sample_financial_df, config=config)
        assert result["sentence"].isna().sum() == 0

    def test_keeps_nan_rows_when_disabled(self, sample_financial_df: pd.DataFrame) -> None:
        config = CleaningConfig(drop_nans=False, drop_duplicates=False, strip_whitespace=False)
        result = clean_data(df=sample_financial_df, config=config)
        assert result["sentence"].isna().sum() == 1

    def test_drops_duplicate_rows(self, sample_financial_df: pd.DataFrame) -> None:
        config = CleaningConfig(drop_duplicates=True)
        result = clean_data(df=sample_financial_df, config=config)
        assert len(result) == len(result.drop_duplicates(subset=["sentence", "label"]))

    def test_strips_whitespace(self, sample_financial_df: pd.DataFrame) -> None:
        config = CleaningConfig()
        result = clean_data(df=sample_financial_df, config=config)
        for val in result["sentence"]:
            assert val == val.strip()

    def test_lowercases_strings(self, sample_financial_df: pd.DataFrame) -> None:
        config = CleaningConfig(lowercase=True)
        result = clean_data(df=sample_financial_df, config=config)
        for val in result["sentence"]:
            assert val == val.lower()

    def test_respects_target_columns(self) -> None:
        df = pd.DataFrame({"a": ["  x  ", None], "b": ["  y  ", "z"]})
        config = CleaningConfig(columns=["b"])
        result = clean_data(df=df, config=config)
        assert result["b"].tolist() == ["y", "z"]
        assert result["a"].tolist() == ["  x  ", None]

    def test_does_not_mutate_input(self, sample_financial_df: pd.DataFrame) -> None:
        original = sample_financial_df.copy()
        clean_data(df=sample_financial_df, config=CleaningConfig())
        pd.testing.assert_frame_equal(sample_financial_df, original)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            {"sentence": pd.Series([], dtype="object"), "label": pd.Series([], dtype="int64")}
        )
        result = clean_data(df=df, config=CleaningConfig())
        assert len(result) == 0

    def test_resets_index(self, sample_financial_df: pd.DataFrame) -> None:
        config = CleaningConfig()
        result = clean_data(df=sample_financial_df, config=config)
        assert list(result.index) == list(range(len(result)))
