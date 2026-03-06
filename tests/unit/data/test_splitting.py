import pandas as pd
import pytest

from crucible.data.config import SplitConfig
from crucible.data.splitting import split_data


def _make_df(n: int = 100) -> pd.DataFrame:
    labels = [0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3))
    return pd.DataFrame(
        {
            "sentence": [f"sentence_{i}" for i in range(n)],
            "label": labels,
        }
    )


class TestSplitData:
    def test_split_sizes_match_ratios(self) -> None:
        df = _make_df(100)
        config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        result = split_data(df=df, config=config)

        assert len(result.train) == 80
        assert len(result.val) == 10
        assert len(result.test) == 10

    def test_no_row_leakage_between_splits(self) -> None:
        df = _make_df(200)
        config = SplitConfig(seed=7)
        result = split_data(df=df, config=config)

        train_idx = set(result.train["sentence"])
        val_idx = set(result.val["sentence"])
        test_idx = set(result.test["sentence"])

        assert train_idx & val_idx == set()
        assert train_idx & test_idx == set()
        assert val_idx & test_idx == set()
        assert len(train_idx | val_idx | test_idx) == 200

    def test_seed_reproducibility(self) -> None:
        df = _make_df(100)
        config = SplitConfig(seed=99)
        r1 = split_data(df=df, config=config)
        r2 = split_data(df=df, config=config)

        pd.testing.assert_frame_equal(r1.train, r2.train)
        pd.testing.assert_frame_equal(r1.val, r2.val)
        pd.testing.assert_frame_equal(r1.test, r2.test)

    def test_stratification_preserves_label_distribution(self) -> None:
        df = _make_df(300)
        config = SplitConfig(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            stratify_column="label",
        )
        result = split_data(df=df, config=config)

        orig_dist = df["label"].value_counts(normalize=True).sort_index()
        train_dist = result.train["label"].value_counts(normalize=True).sort_index()
        for label in orig_dist.index:
            assert abs(orig_dist[label] - train_dist[label]) < 0.05

    def test_bad_ratios_raises(self) -> None:
        df = _make_df(50)
        config = SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_data(df=df, config=config)

    def test_resets_index(self) -> None:
        df = _make_df(100)
        result = split_data(df=df, config=SplitConfig(seed=42))
        for split_df in (result.train, result.val, result.test):
            assert list(split_df.index) == list(range(len(split_df)))
