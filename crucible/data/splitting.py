"""Flyte task for train / val / test splitting."""

from typing import NamedTuple

import pandas as pd
from flytekit import task
from sklearn.model_selection import train_test_split

from crucible.data.config import SplitConfig


class SplitResult(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@task
def split_data(df: pd.DataFrame, config: SplitConfig) -> SplitResult:
    """Split a DataFrame into train / val / test sets with optional stratification."""
    total = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    stratify = df[config.stratify_column] if config.stratify_column else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=config.test_ratio,
        random_state=config.seed,
        stratify=stratify,
    )

    val_relative = config.val_ratio / (config.train_ratio + config.val_ratio)
    stratify_tv = train_val_df[config.stratify_column] if config.stratify_column else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=config.seed,
        stratify=stratify_tv,
    )

    return SplitResult(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
