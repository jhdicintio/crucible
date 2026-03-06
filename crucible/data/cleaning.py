"""Flyte task for data quality cleaning."""

import pandas as pd
from flytekit import task

from crucible.data.config import CleaningConfig


@task
def clean_data(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    """Handle NaNs, whitespace, casing, and duplicates."""
    df = df.copy()
    target_cols = config.columns if config.columns else df.columns.tolist()

    if config.drop_nans:
        df = df.dropna(subset=target_cols)

    str_cols = [c for c in target_cols if df[c].dtype == "object"]

    if config.strip_whitespace:
        for col in str_cols:
            df[col] = df[col].str.strip()

    if config.lowercase:
        for col in str_cols:
            df[col] = df[col].str.lower()

    if config.drop_duplicates:
        df = df.drop_duplicates(subset=target_cols)

    return df.reset_index(drop=True)
