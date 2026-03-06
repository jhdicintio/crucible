"""Flyte task for loading a HuggingFace dataset."""

import datasets
import pandas as pd
from flytekit import task

from crucible.data.config import DatasetConfig


@task
def load_hf_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load a dataset from HuggingFace Hub and return it as a DataFrame."""
    ds = datasets.load_dataset(
        config.name,
        split=config.split,
        trust_remote_code=True,
    )
    return ds.to_pandas()
