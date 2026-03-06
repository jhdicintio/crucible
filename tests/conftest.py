"""Shared pytest fixtures and configuration."""

import pandas as pd
import pytest


@pytest.fixture
def sample_financial_df() -> pd.DataFrame:
    """Small DataFrame mimicking financial_phrasebank structure.

    Includes intentional quality issues: NaN, leading/trailing whitespace,
    and a duplicate row.
    """
    return pd.DataFrame(
        {
            "sentence": [
                "Revenue increased by 15% year-over-year.",
                "The company reported a net loss of $2M.",
                "Operating margins remained stable at 12%.",
                "  Stock price fell sharply after earnings.  ",
                "Revenue increased by 15% year-over-year.",  # duplicate
                None,  # NaN
            ],
            "label": [2, 0, 1, 0, 2, 1],
        }
    )
