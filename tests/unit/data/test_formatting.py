import pandas as pd
import pytest

from crucible.data.config import FormattingConfig
from crucible.data.formatting import format_prompts


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence": ["Revenue rose 10%.", "Net loss of $5M."],
            "label": [2, 0],
        }
    )


class TestFormatPrompts:
    def test_basic_template_substitution(self, simple_df: pd.DataFrame) -> None:
        config = FormattingConfig(
            template="Text: {sentence} | Label: {label}",
            output_column="prompt",
        )
        result = format_prompts(df=simple_df, config=config)
        assert result["prompt"].iloc[0] == "Text: Revenue rose 10%. | Label: 2"
        assert result["prompt"].iloc[1] == "Text: Net loss of $5M. | Label: 0"

    def test_column_mapping_aliases(self, simple_df: pd.DataFrame) -> None:
        config = FormattingConfig(
            template="Input: {text} -> {sentiment}",
            column_mapping={"text": "sentence", "sentiment": "label"},
            output_column="prompt",
        )
        result = format_prompts(df=simple_df, config=config)
        assert result["prompt"].iloc[0] == "Input: Revenue rose 10%. -> 2"

    def test_original_columns_still_available_with_mapping(self, simple_df: pd.DataFrame) -> None:
        config = FormattingConfig(
            template="{sentence} (alias={text})",
            column_mapping={"text": "sentence"},
            output_column="prompt",
        )
        result = format_prompts(df=simple_df, config=config)
        assert result["prompt"].iloc[0] == "Revenue rose 10%. (alias=Revenue rose 10%.)"

    def test_missing_template_var_raises(self, simple_df: pd.DataFrame) -> None:
        config = FormattingConfig(
            template="{nonexistent_column}",
            output_column="prompt",
        )
        with pytest.raises(KeyError, match="nonexistent_column"):
            format_prompts(df=simple_df, config=config)

    def test_does_not_mutate_input(self, simple_df: pd.DataFrame) -> None:
        original = simple_df.copy()
        config = FormattingConfig(template="{sentence}", output_column="prompt")
        format_prompts(df=simple_df, config=config)
        pd.testing.assert_frame_equal(simple_df, original)

    def test_output_column_name_is_configurable(self, simple_df: pd.DataFrame) -> None:
        config = FormattingConfig(template="{sentence}", output_column="my_col")
        result = format_prompts(df=simple_df, config=config)
        assert "my_col" in result.columns
