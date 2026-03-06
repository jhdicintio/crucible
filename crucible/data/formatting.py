"""Flyte task for applying prompt templates to examples."""

import pandas as pd
from flytekit import task

from crucible.data.config import FormattingConfig


def _format_row(
    row: pd.Series,
    template: str,
    column_mapping: dict[str, str],
) -> str:
    """Build the template value dict and format a single row.

    All DataFrame columns are available by name in the template.  Entries in
    ``column_mapping`` add *aliases*: ``{template_var: column_name}`` so a
    template can reference ``{text}`` even if the column is called ``sentence``.
    """
    values: dict[str, object] = row.to_dict()
    for template_var, column_name in column_mapping.items():
        values[template_var] = row[column_name]
    try:
        return template.format_map(values)
    except KeyError as exc:
        available = sorted(values.keys())
        raise KeyError(f"Template variable {exc} not found. Available keys: {available}") from exc


@task
def format_prompts(df: pd.DataFrame, config: FormattingConfig) -> pd.DataFrame:
    """Apply a prompt template to every row, storing the result in a new column."""
    df = df.copy()
    df[config.output_column] = df.apply(
        _format_row,
        axis=1,
        template=config.template,
        column_mapping=config.column_mapping,
    )
    return df
