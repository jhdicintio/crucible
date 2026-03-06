"""Flyte task for loading a HuggingFace dataset."""

from __future__ import annotations

import datasets
import numpy as np
import pandas as pd
from flytekit import task

from crucible.data.config import DatasetConfig


def _sample_stratified(df: pd.DataFrame, n: int, column: str, seed: int) -> pd.DataFrame:
    """Sample n rows preserving the distribution of `column` (proportional allocation)."""
    if column not in df.columns:
        raise ValueError(f"sample_stratify_column '{column}' not in DataFrame columns")
    groups = df.groupby(column, group_keys=False)
    total = len(df)
    # Proportional allocation: each group gets (group_size / total) * n, at least 1 if possible
    quotas = (groups.size() * n / total).round().astype(int).clip(1, None)
    # If sum(quotas) != n due to rounding, adjust the largest or smallest group
    diff = n - quotas.sum()
    if diff != 0:
        adjust_key = quotas.index[quotas.argmax()] if diff > 0 else quotas.index[quotas.argmin()]
        quotas = quotas.copy()
        quotas[adjust_key] = quotas[adjust_key] + diff
    sampled = []
    for key, group in groups:
        size = min(quotas[key], len(group))
        chosen = group.sample(n=size, random_state=seed + hash(str(key)) % (2**31))
        sampled.append(chosen)
    out = pd.concat(sampled, axis=0)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out.head(n) if len(out) > n else out


def _farthest_point_sampling(embeddings: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Return indices of n rows that are maximally diverse (farthest point sampling)."""
    rng = np.random.default_rng(seed)
    n_total = embeddings.shape[0]
    if n >= n_total:
        return np.arange(n_total)
    # Normalize for cosine-like distance (we use L2 for simplicity; embeddings often normalized)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    x_norm = embeddings / norms
    selected = [int(rng.integers(0, n_total))]
    for _ in range(n - 1):
        # (n_selected, dim) vs (n_total, dim) -> (n_total,) min distance to selected set
        chosen = x_norm[selected]
        sim = np.dot(x_norm, chosen.T)  # (n_total, n_selected)
        max_sim = np.max(sim, axis=1)
        # Farthest = smallest similarity (we want to avoid already-selected)
        max_sim[selected] = 1.0
        next_idx = np.argmin(max_sim)
        selected.append(int(next_idx))
    return np.array(selected)


def _build_text_column(df: pd.DataFrame, text_columns: list[str]) -> list[str]:
    """Concatenate text_columns into one string per row."""
    for col in text_columns:
        if col not in df.columns:
            raise ValueError(f"sample_text_columns '{col}' not in DataFrame columns")
    texts = df[text_columns[0]].astype(str)
    for col in text_columns[1:]:
        texts = texts + " " + df[col].astype(str)
    return [str(x) for x in texts]


def _sample_diversity(
    df: pd.DataFrame,
    n: int,
    text_columns: list[str],
    embedding_model: str,
    seed: int,
) -> pd.DataFrame:
    """Sample n rows that are semantically diverse (embed + farthest point sampling)."""
    from sentence_transformers import SentenceTransformer

    texts = _build_text_column(df, text_columns)
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts, convert_to_numpy=True)
    indices = _farthest_point_sampling(embeddings, n, seed)
    return df.iloc[indices].reset_index(drop=True)


def _sample_quality(
    df: pd.DataFrame,
    n: int,
    text_columns: list[str],
    embedding_model: str,
    min_chars: int,
    max_chars: int,
    dedup_threshold: float,
    seed: int,
) -> pd.DataFrame:
    """Remove poor examples (bad length, near-duplicates), then take n by diversity.

    Keeps examples with total text length in [min_chars, max_chars], drops
    near-duplicates (cosine sim > dedup_threshold), then selects n maximally
    diverse from the remainder.
    """
    from sentence_transformers import SentenceTransformer

    texts = _build_text_column(df, text_columns)
    lengths = np.array([len(t) for t in texts])
    valid_mask = (lengths >= min_chars) & (lengths <= max_chars)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return df.head(0)
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    texts_filtered = [texts[i] for i in valid_indices]

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts_filtered, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    x_norm = embeddings / norms

    # Dedup: keep first of each cluster (cosine sim > threshold)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_norm))
    kept: list[int] = []
    for idx in order:
        if not kept:
            kept.append(idx)
            continue
        sims = np.dot(x_norm[idx : idx + 1], x_norm[kept].T)[0]
        if np.max(sims) < dedup_threshold:
            kept.append(idx)
    kept = np.array(kept)
    df_dedup = df_filtered.iloc[kept].reset_index(drop=True)
    embeddings_dedup = x_norm[kept]

    n_take = min(n, len(df_dedup))
    if n_take >= len(df_dedup):
        return df_dedup
    fps_indices = _farthest_point_sampling(embeddings_dedup, n_take, seed)
    return df_dedup.iloc[fps_indices].reset_index(drop=True)


def _downsample(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Downsample to sample_size rows using the configured strategy."""
    n = config.sample_size
    if n is None or len(df) <= n:
        return df
    strategy = (config.sample_strategy or "random").strip().lower()
    seed = getattr(config, "sample_seed", 42) or 42

    if strategy == "random":
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    if strategy == "first":
        return df.head(n).reset_index(drop=True)
    if strategy == "stratified":
        stratify_col = getattr(config, "sample_stratify_column", None)
        if not stratify_col:
            raise ValueError("sample_strategy 'stratified' requires sample_stratify_column")
        return _sample_stratified(df, n, stratify_col, seed)
    if strategy == "diversity":
        text_cols = _text_columns_for_sampling(df, config)
        model_name = (
            getattr(config, "sample_embedding_model", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2"
        )
        return _sample_diversity(df, n, text_cols, model_name, seed)
    if strategy == "quality":
        text_cols = _text_columns_for_sampling(df, config)
        model_name = (
            getattr(config, "sample_embedding_model", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2"
        )
        min_chars = getattr(config, "sample_quality_min_chars", 20) or 20
        max_chars = getattr(config, "sample_quality_max_chars", 4000) or 4000
        dedup = getattr(config, "sample_quality_dedup_threshold", 0.95) or 0.95
        return _sample_quality(df, n, text_cols, model_name, min_chars, max_chars, dedup, seed)

    raise ValueError(
        f"Unknown sample_strategy: {config.sample_strategy}. "
        "Use 'random', 'first', 'stratified', 'diversity', or 'quality'."
    )


def _text_columns_for_sampling(df: pd.DataFrame, config: DatasetConfig) -> list[str]:
    """Resolve sample_text_columns from config or sensible defaults."""
    text_cols = getattr(config, "sample_text_columns", None) or []
    if text_cols:
        return text_cols
    if "instruction" in df.columns and "input" in df.columns:
        return ["instruction", "input"]
    candidates = [
        c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])
    ]
    if candidates:
        return [candidates[0]]
    raise ValueError(
        "sample_strategy 'diversity' or 'quality' requires sample_text_columns or string columns"
    )


@task
def load_hf_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load a dataset from HuggingFace Hub and return it as a DataFrame.

    If ``config.sample_size`` is set, the loaded data is downsampled using
    ``sample_strategy`` (``random`` or ``first``). ``sample_seed`` is used
    when strategy is ``random`` for reproducibility.
    """
    ds = datasets.load_dataset(
        config.name,
        split=config.split,
        trust_remote_code=True,
    )
    df = ds.to_pandas()
    return _downsample(df, config)
