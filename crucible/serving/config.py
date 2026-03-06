"""Structured config for the model serving API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class ServingConfig:
    model_path: str = "outputs/finetuned"
    host: str = "0.0.0.0"
    port: int = 5000
    max_new_tokens: int = 256
    model_name: str = ""  # display name for /health; empty = read from metadata

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServingConfig:
        """Load config from a YAML file, merging with schema defaults."""
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, file_cfg)
        obj = OmegaConf.to_object(merged)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
