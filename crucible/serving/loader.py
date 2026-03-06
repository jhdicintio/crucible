"""Load a saved model by reading crucible metadata and dispatching to the right class."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from crucible.training.model import CausalLMModel

if TYPE_CHECKING:
    from crucible.training.protocol import ModelProtocol

_METADATA_FILE = "crucible_metadata.json"

# Registry: map from metadata "approach" (or future model_type) to concrete class.
# Currently only CausalLMModel writes this metadata.
_MODEL_REGISTRY: dict[str, type[CausalLMModel]] = {
    "full": CausalLMModel,
    "lora": CausalLMModel,
    "qlora": CausalLMModel,
}


def load_model(path: str | Path) -> ModelProtocol:
    """Load a model from a saved directory using crucible_metadata.json."""
    load_dir = Path(path)
    metadata_path = load_dir / _METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"No {_METADATA_FILE} in {load_dir}")
    metadata = json.loads(metadata_path.read_text())
    approach = metadata.get("approach", "lora")
    model_cls = _MODEL_REGISTRY.get(approach)
    if model_cls is None:
        raise ValueError(f"Unknown model approach in metadata: {approach}")
    return model_cls.load(load_dir)
