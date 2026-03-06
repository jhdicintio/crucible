"""Configuration dataclasses for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from crucible.guardrails.config import GuardrailsConfig


@dataclass
class EvaluationConfig:
    model_path: str = "outputs/finetuned"
    metrics: list[str] = field(
        default_factory=lambda: ["rouge", "bertscore", "semantic_similarity"]
    )
    max_new_tokens: int = 256
    input_template: str = (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
    reference_column: str = "output"
    column_mapping: dict[str, str] = field(default_factory=dict)
    system_prompt: str | None = None
    refusal_metadata_column: str | None = None
    use_guardrail: bool = False
    guardrails: GuardrailsConfig | None = None
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    similarity_model: str = "all-MiniLM-L6-v2"
    output_file: str = "evaluation_results.json"
    batch_size: int = 8

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationConfig:
        """Load config from a YAML file, merging with schema defaults."""
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, file_cfg)
        obj = OmegaConf.to_object(merged)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
