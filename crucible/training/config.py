"""Structured configs for the fine-tuning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    name: str = "HuggingFaceTB/SmolLM2-135M"
    approach: str = "lora"  # "full", "lora", "qlora"


@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/finetuned"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    eval_strategy: str = "steps"
    log_file: str = "training_log.jsonl"
    text_column: str = "formatted_text"


@dataclass
class FinetuningConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinetuningConfig:
        """Load config from a YAML file, merging with schema defaults."""
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, file_cfg)
        obj = OmegaConf.to_object(merged)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
