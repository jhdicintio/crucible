"""CausalLMModel — concrete implementation of ModelProtocol for HuggingFace causal LMs."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from crucible.tracking.protocol import ExperimentTracker
from crucible.training.callbacks import FileLoggingCallback, TrackingCallback
from crucible.training.config import FinetuningConfig

logger = logging.getLogger(__name__)

_METADATA_FILE = "crucible_metadata.json"


class CausalLMModel:
    """Train / predict / save / load a causal language model.

    Supports three fine-tuning approaches controlled by ``config.model.approach``:

    * ``"full"``  — update every parameter
    * ``"lora"``  — freeze base weights, train low-rank adapters (peft)
    * ``"qlora"`` — 4-bit quantised base + LoRA adapters (requires ``bitsandbytes``)
    """

    def __init__(self, config: FinetuningConfig) -> None:
        self.config = config
        self.approach = config.model.approach
        self.model_name = config.model.name
        set_seed(config.training.seed)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.model_name)
        self.model: PreTrainedModel = self._load_base_model()
        self._apply_peft_if_needed()
        self._ensure_pad_token()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_base_model(self) -> PreTrainedModel:
        model_kwargs: dict[str, Any] = {}
        if self.approach == "qlora":
            model_kwargs["quantization_config"] = self._build_bnb_config()
        return AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

    def _build_bnb_config(self) -> Any:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "QLoRA requires bitsandbytes. Install with: pip install bitsandbytes"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA GPU. Use 'lora' or 'full' on CPU / MPS.")
        qcfg = self.config.quantization
        return BitsAndBytesConfig(
            load_in_4bit=qcfg.load_in_4bit,
            bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, qcfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant,
        )

    def _apply_peft_if_needed(self) -> None:
        if self.approach not in ("lora", "qlora"):
            return
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        if self.approach == "qlora":
            from peft import prepare_model_for_kbit_training

            self.model = prepare_model_for_kbit_training(self.model)

        lcfg = self.config.lora
        peft_config = PeftLoraConfig(
            r=lcfg.r,
            lora_alpha=lcfg.lora_alpha,
            lora_dropout=lcfg.lora_dropout,
            target_modules=list(lcfg.target_modules),
            bias=lcfg.bias,
            task_type=TaskType[lcfg.task_type],
        )
        self.model = get_peft_model(self.model, peft_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            "LoRA applied — trainable: %s / %s (%.2f%%)",
            trainable,
            total,
            100 * trainable / total,
        )

    def _ensure_pad_token(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(self.model, "config"):
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        tcfg = self.config.training

        def _tok(examples: dict[str, list[str]]) -> dict[str, Any]:
            return self.tokenizer(  # type: ignore[no-any-return]
                examples[tcfg.text_column],
                truncation=True,
                max_length=tcfg.max_seq_length,
            )

        num_proc = getattr(tcfg, "tokenization_num_proc", 1) or 1
        return dataset.map(
            _tok,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc if num_proc > 1 else None,
        )

    # ------------------------------------------------------------------
    # Public API  (satisfies ModelProtocol)
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> dict[str, Any]:
        """Fine-tune and return metrics. Optionally log to an ExperimentTracker."""
        tcfg = self.config.training
        tok_train = self._tokenize_dataset(train_dataset)
        tok_val = self._tokenize_dataset(val_dataset) if val_dataset else None

        training_args = TrainingArguments(
            output_dir=tcfg.output_dir,
            num_train_epochs=tcfg.num_epochs,
            per_device_train_batch_size=tcfg.per_device_train_batch_size,
            per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
            learning_rate=tcfg.learning_rate,
            weight_decay=tcfg.weight_decay,
            warmup_ratio=tcfg.warmup_ratio,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            fp16=tcfg.fp16,
            bf16=tcfg.bf16,
            dataloader_num_workers=getattr(tcfg, "dataloader_num_workers", 0),
            dataloader_pin_memory=getattr(tcfg, "dataloader_pin_memory", True),
            logging_steps=tcfg.logging_steps,
            eval_strategy=tcfg.eval_strategy if tok_val else "no",
            eval_steps=tcfg.eval_steps if tok_val else None,
            save_steps=tcfg.save_steps,
            seed=tcfg.seed,
            report_to="none",
        )

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        callbacks: list[Any] = [FileLoggingCallback(tcfg.log_file)]
        if tracker is not None:
            callbacks.append(TrackingCallback(tracker))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tok_train,
            eval_dataset=tok_val,
            data_collator=collator,
            callbacks=callbacks,
        )

        result = trainer.train()
        metrics: dict[str, Any] = dict(result.metrics)

        if tok_val:
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)

        return metrics

    def predict(self, inputs: list[str], max_new_tokens: int = 256) -> list[str]:
        """Generate text completions."""
        self.model.eval()
        encoded = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)  # type: ignore[no-any-return]

    def save(self, path: str | Path) -> None:
        """Persist model, tokenizer, and crucible metadata."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        metadata = {
            "approach": self.approach,
            "base_model": self.model_name,
            "training_date": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        (save_dir / _METADATA_FILE).write_text(json.dumps(metadata))
        logger.info("Model saved to %s", save_dir)

    @classmethod
    def load(cls, path: str | Path) -> CausalLMModel:
        """Load a previously saved model (full, LoRA, or QLoRA)."""
        load_dir = Path(path)
        metadata = json.loads((load_dir / _METADATA_FILE).read_text())
        approach = metadata["approach"]

        tokenizer = AutoTokenizer.from_pretrained(load_dir)

        if approach in ("lora", "qlora"):
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(load_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(load_dir)

        instance = cls.__new__(cls)
        instance.config = FinetuningConfig()
        instance.model = model
        instance.tokenizer = tokenizer
        instance.model_name = metadata["base_model"]
        instance.approach = approach
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return instance
