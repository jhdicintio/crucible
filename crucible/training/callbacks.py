"""Custom HuggingFace Trainer callbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class FileLoggingCallback(TrainerCallback):  # type: ignore[misc]
    """Append every Trainer log event as a JSON line to *log_file*."""

    def __init__(self, log_file: str | Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        entry: dict[str, Any] = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else None,
        }
        for k, v in logs.items():
            entry[k] = round(v, 6) if isinstance(v, float) else v
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
