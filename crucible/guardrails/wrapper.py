"""Optional wrapper that sits in front of a ModelProtocol and applies RefusalGuardrail."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset

from crucible.guardrails.constants import REFUSAL_MESSAGE
from crucible.guardrails.refusal import RefusalGuardrail
from crucible.training.protocol import ModelProtocol


class GuardrailModelWrapper:
    """Wraps a model and applies refusal guardrail on predict()."""

    def __init__(
        self,
        model: ModelProtocol,
        guardrail: RefusalGuardrail,
        check_before_predict: bool = True,
    ) -> None:
        self._model = model
        self._guardrail = guardrail
        self.check_before_predict = check_before_predict

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._model.train(train_dataset, val_dataset, **kwargs)

    def predict(
        self,
        inputs: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        if self.check_before_predict:
            results: list[str] = []
            for q in inputs:
                in_scope, _ = self._guardrail.is_in_scope(q)
                if not in_scope:
                    results.append(REFUSAL_MESSAGE)
                    continue
                (answer,) = self._model.predict([q], max_new_tokens=max_new_tokens)
                check = self._guardrail.check(q, answer)
                results.append(REFUSAL_MESSAGE if check["should_refuse"] else answer)
            return results

        raw = self._model.predict(inputs, max_new_tokens=max_new_tokens)
        return [
            REFUSAL_MESSAGE if self._guardrail.check(q, a)["should_refuse"] else a
            for q, a in zip(inputs, raw, strict=True)
        ]

    def save(self, path: str | Path) -> None:
        self._model.save(path)

    @classmethod
    def load(cls, path: str | Path) -> GuardrailModelWrapper:
        raise NotImplementedError(
            "Load the inner model and guardrail separately, "
            "then wrap with GuardrailModelWrapper(model, guardrail)"
        )
