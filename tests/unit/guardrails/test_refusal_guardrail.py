"""Unit tests for RefusalGuardrail."""

from crucible.guardrails.config import GuardrailsConfig
from crucible.guardrails.constants import REFUSAL_MESSAGE
from crucible.guardrails.refusal import RefusalGuardrail


class TestRefusalGuardrail:
    def test_check_in_scope_finance_question(self) -> None:
        cfg = GuardrailsConfig(confidence_threshold=0.3, scope_threshold=0.3)
        g = RefusalGuardrail(cfg)
        out = g.check(
            "What is the P/E ratio of Apple?",
            "The P/E ratio is a valuation metric.",
        )
        assert "should_refuse" in out
        assert "scope_score" in out
        assert "confidence_score" in out
        assert out["scope_score"] >= 0

    def test_check_refusal_message_is_not_forced(self) -> None:
        """Guardrail returns structure; REFUSAL_MESSAGE is used by wrapper."""
        assert "don't have enough information" in REFUSAL_MESSAGE
