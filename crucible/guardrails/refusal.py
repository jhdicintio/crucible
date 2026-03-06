"""Refusal guardrail: scope and confidence checks using sentence embeddings."""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer, util

from crucible.guardrails.config import GuardrailsConfig


class RefusalGuardrail:
    """Refuse out-of-scope or low-confidence answers; use with GuardrailModelWrapper."""

    def __init__(
        self,
        config: GuardrailsConfig,
        embedder: SentenceTransformer | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self.confidence_threshold = config.confidence_threshold
        self.scope_threshold = config.scope_threshold
        self.finance_anchors = self.embedder.encode(config.anchor_phrases)

    def is_in_scope(self, question: str) -> tuple[bool, float]:
        """Check if question is finance-related."""
        q_embedding = self.embedder.encode(question)
        similarities = util.cos_sim(q_embedding, self.finance_anchors)
        max_score = float(similarities.max())
        return max_score >= self.scope_threshold, max_score

    def is_confident(self, question: str, answer: str) -> tuple[bool, float]:
        """Check if answer is semantically grounded relative to question."""
        embeddings = self.embedder.encode([question, answer])
        score = float(util.cos_sim(embeddings[0], embeddings[1])[0, 0])
        return score >= self.confidence_threshold, score

    def check(self, question: str, answer: str) -> dict[str, Any]:
        """Return should_refuse, reason, and scores."""
        in_scope, scope_score = self.is_in_scope(question)
        confident, confidence_score = self.is_confident(question, answer)

        should_refuse = not in_scope or not confident
        reason: str | None = None
        if should_refuse:
            reason = "out_of_scope" if not in_scope else "low_confidence"

        return {
            "should_refuse": should_refuse,
            "reason": reason,
            "scope_score": scope_score,
            "confidence_score": confidence_score,
        }
