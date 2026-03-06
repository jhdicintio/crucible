"""Configuration for guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardrailsConfig:
    confidence_threshold: float = 0.45
    scope_threshold: float = 0.40
    anchor_phrases: list[str] = field(
        default_factory=lambda: [
            "stock valuation and equity analysis",
            "financial statements and accounting",
            "investment portfolio management",
            "corporate finance and capital structure",
            "banking and interest rates",
            "financial ratios and metrics",
        ]
    )
