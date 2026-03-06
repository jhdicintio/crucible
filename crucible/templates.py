"""Model-to-prompt-template registry for data processing and evaluation alignment."""

from __future__ import annotations

# Alpaca-style (instruction / input / response) — used by SmolLM2, many instruction-tuned SLMs
ALPACA_FULL = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_INPUT_ONLY = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# Map model identifier (name or short key) -> formatting_template and eval input_template
MODEL_TEMPLATES: dict[str, dict[str, str]] = {
    "HuggingFaceTB/SmolLM2-135M": {
        "formatting_template": ALPACA_FULL,
        "eval_input_template": ALPACA_INPUT_ONLY,
    },
    "HuggingFaceTB/SmolLM2-360M": {
        "formatting_template": ALPACA_FULL,
        "eval_input_template": ALPACA_INPUT_ONLY,
    },
    "SmolLM2": {
        "formatting_template": ALPACA_FULL,
        "eval_input_template": ALPACA_INPUT_ONLY,
    },
    "sshleifer/tiny-gpt2": {
        "formatting_template": ALPACA_FULL,
        "eval_input_template": ALPACA_INPUT_ONLY,
    },
    "gpt2": {
        "formatting_template": ALPACA_FULL,
        "eval_input_template": ALPACA_INPUT_ONLY,
    },
}


def get_template_for_model(model_name: str) -> dict[str, str] | None:
    """Return formatting_template and eval_input_template for a model, or None if unknown.

    Tries exact match first, then suffix match (e.g. '.../SmolLM2-360M' -> SmolLM2).
    """
    if not model_name:
        return None
    if model_name in MODEL_TEMPLATES:
        return MODEL_TEMPLATES[model_name]
    # Fallback: last path component or segment (e.g. SmolLM2-135M)
    for key, templates in MODEL_TEMPLATES.items():
        if key in model_name or model_name.endswith(key.split("/")[-1]):
            return templates
    if "SmolLM" in model_name:
        return MODEL_TEMPLATES["SmolLM2"]
    if "gpt2" in model_name.lower():
        return MODEL_TEMPLATES["gpt2"]
    return None
