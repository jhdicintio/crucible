"""Unit tests for model-template alignment."""

from pathlib import Path

from crucible.config import CrucibleConfig
from crucible.templates import ALPACA_FULL, ALPACA_INPUT_ONLY, get_template_for_model


class TestGetTemplateForModel:
    def test_known_model_exact(self) -> None:
        out = get_template_for_model("HuggingFaceTB/SmolLM2-135M")
        assert out is not None
        assert out["formatting_template"] == ALPACA_FULL
        assert out["eval_input_template"] == ALPACA_INPUT_ONLY

    def test_unknown_model_returns_none(self) -> None:
        assert get_template_for_model("unknown/xyz-123") is None

    def test_empty_returns_none(self) -> None:
        assert get_template_for_model("") is None


class TestCrucibleConfigModelTemplateAlignment:
    def test_from_yaml_applies_model_templates(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "full.yaml"
        yaml_file.write_text(
            "finetuning:\n  model:\n    name: HuggingFaceTB/SmolLM2-135M\n    approach: lora\n"
        )
        cfg = CrucibleConfig.from_yaml(yaml_file)
        assert "### Instruction:" in cfg.data_processing.formatting.template
        assert "{instruction}" in cfg.data_processing.formatting.template
        assert "{output}" in cfg.data_processing.formatting.template
        assert "### Response:" in cfg.evaluation.input_template
        assert "{output}" not in cfg.evaluation.input_template

    def test_use_model_template_false_keeps_user_template(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "full.yaml"
        yaml_file.write_text(
            "data_processing:\n"
            "  formatting:\n"
            "    use_model_template: false\n"
            "    template: 'Custom {x}'\n"
            "finetuning:\n"
            "  model:\n"
            "    name: HuggingFaceTB/SmolLM2-135M\n"
        )
        cfg = CrucibleConfig.from_yaml(yaml_file)
        assert cfg.data_processing.formatting.template == "Custom {x}"
