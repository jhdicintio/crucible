"""Unit tests for EvaluationConfig."""

from pathlib import Path

import pytest

from crucible.evaluation.config import EvaluationConfig


class TestEvaluationConfig:
    def test_defaults(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.model_path == "outputs/finetuned"
        assert cfg.metrics == ["rouge", "bertscore", "semantic_similarity"]
        assert cfg.max_new_tokens == 256
        assert cfg.reference_column == "output"
        assert cfg.batch_size == 8

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            "model_path: my/model\n"
            "metrics:\n"
            "  - rouge\n"
            "max_new_tokens: 128\n"
            "reference_column: answer\n"
        )
        cfg = EvaluationConfig.from_yaml(yaml_file)

        assert cfg.model_path == "my/model"
        assert cfg.metrics == ["rouge"]
        assert cfg.max_new_tokens == 128
        assert cfg.reference_column == "answer"
        # Unspecified fields keep defaults
        assert cfg.batch_size == 8
        assert cfg.similarity_model == "all-MiniLM-L6-v2"

    def test_from_yaml_accepts_string_path(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text("model_path: other/model\n")
        cfg = EvaluationConfig.from_yaml(str(yaml_file))
        assert cfg.model_path == "other/model"

    def test_from_yaml_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            EvaluationConfig.from_yaml("/nonexistent/path.yaml")


class TestCrucibleConfigWithEvaluation:
    def test_combined_from_yaml(self, tmp_path: Path) -> None:
        from crucible.config import CrucibleConfig

        yaml_file = tmp_path / "combined.yaml"
        yaml_file.write_text(
            "data_processing:\n"
            "  dataset:\n"
            "    name: my/data\n"
            "finetuning:\n"
            "  model:\n"
            "    name: gpt2\n"
            "    approach: full\n"
            "evaluation:\n"
            "  model_path: my/model\n"
            "  metrics:\n"
            "    - rouge\n"
        )
        cfg = CrucibleConfig.from_yaml(yaml_file)
        assert cfg.data_processing.dataset.name == "my/data"
        assert cfg.finetuning.model.name == "gpt2"
        assert cfg.evaluation.model_path == "my/model"
        assert cfg.evaluation.metrics == ["rouge"]
