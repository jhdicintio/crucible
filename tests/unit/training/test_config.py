from pathlib import Path

import pytest

from crucible.training.config import FinetuningConfig


class TestFinetuningConfig:
    def test_defaults(self) -> None:
        cfg = FinetuningConfig()
        assert cfg.model.approach == "lora"
        assert cfg.training.seed == 42
        assert cfg.lora.r == 8

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "ft.yaml"
        yaml_file.write_text(
            "model:\n"
            "  name: gpt2\n"
            "  approach: full\n"
            "training:\n"
            "  num_epochs: 1\n"
            "  learning_rate: 1.0e-5\n"
        )
        cfg = FinetuningConfig.from_yaml(yaml_file)

        assert cfg.model.name == "gpt2"
        assert cfg.model.approach == "full"
        assert cfg.training.num_epochs == 1
        assert cfg.training.learning_rate == 1e-5
        assert cfg.lora.r == 8  # default preserved

    def test_from_yaml_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            FinetuningConfig.from_yaml("/nonexistent.yaml")


class TestCrucibleConfig:
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
        )
        cfg = CrucibleConfig.from_yaml(yaml_file)
        assert cfg.data_processing.dataset.name == "my/data"
        assert cfg.finetuning.model.name == "gpt2"
        assert cfg.finetuning.model.approach == "full"
