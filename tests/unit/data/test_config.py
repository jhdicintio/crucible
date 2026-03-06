from pathlib import Path

import pytest

from crucible.data.config import DataProcessingConfig


class TestDataProcessingConfig:
    def test_defaults(self) -> None:
        cfg = DataProcessingConfig()
        assert cfg.dataset.name == "takala/financial_phrasebank"
        assert cfg.split.train_ratio == 0.8

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("dataset:\n  name: my/dataset\nsplit:\n  seed: 99\n")
        cfg = DataProcessingConfig.from_yaml(yaml_file)

        assert cfg.dataset.name == "my/dataset"
        assert cfg.split.seed == 99
        # Unspecified fields keep their defaults
        assert cfg.cleaning.drop_nans is True
        assert cfg.formatting.output_column == "formatted_text"

    def test_from_yaml_accepts_string_path(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("dataset:\n  name: other/ds\n")
        cfg = DataProcessingConfig.from_yaml(str(yaml_file))
        assert cfg.dataset.name == "other/ds"

    def test_from_yaml_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            DataProcessingConfig.from_yaml("/nonexistent/path.yaml")
