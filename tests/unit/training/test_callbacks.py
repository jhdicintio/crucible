import json
from pathlib import Path
from unittest.mock import MagicMock

from crucible.training.callbacks import FileLoggingCallback


class TestFileLoggingCallback:
    def test_writes_jsonl_entry(self, tmp_path: Path) -> None:
        log_file = tmp_path / "logs" / "train.jsonl"
        cb = FileLoggingCallback(log_file)

        state = MagicMock()
        state.global_step = 10
        state.epoch = 0.5

        cb.on_log(
            args=MagicMock(),
            state=state,
            control=MagicMock(),
            logs={"loss": 2.345678, "learning_rate": 1e-4},
        )

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["step"] == 10
        assert entry["epoch"] == 0.5
        assert entry["loss"] == 2.345678
        assert entry["learning_rate"] == 0.0001

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        log_file = tmp_path / "train.jsonl"
        cb = FileLoggingCallback(log_file)

        for step in range(3):
            state = MagicMock()
            state.global_step = step
            state.epoch = step * 0.1
            cb.on_log(
                args=MagicMock(),
                state=state,
                control=MagicMock(),
                logs={"loss": float(step)},
            )

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_skips_none_logs(self, tmp_path: Path) -> None:
        log_file = tmp_path / "train.jsonl"
        cb = FileLoggingCallback(log_file)
        cb.on_log(args=MagicMock(), state=MagicMock(), control=MagicMock(), logs=None)
        assert not log_file.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log_file = tmp_path / "deep" / "nested" / "train.jsonl"
        FileLoggingCallback(log_file)
        assert log_file.parent.exists()
