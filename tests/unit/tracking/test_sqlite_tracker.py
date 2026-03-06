"""Unit tests for SQLite experiment tracker."""

import sqlite3
from pathlib import Path

from crucible.tracking.backends.sqlite_backend import SQLiteTracker


class TestSQLiteTracker:
    def test_start_run_returns_run_id(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        t = SQLiteTracker(str(db))
        run_id = t.start_run(run_name="test_run", config={"a": 1})
        assert run_id is not None
        assert t.run_id == run_id

    def test_log_metrics_and_end_run(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        t = SQLiteTracker(str(db))
        t.start_run(config={"seed": 42})
        t.log_metrics({"loss": 1.5}, step=10)
        t.log_metrics({"loss": 1.2}, step=20)
        t.end_run()

        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT key, value, step FROM metrics ORDER BY step").fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0] == ("loss", 1.5, 10)
        assert rows[1] == ("loss", 1.2, 20)

    def test_config_stored_in_runs(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        t = SQLiteTracker(str(db))
        config = {"finetuning": {"model": {"name": "gpt2"}}}
        t.start_run(run_name="cfg_test", config=config)
        t.end_run()

        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT config_json FROM runs").fetchone()
        conn.close()
        assert row is not None
        assert "gpt2" in row[0]
