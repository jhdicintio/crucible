"""SQLite-backed experiment tracker for full reproducibility."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class SQLiteTracker:
    """Store runs, params, and metrics in SQLite for complete experiment reproducibility."""

    def __init__(self, db_path: str = "experiments.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._run_id: str | None = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_name TEXT,
                    config_json TEXT,
                    start_ts REAL,
                    end_ts REAL,
                    status TEXT DEFAULT 'running'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS params (
                    run_id TEXT,
                    key TEXT,
                    value TEXT,
                    PRIMARY KEY (run_id, key),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT,
                    key TEXT,
                    value REAL,
                    step INTEGER,
                    ts REAL,
                    PRIMARY KEY (run_id, key, step),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
                """
            )
            self._conn.commit()
        return self._conn

    def start_run(
        self,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        conn = self._ensure_conn()
        self._run_id = str(uuid.uuid4())
        now = time.time()
        config_json = json.dumps(config, default=str) if config else None
        conn.execute(
            "INSERT INTO runs (run_id, run_name, config_json, start_ts, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (self._run_id, run_name or self._run_id[:8], config_json, now, "running"),
        )
        conn.commit()
        return self._run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self._run_id is None:
            return
        conn = self._ensure_conn()
        for key, value in _flatten_params(params).items():
            conn.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                (self._run_id, key, json.dumps(value, default=str)),
            )
        conn.commit()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._run_id is None:
            return
        conn = self._ensure_conn()
        now = time.time()
        step_val = step if step is not None else -1
        for key, value in metrics.items():
            if isinstance(value, int | float):
                conn.execute(
                    "INSERT INTO metrics (run_id, key, value, step, ts) VALUES (?, ?, ?, ?, ?)",
                    (self._run_id, key, float(value), step_val, now),
                )
        conn.commit()

    def log_artifact(self, path: str | Path) -> None:
        # Record artifact path for reproducibility; callers can copy files if needed.
        if self._run_id is None:
            return
        conn = self._ensure_conn()
        conn.execute(
            "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
            (self._run_id, f"artifact:{Path(path).name}", str(Path(path).resolve())),
        )
        conn.commit()

    def end_run(self) -> None:
        if self._run_id is None:
            return
        conn = self._ensure_conn()
        conn.execute(
            "UPDATE runs SET end_ts = ?, status = ? WHERE run_id = ?",
            (time.time(), "finished", self._run_id),
        )
        conn.commit()
        self._run_id = None


def _flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict for key-value storage; values are JSON-serialized."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict) and v and not _is_leaf(v):
            out.update(_flatten_params(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _is_leaf(d: dict[str, Any]) -> bool:
    return not any(isinstance(v, dict) for v in d.values())
