"""Tests for the model serving Flask app (POST /ask, GET /health)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask.testing import FlaskClient

from crucible.serving.app import create_app


@pytest.fixture
def mock_model() -> MagicMock:
    """Fake ModelProtocol that returns a fixed answer."""
    m = MagicMock()
    m.predict.return_value = ["mock model answer"]
    return m


@pytest.fixture
def serving_config_dir(tmp_path: Path) -> Path:
    """Create a temp dir with serving.yaml and model dir containing crucible_metadata.json."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "crucible_metadata.json").write_text(
        '{"approach": "lora", "base_model": "test/model", "training_date": "2025-01-01T12:00:00Z"}'
    )
    serving_yaml = tmp_path / "serving.yaml"
    serving_yaml.write_text(
        f"model_path: {model_dir}\nhost: 0.0.0.0\nport: 5000\nmax_new_tokens: 64\n"
    )
    return tmp_path


@pytest.fixture
def app_client(
    mock_model: MagicMock, serving_config_dir: Path
) -> Generator[FlaskClient, None, None]:
    """Flask test client with load_model patched to return mock_model."""
    config_path = serving_config_dir / "serving.yaml"
    with patch("crucible.serving.app.load_model", return_value=mock_model):
        app = create_app(config_path=config_path)
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


def test_health_returns_200_and_metadata(app_client: FlaskClient) -> None:
    """GET /health returns 200 with status and model metadata."""
    r = app_client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["model"]["name"] == "test/model"
    assert data["model"]["adaptation_method"] == "lora"
    assert data["model"]["training_date"] == "2025-01-01T12:00:00Z"


def test_ask_with_question_returns_answer(app_client: FlaskClient, mock_model: MagicMock) -> None:
    """POST /ask with question returns 200 and model answer."""
    r = app_client.post(
        "/ask",
        json={"question": "What is 2+2?"},
        content_type="application/json",
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["answer"] == "mock model answer"
    mock_model.predict.assert_called_once()
    call_args = mock_model.predict.call_args
    assert call_args[0][0] == ["What is 2+2?"]
    assert call_args[1]["max_new_tokens"] == 64


def test_ask_with_question_and_context_formats_prompt(
    app_client: FlaskClient, mock_model: MagicMock
) -> None:
    """POST /ask with context prepends context to the prompt."""
    r = app_client.post(
        "/ask",
        json={
            "question": "What is the sentiment?",
            "context": "Revenue increased by 20%.",
        },
        content_type="application/json",
    )
    assert r.status_code == 200
    mock_model.predict.assert_called_once()
    prompt = mock_model.predict.call_args[0][0][0]
    assert "Context: Revenue increased by 20%." in prompt
    assert "Question: What is the sentiment?" in prompt


def test_ask_missing_question_returns_400(app_client: FlaskClient) -> None:
    """POST /ask without question returns 400."""
    r = app_client.post("/ask", json={}, content_type="application/json")
    assert r.status_code == 400
    data = r.get_json()
    assert "error" in data and "question" in data["error"].lower()


def test_ask_empty_question_returns_400(app_client: FlaskClient) -> None:
    """POST /ask with empty string question returns 400."""
    r = app_client.post("/ask", json={"question": "   "}, content_type="application/json")
    assert r.status_code == 400
