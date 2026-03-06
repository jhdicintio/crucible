"""Flask application factory and routes for model serving."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from crucible.serving.config import ServingConfig
from crucible.serving.loader import load_model

_METADATA_FILE = "crucible_metadata.json"


def create_app(config_path: str | Path | None = None) -> Flask:
    """Create and configure the Flask app. Loads config from env or default path."""
    app = Flask(__name__)
    path: str | Path = (
        config_path
        if config_path is not None
        else os.environ.get("CRUCIBLE_SERVING_CONFIG", "conf/serving.yaml")
    )
    config = ServingConfig.from_yaml(path)
    app.config["serving_config"] = config

    model_path = Path(config.model_path)
    app.config["model"] = load_model(model_path)

    metadata_path = model_path / _METADATA_FILE
    if metadata_path.exists():
        model_metadata: dict[str, Any] = json.loads(metadata_path.read_text())
        app.config["model_metadata"] = model_metadata
    else:
        app.config["model_metadata"] = {}

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        meta = app.config["model_metadata"]
        name = config.model_name or meta.get("base_model", "unknown")
        return (
            jsonify(
                {
                    "status": "ok",
                    "model": {
                        "name": name,
                        "adaptation_method": meta.get("approach", "unknown"),
                        "training_date": meta.get("training_date"),
                    },
                }
            ),
            200,
        )

    @app.route("/ask", methods=["POST"])
    def ask() -> tuple[Response, int]:
        body: dict[str, Any] = request.get_json(silent=True) or {}
        question = body.get("question")
        if question is None or (isinstance(question, str) and not question.strip()):
            return jsonify({"error": "Missing or empty 'question' field"}), 400
        context = body.get("context")
        prompt = f"Context: {context}\n\nQuestion: {question}" if context else question
        model = app.config["model"]
        max_new_tokens = config.max_new_tokens
        answers = model.predict([prompt], max_new_tokens=max_new_tokens)
        return jsonify({"answer": answers[0] if answers else ""}), 200

    return app
