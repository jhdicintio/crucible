"""WSGI entry point for gunicorn."""

from crucible.serving.app import create_app

app = create_app()
