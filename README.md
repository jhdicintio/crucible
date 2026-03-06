# Crucible

Fine-tuning small language models (SLMs) for financial use cases.

## Prerequisites

- Python 3.11+
- [Poetry 2.x](https://python-poetry.org/)
- [Task](https://taskfile.dev/) (task runner)
- [Docker](https://www.docker.com/) (optional, for containerised workflows)

## Quick Start

```bash
# Install dependencies
task install

# Install pre-commit hooks
task pre-commit:install

# Run tests
task test

# Lint & format
task lint
task fmt
```

## Running the model serving API

The repo includes a small Flask API that serves a fine-tuned model over HTTP. You can run it on your machine without reading the code.

### What you need

- A **saved model** on disk. The service loads the model from a directory that was produced by the fine-tuning pipeline (e.g. `outputs/finetuned`). If you haven’t trained a model yet, run the fine-tuning workflow first; it will write the model to the path configured in `conf/finetuning.yaml` (by default `outputs/finetuned`).

### Run locally

1. Install dependencies (if you haven’t already):
   `task install`
2. Point the service at your model by editing `conf/serving.yaml` and setting `model_path` to the directory that contains your saved model (default is `outputs/finetuned`).
3. Start the API:
   `task serve`
4. The API listens on **http://0.0.0.0:5000** (all interfaces).
   - **GET /health** — returns 200 and model metadata (name, adaptation method, training date).
   - **POST /ask** — send JSON with a required `question` field and optional `context`; you get back `{"answer": "..."}`.

### Run with Docker

1. Build the image:
   `task docker:build`
2. Start the container with your model directory mounted and port 5000 exposed:
   `task docker:serve`
   This uses `./outputs/finetuned` as the model directory by default. To use another path, pass `MODEL_DIR`:
   `task docker:serve -- MODEL_DIR=/path/to/your/model`
3. The API is available at **http://localhost:5000**. Use **GET /health** and **POST /ask** as above.

### Config file

Serving options (model path, host, port, max generation length) are read from **conf/serving.yaml**. Override the config file path with the environment variable `CRUCIBLE_SERVING_CONFIG` if needed.

## Project Structure

```
crucible/          # Source package
tests/             # Test suite
Taskfile.yml       # Task runner definitions
Dockerfile         # Container for model service
pyproject.toml     # Poetry / project config
```

## Docker

```bash
task docker:build      # production image
task docker:build:dev  # dev image with test tooling
task docker:test       # run tests inside container
```
