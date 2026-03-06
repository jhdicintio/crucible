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

## Project Structure

```
crucible/          # Source package
tests/             # Test suite
Taskfile.yml       # Task runner definitions
Dockerfile         # Multi-stage Docker build (dev + prod)
pyproject.toml     # Poetry / project config
```

## Docker

```bash
task docker:build      # production image
task docker:build:dev  # dev image with test tooling
task docker:test       # run tests inside container
```
