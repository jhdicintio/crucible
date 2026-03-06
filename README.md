# Crucible

I built Crucible as an end-to-end pipeline for fine-tuning a small language model on financial QA data and serving it behind a REST API. The goal was to stand up something runnable on a laptop (CPU-first) with no existing ML infra—so everything is config-driven, reproducible, and single-command where possible.

---

## Architecture and key design decisions

**Pipeline shape.** The flow is: load data from Hugging Face → clean and split → format with a prompt template → fine-tune → evaluate on a held-out test set → (optionally) serve via API. I wanted one config file to drive the whole run so a reviewer could reproduce results without touching code. I use **Flyte** for workflow steps (load, clean, split, format, finetune, evaluate) and **OmegaConf** for YAML config. That gives a clear DAG, serializable inputs/outputs, and a single source of truth per run (e.g. `conf/experiments/experiment_0_baseline.yaml`).

**Data.** The main dataset is **gbharti/finance-alpaca** (instruction/input/output in Alpaca style). The pipeline loads it via `datasets`, then: drop NaNs and duplicates, strip whitespace, optionally lowercase; split into train/val/test with a fixed seed; format each row with a configurable template into a single string (e.g. `### Instruction: ... ### Input: ... ### Response: ...`). I added support for **sampling** (e.g. by semantic diversity or stratified) so I could cap training size for CPU and time—the dataset is large and data quality is uneven, so I document the sampling strategy in the experiment YAML and treat it as part of the run’s lineage.

**Template.** The prompt template is chosen so that training and inference use the same format. I keep an Alpaca-style block (Instruction / Input / Response) and a **model–template registry**: when you pick a model in config (e.g. SmolLM2), the pipeline can align the data-formatting template and the evaluation input template to that model. That avoids training on one format and serving with another.

**Fine-tuning.** I support full fine-tuning, **LoRA**, and **QLoRA**. For a laptop with no GPU I default to LoRA so training finishes in a reasonable time; hyperparameters (learning rate, batch size, epochs, LoRA r/alpha, etc.) all live in YAML. The training step sets a random seed, logs loss and validation metrics to a JSONL file and (when tracking is on) to the experiment backend, and saves the model to disk in a loadable form (HuggingFace `save_pretrained` plus a small metadata JSON so the server knows base model and adaptation method).

**Evaluation.** I wanted more than one number. The eval step runs **ROUGE**, **BERTScore**, and **semantic similarity** (sentence embeddings) against reference answers on the test set, and reports them in a structured way. I also added a **refusal metric**: if the test set has an `answerable` (or similar) flag, the pipeline can compute how often the model correctly refuses unanswerable or out-of-scope questions vs. false refusals. I defined a small **metric protocol** so additional metrics (or custom ones) can be plugged in without changing the orchestrator.

**Refusal and guardrails.** A banking assistant that confidently hallucinates is worse than one that says “I don’t know.” So I added: (1) an optional **system prompt** (in data formatting and in evaluation) that instructs the model to answer only finance-related questions and to respond with a fixed refusal phrase when outside knowledge or scope; (2) an optional **refusal guardrail** (scope + confidence checks via sentence embeddings) that can sit in front of the model at inference and override low-confidence or out-of-scope answers with that refusal message; (3) the **refusal evaluation metric** above so I can measure how well the model (or guardrail) refuses when it should. Tradeoff: aggressive refusal improves safety but can over-refuse on valid questions; I document the thresholds in config and tune them per use case.

**Experiment tracking.** Each training run can be logged to a **SQLite** backend: full config (data + training + eval), per-step metrics, and a link to the saved model directory. That gives reproducibility and a way to compare runs without MLflow or W&B; the design allows swapping in MLflow or WandB later via the same tracker interface.

**Serving.** The API is a small **Flask** app. It loads the saved model once at startup (no retraining). **GET /health** returns 200 and model metadata (name, adaptation method, training date). **POST /ask** takes JSON with a required `question` and optional `context` and returns the model’s answer. The server reads options (model path, port, max tokens) from a YAML config so you can point it at any run’s output directory.

---

## Setup and run instructions

**Prerequisites**

- **Python 3.11 or 3.12** (the project pins `>=3.11,<3.13` for dependency compatibility).
- **Poetry** for install and env.
- **Task** ([taskfile.dev](https://taskfile.dev/)) for single-command targets. (`brew install go-task`)
- **Docker** (optional) for serving in a container.

**Install**

```bash
task install
task pre-commit:install
```

**Tests**

```bash
task test
```

**Lint / format**

```bash
task lint
task fmt
```

---

## Reproducing training results

1. Pick an experiment config under `conf/experiments/` (e.g. `experiment_0_baseline.yaml`). It defines dataset, sampling, split, template, model, LoRA/training knobs, eval metrics, and tracking.
2. Run the full pipeline (data → train → eval) with that config:

   ```bash
   task train -- conf/experiments/experiment_0_baseline.yaml
   ```

   Omit the path to use the default (`conf/experiments/experiment_0_baseline.yaml`):

   ```bash
   task train
   ```

   The task runs `pyflyte run crucible/pipeline.py full_pipeline --config_path <expt.yaml>`. The pipeline writes the model to the `output_dir` in the config (e.g. `outputs/experiment_0_baseline`) and writes evaluation results to the path in `evaluation.output_file`.

3. All inputs to the run are in that single YAML (dataset name, sample size/strategy, seed, template, model name, LoRA/training args, eval metrics). So “reproduce” = same YAML + same Python/env = same data subset, split, and training run.

---

**Data sampling.** When `sample_size` is set (e.g. 1000), `sample_strategy` can be `random`, `first`, `stratified`, `diversity`, or **`quality`** — which drops poor examples (bad length, near-duplicates) then takes a diverse subset. Experiment 0 uses `quality` so the small training set is a better subset rather than distribution-preserving.

---

## Model performance summary and limitations

- **What the metrics do:** ROUGE and BERTScore measure overlap and semantic similarity with reference answers; they don’t measure correctness of facts or suitability for banking. The refusal metric measures how often the model (or guardrail) refuses when the example is marked unanswerable/out-of-scope vs. when it wrongly refuses answerable questions.
- **Honest assessment:** The default setup uses a small model (e.g. SmolLM2-135M) and CPU-friendly LoRA. It’s good for iteration and for proving the pipeline end-to-end. Output quality is limited by model size and by the fact that Finance Alpaca is mixed quality—some examples are strong regulatory Q&A, others are generic or noisy. I didn’t fabricate numbers; the pipeline really trains and evaluates. With more time I’d add human eval on a fixed set of banking questions and compare multiple configs (e.g. different data subsets or prompt formats) in a structured way.

---

## API documentation

**Health check**

```bash
curl -s http://localhost:8080/health
```

Example response:

```json
{
  "status": "ok",
  "model": {
    "name": "HuggingFaceTB/SmolLM2-135M",
    "adaptation_method": "lora",
    "training_date": null
  }
}
```

**Ask a question**

```bash
curl -s -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Tier 1 capital ratio?"}'
```

With optional context:

```bash
curl -s -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the ratio in this document?", "context": "The bank reported a Tier 1 ratio of 14.2%."}'
```

Response:

```json
{
  "answer": "..."
}
```

**Serving the API**

- **Locally:** Ensure a trained model exists at the path in the serving config (default `conf/defaults/serving.yaml`, model dir `outputs/finetuned`). Then:

  ```bash
  task serve
  ```

  The API listens on `http://0.0.0.0:8080`. Override config path with `CRUCIBLE_SERVING_CONFIG` if needed.

- **Docker:** Build and run with the model directory mounted:

  ```bash
  task docker:build
  task docker:serve
  ```

  Default mount is `./outputs/finetuned`. To use another dir:

  ```bash
  task docker:serve -- MODEL_DIR=/path/to/your/model
  ```

---

## What I would do differently with more time

- **Data curation:** Score training examples by relevance and quality (e.g. embedding similarity to a banking FAQ set, length/heuristic filters), then compare training on a curated subset vs. a random sample of the same size and measure impact on downstream metrics and human judgment.
- **Evaluation depth:** Add a small human eval protocol (e.g. 50–100 questions) with dimensions like correctness, refusal appropriateness, and tone, and track those alongside ROUGE/BERTScore so the numbers map to something a banker would care about.
- **Multi-run comparison:** Train at least two meaningfully different configs (e.g. different sample strategies, different LoRA rank, or different prompt formats) and log everything to the same experiment backend so I could produce a short comparison table and a recommendation for which config to deploy.
- **Larger model on GPU:** Move to a 0.5B–1B model and GPU for real training runs; keep the current small model + CPU path for fast iteration and demos.
- **Refusal tuning:** Sweep guardrail thresholds (scope/confidence) and document the precision–recall tradeoff for refusals so we can choose a policy that matches risk tolerance.

---

## Project structure

```
crucible/           # Main package
  data/             # Load, clean, split, format (Flyte tasks + workflow)
  training/         # Model (CausalLMModel), LoRA/QLoRA, Trainer, callbacks
  evaluation/       # Metrics (ROUGE, BERTScore, similarity, refusal), evaluator
  guardrails/       # System prompt constant, RefusalGuardrail, wrapper
  tracking/         # Experiment tracker protocol + SQLite backend
  serving/          # Flask app, loader, config
  config.py         # Top-level CrucibleConfig (data + training + eval + tracking)
  pipeline.py       # full_pipeline workflow + run_full_pipeline()
  templates.py      # Model → prompt template registry
conf/
  defaults/         # Default YAMLs for data, finetuning, evaluation, serving
  experiments/     # Full run configs (e.g. experiment_0_baseline.yaml)
tests/
Taskfile.yml       # install, test, lint, fmt, serve, docker:build, docker:serve
prompt-log.md      # Log of all AI tool interactions (required for assessment)
```

---

## Single-command startup

- **Train (full pipeline):** Run the pipeline with an experiment YAML as above; the exact one-liner is in “Reproducing training results.”
- **Serve:** `task serve` (local) or `task docker:serve` (container with model mount).

All interactions with the AI tool used to build this pipeline are logged in **prompt-log.md** in the repo root for review.
