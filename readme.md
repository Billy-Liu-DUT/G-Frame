# G-Frame

G-Frame is an adaptive multi-agent framework for scientific-language-model
workflows in chemistry. It supports source processing and synthetic record
construction, DeepSpeed model training, runtime observation and adaptive
request concurrency, OpenAI-compatible inference, GraphRAG-backed retrieval,
and local chat.

## Workflow

1. **Data processing and synthesis:** `src/g_frame/` and `scripts/` process
   operator-supplied JSONL, construct traceable reasoning and QA records, and
   prepare run-local handoff files.
2. **Runtime control:** `src/g_frame/telemetry.py`,
   `src/g_frame/orchestration.py`, and `scripts/live_gpu_smoke.py` combine
   engine and task observations, then apply bounded concurrency decisions to
   active work requests.
3. **Model training:** `llm_training/` provides DeepSpeed entrypoints for
   continued pre-training and full-parameter supervised fine-tuning.
4. **Inference, retrieval, and chat:** `llm_inference/`, `graph_rag/`, and
   `chat_interface/` provide OpenAI-compatible serving, GraphRAG utilities,
   and a local chat interface.
5. **Research utilities:** `deep_research/` contains the literature research
   workflow and its supporting configuration.

## Repository Structure

```text
.
+-- src/g_frame/                         Workflow, prompts, telemetry, and actions
+-- llm_training/                        DeepSpeed pre-training and SFT entrypoints
+-- concurrency_optimization_for_data_cleaning/
|   +-- adaptive_manager.py              Data-processing concurrency manager
|   +-- task_executor.py                 Asynchronous task execution
|   +-- vllm_server.py                   vLLM server helper
+-- llm_inference/                       OpenAI-compatible inference launch assets
+-- graph_rag/                           GraphRAG ingestion and serving utilities
+-- chat_interface/                      Local chat interface
+-- deep_research/                       Literature research workflow
+-- configs/                             Example configuration shapes
+-- scripts/                             Workflow and smoke utilities
+-- tests/                               CPU-safe tests
+-- pyproject.toml
```

Runtime outputs, model files, caches, local environments, and generated
artifacts are excluded by `.gitignore`. Configure output locations outside
tracked source paths for normal runs.

## Installation

Clone the repository:

```bash
git clone https://github.com/Billy-Liu-DUT/G-Frame.git
cd G-Frame
```

Create and activate an environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[runtime,training,dev]"
```

For GPU training or serving, install PyTorch, DeepSpeed, and vLLM builds that
match the host CUDA driver before starting those components.

## Usage

### Run Data Processing and Synthesis

Run the local workflow check:

```bash
export PYTHONPATH=src
python scripts/run_mock_pipeline.py --output-dir runs/data-flow
```

Provide a caller-owned JSONL input with `--source-path <path>` when needed.
The command writes its outputs only under the selected run directory.

### Run Runtime Control

Start an OpenAI-compatible endpoint separately, then run the bounded control
check against it:

```bash
export PYTHONPATH=src
python scripts/live_gpu_smoke.py \
  --base-url <OPENAI_COMPATIBLE_BASE_URL> \
  --api-key <API_KEY> \
  --model <MODEL_NAME> \
  --decision-model <DECISION_MODEL_NAME> \
  --output-dir runs/live-smoke
```

Use `--help` to inspect the telemetry, request-limit, and output options.

### Run Training

Validate configuration before launching a GPU job:

```bash
deepspeed --num_gpus=1 llm_training/run_pretraining.py --config <config>
deepspeed --num_gpus=1 llm_training/run_sft.py --config <config> --validate
```

Start the corresponding training stage only after supplying the required
model, input, evaluation, and output paths:

```bash
deepspeed --num_gpus=1 llm_training/run_pretraining.py --config <config> --run
deepspeed --num_gpus=1 llm_training/run_sft.py --config <config> --run
```

`scripts/run_mock_training_schedule.py` coordinates the three-stage smoke
path, including checkpoint handoff, when invoked with `--run`.

### Run Inference, Retrieval, and Chat

The component directories contain their own operational notes:

```bash
cd llm_inference && bash inference_command.sh
cd graph_rag && python serve.py
cd chat_interface && python chat_app.py
```

Configure host-specific paths, ports, and credentials before starting these
services.

## Smoke Tests

After installation, run the CPU-safe checks:

```bash
export PYTHONPATH=src
python -m unittest discover -s tests -p "test_*.py" -v
gframe smoke --offline --output-dir runs/offline-smoke
```

These commands do not start model serving or DeepSpeed training.

## License

See `LICENSE` for license terms.

## Acknowledgements

This project is developed by the Digital Chemistry Research Group at Dalian
University of Technology.
