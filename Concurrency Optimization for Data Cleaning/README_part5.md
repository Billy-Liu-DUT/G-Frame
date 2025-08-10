# Adaptive Concurrency Manager for vLLM

This project provides a sophisticated framework for dynamically optimizing the request concurrency of a vLLM (Very Large Language Model) server. It is designed for large-scale data processing workloads, such as data cleaning or augmentation, where maximizing throughput is critical.

## Architecture

The system consists of three main components that run asynchronously:

1.  **`vllm_server.py` (VLLM Server)**: This module programmatically launches and manages the vLLM subprocess. It includes a monitor that parses vLLM's log output to extract key performance indicators (KPIs).

2.  **`task_executor.py` (Task Executor)**: This module is responsible for the actual workload. It reads input data, creates a queue of tasks (e.g., API calls for data cleaning), and sends requests to the vLLM server, respecting the concurrency limit set by the manager.

3.  **`adaptive_manager.py` (The Manager)**: This is the central orchestrator. It starts the VLLM server and the Task Executor. It continuously collects metrics from both, merges them, and uses an adjustment policy to dynamically raise or lower the request concurrency limit.

## Setup

### Step 1: Install Dependencies

Ensure you have Python 3.9+ and the required packages installed.

```bash
pip install openai pyyaml asyncio tqdm aiofiles
```
*(Note: `vllm` itself is also a prerequisite and should be installed according to its official documentation.)*

### Step 2: Create the Configuration File

All settings for the project are controlled by a single `config.yaml` file. **Create a file named `config.yaml`** in the root of your project directory using the template provided separately in this repository.

**You must edit the placeholder values in this file to match your environment before running the application.**

### Step 3: Prepare Your Data

-   Place your input data chunks (JSON files) in the directory specified by `input_chunk_dir` in your `config.yaml`.
-   Ensure your original reference data (if needed by your prompt) is available at the path specified by `original_data_path`.

## How to Run

Once your `config.yaml` is configured, launch the application from your terminal:

```bash
python adaptive_manager.py
```

The manager will start the vLLM server, begin processing tasks, and log its progress and concurrency adjustments to the console.