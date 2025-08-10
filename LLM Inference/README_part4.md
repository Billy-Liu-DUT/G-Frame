# Serving a Model with vLLM's OpenAI-Compatible Server

This document provides instructions on how to launch a local inference server for a Large Language Model (LLM) using vLLM. The server exposes an OpenAI-compatible API endpoint, making it easy to integrate with a wide range of clients and applications, such as the chat interface provided in this project.

## Usage

The following command launches the vLLM API server. You will need to replace the placeholder values (e.g., `<your_gpu_id>`, `<path_to_your_model>`) with your specific configuration.

### Command

```bash
export CUDA_VISIBLE_DEVICES="<your_gpu_id>" && \
python -m vllm.entrypoints.openai.api_server \
    --model <path_to_your_model> \
    --served-model-name <your_model_name> \
    --max-model-len 18000 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-num-seqs 1024 \
    --port 8002
```

### Parameters Explained

| Parameter | Description |
| :--- | :--- |
| `CUDA_VISIBLE_DEVICES` | An environment variable that specifies which GPU device(s) to use. For example: `"0"` or `"0,1"`. |
| `--model` | The local path to your model's weights and configuration files. |
| `--served-model-name` | The name that the model will be identified by in API calls. This is the name you would use in your client application. |
| `--max-model-len` | The maximum sequence length (context window) the model can handle. |
| `--trust-remote-code` | This flag is required to load models that have custom code. |
| `--tensor-parallel-size`| The number of GPUs to use for tensor parallelism. Set to `1` for single-GPU inference. |
| `--max-num-seqs` | The maximum number of sequences to process in a single batch. |
| `--port` | The network port on which the API server will listen for requests. |