export CUDA_VISIBLE_DEVICES="<your_gpu_id>" && \
python -m vllm.entrypoints.openai.api_server \
    --model <path_to_your_model> \
    --served-model-name <your_model_name> \
    --max-model-len 18000 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-num-seqs 1024 \
    --port 8002