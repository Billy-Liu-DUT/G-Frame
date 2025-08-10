import re
import subprocess
import time
import asyncio
from data_models import DataPoint_V


class VLLMMonitor:
    """Manages the monitoring state and data queue for the VLLM server."""

    def __init__(self):
        self.timestep = 0
        self.record_queue = asyncio.Queue()


class VLLMServer:
    """Handles the lifecycle and log monitoring of the vLLM subprocess."""

    def __init__(self, config: dict):
        self.config = config
        self.host = config['host']
        self.port = config['port']
        self.is_server_ready = False
        self.start_time = 0.0
        self.process: subprocess.Popen = None
        self.monitor = VLLMMonitor()
        self.vllm_metric_queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        self._start_server()

    def _start_server(self):
        self.start_time = time.time()
        command = [
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--host', self.host,
            '--port', str(self.port),
            '--model', self.config['model_path'],
            '--served-model-name', self.config['model_name'],
            '--tensor-parallel-size', str(self.config['num_gpus']),
            '--max-num-seqs', str(self.config['max_num_seqs']),
            '--max-num-batched-tokens', str(self.config['max_num_batched_tokens']),
            '--max-model-len', str(self.config['max_num_batched_tokens'])
        ]
        print("[VLLM] Starting VLLM server...")
        print(f"[VLLM] Command: {' '.join(command)}")
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        self.loop.run_in_executor(None, self._blocking_stdio_reader)
        asyncio.create_task(self._metric_processor())

    async def shutdown(self):
        if self.process:
            print("[VLLM] Shutting down VLLM server...")
            self.process.terminate()
            try:
                await self.loop.run_in_executor(None, self.process.wait, 10)
            except Exception:
                self.process.kill()
            print("[VLLM] Server process terminated.")

    def force_shutdown(self):
        if self.process:
            print("[VLLM] Forcing shutdown of VLLM server...")
            self.process.kill()

    async def wait_for_server_ready(self):
        print("[VLLM] Waiting for server to become ready...")
        start = time.time()
        while not self.is_server_ready:
            await asyncio.sleep(1)
        print(f"[VLLM] Server ready in {time.time() - start:.2f}s.")

    async def _metric_processor(self, interval: float = 5.0):
        print("[VLLM] Metric processor started.")
        while True:
            await asyncio.sleep(interval)
            if self.is_server_ready:
                try:
                    stats = self.vllm_metric_queue.get_nowait()
                    stats.timestep = self.monitor.timestep
                    await self.monitor.record_queue.put(stats)
                except asyncio.QueueEmpty:
                    pass
            self.monitor.timestep += interval

    def _parse_vllm_log_line(self, line: str) -> DataPoint_V | None:
        pattern = r"Running: (\d+), Swapped: \d+, Pending: (\d+), GPU KV cache usage: ([\d.]+)%"
        match = re.search(pattern, line)
        if match:
            running = int(match.group(1))
            pending = int(match.group(2))
            kv_cache = float(match.group(3)) / 100.0

            avg_input, avg_output = 0.0, 0.0

            return DataPoint_V(
                timestamp_v=time.time(),
                kv_cache_usage=kv_cache,
                running_requests=running,
                pending_requests=pending,
                avg_input_tokens=avg_input,
                avg_output_tokens=avg_output,
            )
        return None

    def _blocking_stdio_reader(self):
        if not self.process or not self.process.stdout:
            return

        for line in iter(self.process.stdout.readline, ''):
            print(f"[VLLM_LOG] {line.strip()}")
            if "Uvicorn running on" in line:
                self.is_server_ready = True

            if stats := self._parse_vllm_log_line(line):
                self.loop.call_soon_threadsafe(self.vllm_metric_queue.put_nowait, stats)

        self.process.stdout.close()