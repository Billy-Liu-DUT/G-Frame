"""v1-compatible vLLM process wrapper with bounded readiness checks."""

import asyncio
from collections import deque
import re
import subprocess
import time
from typing import Optional

from data_models import DataPoint_V


class VLLMMonitor:
    """Manages monitor state and the v1-compatible telemetry queue."""

    def __init__(self):
        self.timestep = 0
        self.record_queue = asyncio.Queue()


class VLLMServer:
    """Handles a vLLM subprocess without an unbounded startup wait."""

    def __init__(self, config: dict):
        self.config = config
        self.host = config["host"]
        self.port = config["port"]
        self.startup_timeout_s = float(config.get("startup_timeout_s", 300))
        self.is_server_ready = False
        self.start_time = 0.0
        self.process: Optional[subprocess.Popen] = None
        self.monitor = VLLMMonitor()
        self.vllm_metric_queue = asyncio.Queue()
        self._recent_logs = deque(maxlen=30)
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError("VLLMServer must be created from an active asyncio event loop") from exc
        self._start_server()

    def _start_server(self):
        self.start_time = time.monotonic()
        command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--host", self.host,
            "--port", str(self.port),
            "--model", self.config["model_path"],
            "--served-model-name", self.config["model_name"],
            "--tensor-parallel-size", str(self.config["num_gpus"]),
            "--max-num-seqs", str(self.config["max_num_seqs"]),
            "--max-num-batched-tokens", str(self.config["max_num_batched_tokens"]),
            "--max-model-len", str(self.config["max_num_batched_tokens"]),
        ]
        print("[VLLM] Starting vLLM server.")
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.loop.run_in_executor(None, self._blocking_stdio_reader)
        asyncio.create_task(self._metric_processor())

    async def shutdown(self):
        if not self.process or self.process.poll() is not None:
            return
        print("[VLLM] Shutting down vLLM server.")
        self.process.terminate()
        try:
            await self.loop.run_in_executor(None, self.process.wait, 10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            await self.loop.run_in_executor(None, self.process.wait)
        print("[VLLM] Server process terminated.")

    def force_shutdown(self):
        if self.process and self.process.poll() is None:
            print("[VLLM] Forcing vLLM server shutdown.")
            self.process.kill()

    async def wait_for_server_ready(self):
        """Wait until vLLM is ready, exits, or reaches the configured deadline."""
        print("[VLLM] Waiting for server readiness.")
        deadline = time.monotonic() + self.startup_timeout_s
        while not self.is_server_ready:
            if not self.process:
                raise RuntimeError("vLLM subprocess was not created")
            return_code = self.process.poll()
            if return_code is not None:
                logs = "".join(self._recent_logs).strip()
                raise RuntimeError(f"vLLM exited before readiness with code {return_code}. Recent logs: {logs}")
            if time.monotonic() >= deadline:
                await self.shutdown()
                raise TimeoutError(f"vLLM did not become ready within {self.startup_timeout_s:g} seconds")
            await asyncio.sleep(0.5)
        print(f"[VLLM] Server ready in {time.monotonic() - self.start_time:.2f}s.")

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

    def _parse_vllm_log_line(self, line: str) -> Optional[DataPoint_V]:
        pattern = r"Running: (\d+), Swapped: \d+, Pending: (\d+), GPU KV cache usage: ([\d.]+)%"
        match = re.search(pattern, line)
        if not match:
            return None
        return DataPoint_V(
            timestamp_v=time.time(),
            kv_cache_usage=float(match.group(3)) / 100.0,
            running_requests=int(match.group(1)),
            pending_requests=int(match.group(2)),
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
        )

    def _blocking_stdio_reader(self):
        if not self.process or not self.process.stdout:
            return
        for line in iter(self.process.stdout.readline, ""):
            self._recent_logs.append(line)
            print(f"[VLLM_LOG] {line.strip()}")
            if "Uvicorn running on" in line:
                self.is_server_ready = True
            stats = self._parse_vllm_log_line(line)
            if stats is not None:
                self.loop.call_soon_threadsafe(self.vllm_metric_queue.put_nowait, stats)
        self.process.stdout.close()
