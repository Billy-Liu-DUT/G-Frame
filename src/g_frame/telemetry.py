"""Telemetry primitives evolved from v1 DataPoint_V/DataPoint_T."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import math
import re
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Union
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .schemas import EngineTelemetry, SystemState, TaskTelemetry


class DynamicSemaphore:
    """An asyncio semaphore whose limit can be safely changed at runtime."""

    def __init__(self, initial_limit: int) -> None:
        if initial_limit < 1:
            raise ValueError("initial_limit must be at least 1")
        self._limit = initial_limit
        self._running_tasks = 0
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def running_tasks(self) -> int:
        return self._running_tasks

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._running_tasks < self._limit)
            self._running_tasks += 1

    async def release(self) -> None:
        async with self._condition:
            self._running_tasks -= 1
            if self._running_tasks < 0:
                self._running_tasks = 0
                raise RuntimeError("semaphore released more times than acquired")
            self._condition.notify_all()

    async def set_limit(self, new_limit: int) -> None:
        if new_limit < 1:
            raise ValueError("new_limit must be at least 1")
        async with self._condition:
            self._limit = new_limit
            self._condition.notify_all()

    async def __aenter__(self) -> "DynamicSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.release()


class TelemetryMerger:
    """Matches engine and task telemetry by timestep before decision making."""

    def __init__(self) -> None:
        self._engines: Dict[int, EngineTelemetry] = {}
        self._tasks: Dict[int, TaskTelemetry] = {}

    def ingest_engine(self, value: EngineTelemetry) -> Optional[SystemState]:
        self._engines[value.timestep] = value
        return self._try_merge(value.timestep)

    def ingest_task(self, value: TaskTelemetry) -> Optional[SystemState]:
        self._tasks[value.timestep] = value
        return self._try_merge(value.timestep)

    def _try_merge(self, timestep: int) -> Optional[SystemState]:
        engine = self._engines.get(timestep)
        task = self._tasks.get(timestep)
        if engine is None or task is None:
            return None
        del self._engines[timestep]
        del self._tasks[timestep]
        return SystemState(timestep=timestep, engine=engine, tasks=task)


def engine_telemetry_from_v1(value: object) -> EngineTelemetry:
    """Adapt a v1 ``DataPoint_V`` without importing legacy modules."""
    timestep = getattr(value, "timestep", None)
    if not isinstance(timestep, int):
        raise ValueError("v1 DataPoint_V must have an integer timestep")
    return EngineTelemetry(
        timestep=timestep,
        kv_cache_usage=float(getattr(value, "kv_cache_usage")),
        pending_requests=int(getattr(value, "pending_requests")),
        active_requests=int(getattr(value, "running_requests", 0)),
    )


def task_telemetry_from_v1(value: object, total_tasks: int, failed_tasks: int = 0) -> TaskTelemetry:
    """Adapt a v1 ``DataPoint_T`` while keeping the original fields usable."""
    if total_tasks < 0 or failed_tasks < 0:
        raise ValueError("task counts must be non-negative")
    timestep = getattr(value, "timestep", None)
    if not isinstance(timestep, int):
        raise ValueError("v1 DataPoint_T must have an integer timestep")
    completion_rate = float(getattr(value, "completion_rate"))
    completed_tasks = round(completion_rate * total_tasks)
    return TaskTelemetry(
        timestep=timestep,
        completion_rate=completion_rate,
        queued_tasks=max(total_tasks - completed_tasks, 0),
        completed_tasks=completed_tasks,
        failed_tasks=failed_tasks,
    )


_PROMETHEUS_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[^\s]+)"
)


def metrics_url_from_openai_base_url(base_url: str) -> str:
    """Derive a vLLM ``/metrics`` URL from an OpenAI-compatible base URL."""

    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    return urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))


def parse_prometheus_metrics(payload: str) -> Dict[str, float]:
    """Parse the numeric samples needed from a Prometheus text payload.

    Labels are intentionally discarded because vLLM exposes one aggregate
    value for the service in the supported smoke configuration.  The parser is
    kept dependency-free so it can be exercised in a CPU-only test suite.
    """

    values: Dict[str, float] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_SAMPLE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        values[match.group("name")] = value
    return values


def _normalise_metric_name(name: str) -> str:
    return name.lower().replace(":", "_")


def _metric_value(metrics: Mapping[str, float], *aliases: str) -> Optional[float]:
    normalised_aliases = {_normalise_metric_name(alias) for alias in aliases}
    for alias in aliases:
        if alias in metrics:
            return metrics[alias]
    for name, value in metrics.items():
        if _normalise_metric_name(name) in normalised_aliases:
            return value
    return None


def _as_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(number)


def _as_fraction(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number between 0 and 1")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number between 0 and 1") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a number between 0 and 1")
    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be a number between 0 and 1")
    return number


async def _fetch_metrics_text(url: str, timeout_s: float) -> str:
    """Fetch Prometheus text without importing an HTTP client dependency."""

    def fetch() -> str:
        request = Request(url, headers={"Accept": "text/plain"})
        with urlopen(request, timeout=timeout_s) as response:  # nosec B310 - caller owns the endpoint
            return response.read().decode("utf-8")

    return await asyncio.to_thread(fetch)


EngineSnapshot = Union[EngineTelemetry, Mapping[str, object]]
TaskSnapshotValue = Union["TaskSnapshot", TaskTelemetry, Mapping[str, object]]
EngineSnapshotProvider = Callable[[], Union[EngineSnapshot, Awaitable[EngineSnapshot]]]
TaskSnapshotProvider = Callable[[], Union[TaskSnapshotValue, Awaitable[TaskSnapshotValue]]]
MetricsFetcher = Callable[[str], Union[str, Awaitable[str]]]


async def _resolve_provider(provider: Callable[[], object]) -> object:
    value = provider()
    if inspect.isawaitable(value):
        return await value
    return value


def _mapping_value(value: Mapping[str, object], names: tuple[str, ...]) -> Optional[object]:
    for name in names:
        if name in value:
            return value[name]
    return None


def _engine_from_snapshot(snapshot: EngineSnapshot, timestep: int) -> EngineTelemetry:
    if isinstance(snapshot, EngineTelemetry):
        return EngineTelemetry(
            timestep=timestep,
            kv_cache_usage=snapshot.kv_cache_usage,
            pending_requests=snapshot.pending_requests,
            active_requests=snapshot.active_requests,
            average_latency_s=snapshot.average_latency_s,
            error_count=snapshot.error_count,
        )
    if not isinstance(snapshot, Mapping):
        raise TypeError("engine snapshot must be EngineTelemetry or a mapping")
    kv_cache_usage = _mapping_value(snapshot, ("kv_cache_usage", "gpu_cache_usage_perc"))
    pending_requests = _mapping_value(snapshot, ("pending_requests", "num_requests_waiting"))
    if kv_cache_usage is None or pending_requests is None:
        raise ValueError("engine snapshot requires kv_cache_usage and pending_requests")
    active_requests = _mapping_value(snapshot, ("active_requests", "running_requests", "num_requests_running"))
    average_latency_s = _mapping_value(snapshot, ("average_latency_s", "avg_latency_s"))
    error_count = _mapping_value(snapshot, ("error_count", "failed_requests"))
    return EngineTelemetry(
        timestep=timestep,
        kv_cache_usage=_as_fraction(kv_cache_usage, "kv_cache_usage"),
        pending_requests=_as_non_negative_int(pending_requests, "pending_requests"),
        active_requests=_as_non_negative_int(active_requests or 0, "active_requests"),
        average_latency_s=float(average_latency_s or 0.0),
        error_count=_as_non_negative_int(error_count or 0, "error_count"),
    )


@dataclass(frozen=True)
class TaskSnapshot:
    """A task-layer snapshot before it is assigned a telemetry timestep."""

    queued_tasks: int
    completed_tasks: int
    failed_tasks: int = 0
    running_tasks: int = 0
    total_tasks: Optional[int] = None

    def __post_init__(self) -> None:
        if min(self.queued_tasks, self.completed_tasks, self.failed_tasks, self.running_tasks) < 0:
            raise ValueError("task snapshot counts must be non-negative")
        if self.total_tasks is not None and self.total_tasks < 0:
            raise ValueError("task snapshot total_tasks must be non-negative")

    @property
    def completion_rate(self) -> float:
        total = self.total_tasks
        if total is None:
            total = self.queued_tasks + self.running_tasks + self.completed_tasks + self.failed_tasks
        if total == 0:
            return 1.0
        return (self.completed_tasks + self.failed_tasks) / total


def _task_from_snapshot(snapshot: TaskSnapshotValue, timestep: int) -> TaskTelemetry:
    if isinstance(snapshot, TaskTelemetry):
        return TaskTelemetry(
            timestep=timestep,
            completion_rate=snapshot.completion_rate,
            queued_tasks=snapshot.queued_tasks,
            completed_tasks=snapshot.completed_tasks,
            failed_tasks=snapshot.failed_tasks,
        )
    if isinstance(snapshot, TaskSnapshot):
        return TaskTelemetry(
            timestep=timestep,
            completion_rate=snapshot.completion_rate,
            queued_tasks=snapshot.queued_tasks,
            completed_tasks=snapshot.completed_tasks,
            failed_tasks=snapshot.failed_tasks,
        )
    if not isinstance(snapshot, Mapping):
        raise TypeError("task snapshot must be TaskSnapshot, TaskTelemetry, or a mapping")
    queued_tasks = _as_non_negative_int(snapshot.get("queued_tasks", 0), "queued_tasks")
    completed_tasks = _as_non_negative_int(snapshot.get("completed_tasks", 0), "completed_tasks")
    failed_tasks = _as_non_negative_int(snapshot.get("failed_tasks", 0), "failed_tasks")
    completion_rate = snapshot.get("completion_rate")
    if completion_rate is None:
        running_tasks = _as_non_negative_int(snapshot.get("running_tasks", 0), "running_tasks")
        total_tasks = snapshot.get("total_tasks")
        total = (
            _as_non_negative_int(total_tasks, "total_tasks")
            if total_tasks is not None
            else queued_tasks + running_tasks + completed_tasks + failed_tasks
        )
        completion_rate = 1.0 if total == 0 else (completed_tasks + failed_tasks) / total
    return TaskTelemetry(
        timestep=timestep,
        completion_rate=_as_fraction(completion_rate, "completion_rate"),
        queued_tasks=queued_tasks,
        completed_tasks=completed_tasks,
        failed_tasks=failed_tasks,
    )


class VLLMMetricsProducer:
    """Publish engine telemetry from vLLM ``/metrics`` or an injected snapshot."""

    def __init__(
        self,
        queue: "asyncio.Queue[EngineTelemetry]",
        *,
        base_url: Optional[str] = None,
        metrics_url: Optional[str] = None,
        snapshot_provider: Optional[EngineSnapshotProvider] = None,
        fetcher: Optional[MetricsFetcher] = None,
        timeout_s: float = 3.0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if snapshot_provider is None and metrics_url is None and base_url is None:
            raise ValueError("provide base_url, metrics_url, or snapshot_provider")
        self.queue = queue
        self.metrics_url = metrics_url or (metrics_url_from_openai_base_url(base_url) if base_url else None)
        self.snapshot_provider = snapshot_provider
        self.timeout_s = timeout_s
        self.fetcher = fetcher or (lambda url: _fetch_metrics_text(url, timeout_s))

    async def _snapshot(self) -> EngineSnapshot:
        if self.snapshot_provider is not None:
            return _engine_snapshot_value(await _resolve_provider(self.snapshot_provider))
        if self.metrics_url is None:
            raise RuntimeError("metrics URL is not configured")
        raw_metrics = self.fetcher(self.metrics_url)
        if inspect.isawaitable(raw_metrics):
            raw_metrics = await raw_metrics
        if not isinstance(raw_metrics, str):
            raise TypeError("metrics fetcher must return Prometheus text")
        metrics = parse_prometheus_metrics(raw_metrics)
        return self._snapshot_from_metrics(metrics)

    @staticmethod
    def _snapshot_from_metrics(metrics: Mapping[str, float]) -> Mapping[str, object]:
        kv_cache_usage = _metric_value(
            metrics,
            "vllm:gpu_cache_usage_perc",
            "vllm_gpu_cache_usage_perc",
            "gpu_cache_usage_perc",
            "kv_cache_usage",
        )
        pending_requests = _metric_value(
            metrics,
            "vllm:num_requests_waiting",
            "vllm_num_requests_waiting",
            "num_requests_waiting",
            "pending_requests",
        )
        active_requests = _metric_value(
            metrics,
            "vllm:num_requests_running",
            "vllm_num_requests_running",
            "num_requests_running",
            "active_requests",
        )
        if kv_cache_usage is None or pending_requests is None:
            raise ValueError(
                "vLLM /metrics is missing gpu cache usage or waiting-request gauges"
            )
        latency_sum = _metric_value(
            metrics,
            "vllm:request_latency_seconds_sum",
            "vllm_request_latency_seconds_sum",
        )
        latency_count = _metric_value(
            metrics,
            "vllm:request_latency_seconds_count",
            "vllm_request_latency_seconds_count",
        )
        average_latency_s = 0.0
        if latency_sum is not None and latency_count is not None and latency_count > 0:
            average_latency_s = latency_sum / latency_count
        error_count = _metric_value(
            metrics,
            "vllm:request_failure_total",
            "vllm_request_failure_total",
            "request_failure_total",
            "error_count",
        )
        return {
            "kv_cache_usage": kv_cache_usage,
            "pending_requests": pending_requests,
            "active_requests": active_requests or 0,
            "average_latency_s": average_latency_s,
            "error_count": error_count or 0,
        }

    async def publish(self, timestep: int) -> EngineTelemetry:
        telemetry = _engine_from_snapshot(await self._snapshot(), timestep)
        await self.queue.put(telemetry)
        return telemetry


def _engine_snapshot_value(value: object) -> EngineSnapshot:
    if isinstance(value, EngineTelemetry) or isinstance(value, Mapping):
        return value
    raise TypeError("engine snapshot provider must return EngineTelemetry or a mapping")


class TaskTelemetryTracker:
    """Small task-layer counter source for persistent telemetry producers."""

    def __init__(self) -> None:
        self._queued = 0
        self._running = 0
        self._completed = 0
        self._failed = 0
        self._total = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _validate_count(count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")

    async def enqueue(self, count: int = 1) -> None:
        self._validate_count(count)
        async with self._lock:
            self._queued += count
            self._total += count

    async def start(self, count: int = 1) -> None:
        self._validate_count(count)
        async with self._lock:
            if count > self._queued:
                raise RuntimeError("cannot start more tasks than are queued")
            self._queued -= count
            self._running += count

    async def complete(self, count: int = 1) -> None:
        self._validate_count(count)
        async with self._lock:
            if count > self._running:
                raise RuntimeError("cannot complete more tasks than are running")
            self._running -= count
            self._completed += count

    async def fail(self, count: int = 1) -> None:
        self._validate_count(count)
        async with self._lock:
            if count > self._running:
                raise RuntimeError("cannot fail more tasks than are running")
            self._running -= count
            self._failed += count

    async def snapshot(self) -> TaskSnapshot:
        async with self._lock:
            return TaskSnapshot(
                queued_tasks=self._queued,
                running_tasks=self._running,
                completed_tasks=self._completed,
                failed_tasks=self._failed,
                total_tasks=self._total,
            )


class TaskTelemetryProducer:
    """Publish task-layer telemetry from an asynchronous or synchronous snapshot source."""

    def __init__(self, queue: "asyncio.Queue[TaskTelemetry]", snapshot_provider: TaskSnapshotProvider) -> None:
        self.queue = queue
        self.snapshot_provider = snapshot_provider

    async def publish(self, timestep: int) -> TaskTelemetry:
        snapshot = await _resolve_provider(self.snapshot_provider)
        telemetry = _task_from_snapshot(snapshot, timestep)
        await self.queue.put(telemetry)
        return telemetry


class TelemetryQueueMerger:
    """Consume independent telemetry queues and emit only aligned system states."""

    def __init__(
        self,
        engine_queue: "asyncio.Queue[EngineTelemetry]",
        task_queue: "asyncio.Queue[TaskTelemetry]",
        merger: Optional[TelemetryMerger] = None,
    ) -> None:
        self.engine_queue = engine_queue
        self.task_queue = task_queue
        self.merger = merger or TelemetryMerger()
        self._engine_get: Optional["asyncio.Task[EngineTelemetry]"] = None
        self._task_get: Optional["asyncio.Task[TaskTelemetry]"] = None
        self._ready_states: list[SystemState] = []

    def _ensure_waiters(self) -> None:
        if self._engine_get is None:
            self._engine_get = asyncio.create_task(self.engine_queue.get())
        if self._task_get is None:
            self._task_get = asyncio.create_task(self.task_queue.get())

    def _ingest_engine(self, value: EngineTelemetry) -> None:
        state = self.merger.ingest_engine(value)
        if state is not None:
            self._ready_states.append(state)

    def _ingest_task(self, value: TaskTelemetry) -> None:
        state = self.merger.ingest_task(value)
        if state is not None:
            self._ready_states.append(state)

    async def next_state(self, timeout_s: Optional[float] = None) -> SystemState:
        """Wait until one matching engine/task timestep is available."""

        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive when provided")
        if self._ready_states:
            return self._ready_states.pop(0)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s if timeout_s is not None else None
        while True:
            self._ensure_waiters()
            engine_waiter = self._engine_get
            task_waiter = self._task_get
            if engine_waiter is None or task_waiter is None:
                raise RuntimeError("telemetry waiters were not initialized")
            remaining = None if deadline is None else max(deadline - loop.time(), 0.0)
            done, _ = await asyncio.wait(
                (engine_waiter, task_waiter),
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError("timed out waiting for aligned telemetry")
            if engine_waiter in done:
                self._engine_get = None
                self._ingest_engine(engine_waiter.result())
                self.engine_queue.task_done()
            if task_waiter in done:
                self._task_get = None
                self._ingest_task(task_waiter.result())
                self.task_queue.task_done()
            if self._ready_states:
                return self._ready_states.pop(0)

    async def aclose(self) -> None:
        """Cancel outstanding queue waits when a long-running controller stops."""

        pending = [task for task in (self._engine_get, self._task_get) if task is not None]
        self._engine_get = None
        self._task_get = None
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
