from dataclasses import dataclass
from typing import Optional

@dataclass
class DataPoint_V:
    """Represents a single data point from the VLLM server monitor."""
    timestamp_v: float
    kv_cache_usage: float
    pending_requests: int
    running_requests: int
    avg_input_tokens: float
    avg_output_tokens: float
    timestep: Optional[int] = None

@dataclass
class DataPoint_T:
    """Represents a single data point from the Task Executor monitor."""
    timestamp_t: float
    timestep: int
    completion_rate: float