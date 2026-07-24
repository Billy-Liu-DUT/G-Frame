"""Typed, serializable contracts shared by the G-Frame v2 workflow.

The v1 project exchanged unstructured strings between data generation and
training. These dataclasses make the hand-off explicit without imposing a
runtime dependency on pydantic or a model-serving stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Mapping


class SchemaError(ValueError):
    """Raised when an external agent response does not satisfy a contract."""


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class AgentMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class AgentResult:
    role: str
    raw_response: str
    payload: Dict[str, Any]
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRecord:
    """A reviewed record that is safe to hand to the SFT dataset builder."""

    task_id: str
    source_id: str
    source: str
    question: str
    reasoning: str
    answer: str
    reviewer_feedback: str
    approved: bool
    final_answer: str
    agent_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_chat_messages(self) -> List[Dict[str, str]]:
        assistant_content = self.final_answer or self.answer
        if self.reasoning:
            assistant_content = f"<think>{self.reasoning}</think>\n{assistant_content}"
        return [
            {
                "role": "system",
                "content": "You are a chemistry expert. Answer rigorously and ground claims in the provided source.",
            },
            {
                "role": "user",
                "content": f"Source:\n{self.source}\n\nQuestion:\n{self.question}",
            },
            {"role": "assistant", "content": assistant_content},
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRecord":
        required = (
            "task_id",
            "source_id",
            "source",
            "question",
            "reasoning",
            "answer",
            "reviewer_feedback",
            "final_answer",
        )
        for name in required:
            _require_string(value.get(name), name)
        approved = value.get("approved")
        if not isinstance(approved, bool):
            raise SchemaError("approved must be a boolean")
        trace = value.get("agent_trace", [])
        if not isinstance(trace, list):
            raise SchemaError("agent_trace must be a list")
        return cls(
            task_id=str(value["task_id"]),
            source_id=str(value["source_id"]),
            source=str(value["source"]),
            question=str(value["question"]),
            reasoning=str(value["reasoning"]),
            answer=str(value["answer"]),
            reviewer_feedback=str(value["reviewer_feedback"]),
            approved=approved,
            final_answer=str(value["final_answer"]),
            agent_trace=[dict(item) for item in trace if isinstance(item, Mapping)],
        )


@dataclass(frozen=True)
class EngineTelemetry:
    timestep: int
    kv_cache_usage: float
    pending_requests: int
    active_requests: int = 0
    average_latency_s: float = 0.0
    error_count: int = 0

    def __post_init__(self) -> None:
        if self.timestep < 0:
            raise SchemaError("timestep must be non-negative")
        if not 0.0 <= self.kv_cache_usage <= 1.0:
            raise SchemaError("kv_cache_usage must be between 0 and 1")
        if min(self.pending_requests, self.active_requests, self.error_count) < 0:
            raise SchemaError("request and error counts must be non-negative")


@dataclass(frozen=True)
class TaskTelemetry:
    timestep: int
    completion_rate: float
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int = 0

    def __post_init__(self) -> None:
        if self.timestep < 0:
            raise SchemaError("timestep must be non-negative")
        if not 0.0 <= self.completion_rate <= 1.0:
            raise SchemaError("completion_rate must be between 0 and 1")
        if min(self.queued_tasks, self.completed_tasks, self.failed_tasks) < 0:
            raise SchemaError("task counts must be non-negative")


@dataclass(frozen=True)
class SystemState:
    timestep: int
    engine: EngineTelemetry
    tasks: TaskTelemetry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep": self.timestep,
            "engine": asdict(self.engine),
            "tasks": asdict(self.tasks),
        }


@dataclass(frozen=True)
class PlanAction:
    """A typed action. It deliberately does not contain shell commands."""

    action: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanAction":
        action = _require_string(value.get("action"), "action")
        arguments = value.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise SchemaError("action.arguments must be an object")
        return cls(action=action, arguments=dict(arguments))

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "arguments": self.arguments}


@dataclass(frozen=True)
class DecisionPlan:
    rationale: str
    actions: List[PlanAction]

    @classmethod
    def from_json(cls, raw_response: str) -> "DecisionPlan":
        raw_response = raw_response.strip()
        if raw_response.startswith("```") and raw_response.endswith("```"):
            raw_response = raw_response.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"decision response is not valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SchemaError("decision response must be a JSON object")
        rationale_value = None
        for rationale_key in ("rationale", "rationality", "reasoning", "explanation", "analysis", "reason"):
            if payload.get(rationale_key) is not None:
                rationale_value = payload[rationale_key]
                break
        if rationale_value is None:
            rationale_value = "Model omitted rationale; raw response is retained in the decision artifact."
        rationale = _require_string(rationale_value, "rationale")
        raw_actions = payload.get("plan_of_action", payload.get("Plan_of_Action", payload.get("actions")))
        if not isinstance(raw_actions, list) or not raw_actions:
            raise SchemaError("plan_of_action must be a non-empty list")
        actions = [PlanAction.from_dict(item) for item in raw_actions if isinstance(item, Mapping)]
        if len(actions) != len(raw_actions):
            raise SchemaError("each plan_of_action item must be an object")
        return cls(rationale=rationale, actions=actions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rationale": self.rationale,
            "plan_of_action": [action.to_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class ActionExecution:
    action: PlanAction
    status: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action.to_dict(), "status": self.status, "detail": self.detail}
