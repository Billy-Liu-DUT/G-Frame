"""Safe execution of typed decision actions."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .schemas import ActionExecution, DecisionPlan, PlanAction, SchemaError


STAGE_ACTIONS = {
    "data_clean",
    "data_augment",
    "cot_distill",
    "cot_synth",
    "qa_synth",
    "model_train",
    "system_health_check",
}


@dataclass
class RuntimeState:
    concurrency: int = 1
    max_concurrency: int = 400
    learning_rate: float = 2e-5
    data_mix: Dict[str, float] = field(default_factory=lambda: {"domain": 1.0})
    scheduled_stages: List[str] = field(default_factory=list)
    completed_idempotency_keys: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int):
            raise ValueError("max_concurrency must be a positive integer")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int):
            raise ValueError("concurrency must be an integer from 1 to max_concurrency")
        if not 1 <= self.concurrency <= self.max_concurrency:
            raise ValueError("concurrency must be an integer from 1 to max_concurrency")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "max_concurrency": self.max_concurrency,
            "learning_rate": self.learning_rate,
            "data_mix": dict(self.data_mix),
            "scheduled_stages": list(self.scheduled_stages),
            "completed_idempotency_keys": list(self.completed_idempotency_keys),
        }


StageHandler = Callable[[PlanAction], object]
ConcurrencyHandler = Callable[[int], Awaitable[None]]


class ActionExecutor:
    """Validates and applies a Plan_of_Action without shell execution."""

    def __init__(self, state: Optional[RuntimeState] = None) -> None:
        self.state = state or RuntimeState()
        self._concurrency_handler: Optional[ConcurrencyHandler] = None
        self._stage_handlers: Dict[str, StageHandler] = {}

    def bind_concurrency_handler(self, handler: ConcurrencyHandler) -> None:
        """Bind the live worker limiter controlled by ``set_concurrency``."""
        self._concurrency_handler = handler

    def bind_stage_handler(self, action: str, handler: StageHandler) -> None:
        if action not in STAGE_ACTIONS:
            raise ValueError(f"{action} is not a pipeline stage")
        self._stage_handlers[action] = handler

    @staticmethod
    def _idempotency_key(action: PlanAction) -> Optional[str]:
        value = action.arguments.get("idempotency_key")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise SchemaError("idempotency_key must be a non-empty string when provided")
        return f"{action.action}:{value.strip()}"

    def validate(self, action: PlanAction) -> None:
        if action.action in STAGE_ACTIONS:
            if any(name in action.arguments for name in ("command", "shell", "executable")):
                raise SchemaError("pipeline actions cannot contain shell execution fields")
            self._idempotency_key(action)
            return
        if action.action == "set_concurrency":
            value = action.arguments.get("value")
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= self.state.max_concurrency
            ):
                raise SchemaError(
                    "set_concurrency.value must be an integer from 1 to "
                    f"{self.state.max_concurrency}"
                )
            return
        if action.action == "set_learning_rate":
            value = action.arguments.get("value")
            if not isinstance(value, (int, float)) or not 1e-8 <= float(value) <= 1.0:
                raise SchemaError("set_learning_rate.value must be between 1e-8 and 1")
            return
        if action.action == "set_data_mix":
            value = action.arguments.get("value")
            if not isinstance(value, dict) or not value:
                raise SchemaError("set_data_mix.value must be a non-empty object")
            if any(not isinstance(weight, (int, float)) or weight < 0 for weight in value.values()):
                raise SchemaError("set_data_mix weights must be non-negative numbers")
            if sum(float(weight) for weight in value.values()) <= 0:
                raise SchemaError("set_data_mix weights must sum to a positive value")
            return
        raise SchemaError(f"unsupported action: {action.action}")

    def _apply_state(self, action: PlanAction) -> ActionExecution:
        key = self._idempotency_key(action)
        if key and key in self.state.completed_idempotency_keys:
            return ActionExecution(action, "unchanged", f"idempotency key {key} was already applied")
        self.validate(action)
        if action.action in STAGE_ACTIONS:
            self.state.scheduled_stages.append(action.action)
            result = ActionExecution(action, "scheduled", f"scheduled stage {action.action}")
        elif action.action == "set_concurrency":
            self.state.concurrency = int(action.arguments["value"])
            result = ActionExecution(action, "applied", f"concurrency={self.state.concurrency}")
        elif action.action == "set_learning_rate":
            self.state.learning_rate = float(action.arguments["value"])
            result = ActionExecution(action, "applied", f"learning_rate={self.state.learning_rate}")
        else:
            weights = {name: float(value) for name, value in action.arguments["value"].items()}
            total = sum(weights.values())
            self.state.data_mix = {name: value / total for name, value in weights.items()}
            result = ActionExecution(action, "applied", "data_mix normalized")
        if key:
            self.state.completed_idempotency_keys.append(key)
        return result

    def execute(self, action: PlanAction) -> ActionExecution:
        """Apply a plan in a synchronous planning or test context.

        Live workers should use :meth:`execute_async` so a concurrency decision
        is committed to the actual semaphore before state is updated.
        """
        return self._apply_state(action)

    async def execute_async(self, action: PlanAction) -> ActionExecution:
        self.validate(action)
        key = self._idempotency_key(action)
        if key and key in self.state.completed_idempotency_keys:
            return ActionExecution(action, "unchanged", f"idempotency key {key} was already applied")
        if action.action == "set_concurrency" and self._concurrency_handler is not None:
            await self._concurrency_handler(int(action.arguments["value"]))
        stage_handler = self._stage_handlers.get(action.action)
        if stage_handler is not None:
            outcome = stage_handler(action)
            if inspect.isawaitable(outcome):
                await outcome
            self._apply_state(action)
            return ActionExecution(action, "executed", f"executed stage {action.action}")
        return self._apply_state(action)

    def execute_plan(self, plan: DecisionPlan) -> List[ActionExecution]:
        return [self.execute(action) for action in plan.actions]

    async def execute_plan_async(self, plan: DecisionPlan) -> List[ActionExecution]:
        return [await self.execute_async(action) for action in plan.actions]
