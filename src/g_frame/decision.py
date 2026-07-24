"""Decision-agent implementation for the runtime feedback loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

from .actions import ActionExecutor
from .clients import AsyncChatClient
from .prompts import PromptCatalog
from .schemas import AgentMessage, DecisionPlan, SchemaError, SystemState


@dataclass(frozen=True)
class DecisionArtifact:
    state: SystemState
    raw_response: str
    plan: DecisionPlan

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "raw_response": self.raw_response,
            "plan": self.plan.to_dict(),
        }


class DecisionAgent:
    """Uses an LLM response as the source of a validated Plan_of_Action."""

    def __init__(
        self,
        client: AsyncChatClient,
        prompts: PromptCatalog,
        executor: ActionExecutor,
        model: str,
        max_attempts: int = 2,
        prompt_name: str = "plan",
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not prompt_name.strip():
            raise ValueError("prompt_name must be non-empty")
        self.client = client
        self.prompts = prompts
        self.executor = executor
        self.model = model
        self.max_attempts = max_attempts
        self.prompt_name = prompt_name
        self._last_timestep = -1

    async def decide(self, state: SystemState, history: Iterable[DecisionArtifact] = ()) -> DecisionArtifact:
        if state.timestep <= self._last_timestep:
            raise SchemaError(f"stale decision state timestep {state.timestep}")
        previous = [item.plan.to_dict() for item in history]
        prompt = self.prompts.render(
            "decision",
            self.prompt_name,
            system_state=json.dumps(state.to_dict(), ensure_ascii=False),
            runtime_state=json.dumps(self.executor.state.to_dict(), ensure_ascii=False),
            previous_plans=json.dumps(previous, ensure_ascii=False),
        )
        messages = [
            AgentMessage("system", "You are the G-Frame decisional agent. Return only valid JSON."),
            AgentMessage("user", prompt),
        ]
        last_error: SchemaError | None = None
        for attempt in range(1, self.max_attempts + 1):
            raw_response = await self.client.complete(messages, model=self.model, role="decisional")
            try:
                plan = DecisionPlan.from_json(raw_response)
                for action in plan.actions:
                    self.executor.validate(action)
            except SchemaError as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    break
                messages.extend(
                    [
                        AgentMessage("assistant", raw_response),
                        AgentMessage("user", f"Repair the JSON plan. Validation error: {exc}"),
                    ]
                )
                continue
            self._last_timestep = state.timestep
            return DecisionArtifact(state=state, raw_response=raw_response, plan=plan)
        raise last_error or SchemaError("decision agent did not produce a valid plan")
