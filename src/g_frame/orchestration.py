"""The v2 closed-loop coordinator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from .actions import ActionExecutor
from .decision import DecisionAgent, DecisionArtifact
from .schemas import ActionExecution, EngineTelemetry, SystemState, TaskTelemetry, TrainingRecord
from .team_game import TeamGame
from .telemetry import (
    TaskTelemetryProducer,
    TelemetryMerger,
    TelemetryQueueMerger,
    VLLMMetricsProducer,
)


@dataclass
class CycleResult:
    record: TrainingRecord
    state: SystemState
    decision: DecisionArtifact
    executions: List[ActionExecution]

    def to_dict(self) -> dict:
        return {
            "record": self.record.to_dict(),
            "system_state": self.state.to_dict(),
            "decision": self.decision.to_dict(),
            "executions": [item.to_dict() for item in self.executions],
        }


class GFrameOrchestrator:
    """Coordinates Team Game output with the decision/action feedback loop."""

    def __init__(
        self,
        team_game: TeamGame,
        decision_agent: DecisionAgent,
        action_executor: ActionExecutor,
        telemetry_merger: Optional[TelemetryMerger] = None,
    ) -> None:
        self.team_game = team_game
        self.decision_agent = decision_agent
        self.action_executor = action_executor
        self.telemetry_merger = telemetry_merger or TelemetryMerger()
        self.history: List[DecisionArtifact] = []

    async def run_cycle(
        self,
        task_id: str,
        source_id: str,
        source: str,
        engine: EngineTelemetry,
        tasks: TaskTelemetry,
    ) -> CycleResult:
        record = await self.team_game.run(task_id=task_id, source_id=source_id, source=source)
        state = self.telemetry_merger.ingest_engine(engine)
        state = self.telemetry_merger.ingest_task(tasks) or state
        if state is None:
            raise RuntimeError("engine and task telemetry did not merge")
        decision = await self.decision_agent.decide(state, self.history)
        executions = await self.action_executor.execute_plan_async(decision.plan)
        self.history.append(decision)
        return CycleResult(record=record, state=state, decision=decision, executions=executions)


@dataclass
class ControlCycleResult:
    """One telemetry-aligned decision and action-application cycle."""

    state: SystemState
    decision: DecisionArtifact
    executions: List[ActionExecution]

    def to_dict(self) -> dict:
        return {
            "system_state": self.state.to_dict(),
            "decision": self.decision.to_dict(),
            "executions": [item.to_dict() for item in self.executions],
        }


class AdaptiveRuntimeController:
    """Continuously turn paired telemetry snapshots into applied decision plans.

    Producers stay independent, communicate through separate queues, and are
    aligned by timestep before the decisional agent sees a system state.  This
    lets the controller run against vLLM in production or injected snapshots
    in an offline smoke test.
    """

    def __init__(
        self,
        engine_producer: VLLMMetricsProducer,
        task_producer: TaskTelemetryProducer,
        telemetry_merger: TelemetryQueueMerger,
        decision_agent: DecisionAgent,
        action_executor: ActionExecutor,
    ) -> None:
        self.engine_producer = engine_producer
        self.task_producer = task_producer
        self.telemetry_merger = telemetry_merger
        self.decision_agent = decision_agent
        self.action_executor = action_executor
        self.history: List[DecisionArtifact] = []

    async def run_cycle(self, timestep: int, timeout_s: Optional[float] = None) -> ControlCycleResult:
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        await asyncio.gather(
            self.engine_producer.publish(timestep),
            self.task_producer.publish(timestep),
        )
        state = await self.telemetry_merger.next_state(timeout_s=timeout_s)
        decision = await self.decision_agent.decide(state, self.history)
        executions = await self.action_executor.execute_plan_async(decision.plan)
        self.history.append(decision)
        return ControlCycleResult(state=state, decision=decision, executions=executions)

    async def run_cycles(
        self,
        cycles: int,
        *,
        start_timestep: int = 1,
        timeout_s: Optional[float] = None,
    ) -> List[ControlCycleResult]:
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
            raise ValueError("cycles must be a positive integer")
        if start_timestep < 0:
            raise ValueError("start_timestep must be non-negative")
        return [
            await self.run_cycle(start_timestep + offset, timeout_s=timeout_s)
            for offset in range(cycles)
        ]

    async def run(
        self,
        *,
        start_timestep: int = 1,
        poll_interval_s: float = 0.0,
        max_cycles: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
        timeout_s: Optional[float] = None,
    ) -> List[ControlCycleResult]:
        """Run a bounded or stop-event-controlled persistent telemetry loop."""

        if poll_interval_s < 0:
            raise ValueError("poll_interval_s must be non-negative")
        if max_cycles is not None and (
            isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles < 1
        ):
            raise ValueError("max_cycles must be a positive integer when provided")
        if max_cycles is None and stop_event is None:
            raise ValueError("provide max_cycles or stop_event for a persistent loop")

        results: List[ControlCycleResult] = []
        timestep = start_timestep
        while max_cycles is None or len(results) < max_cycles:
            if stop_event is not None and stop_event.is_set():
                break
            results.append(await self.run_cycle(timestep, timeout_s=timeout_s))
            timestep += 1
            if max_cycles is not None and len(results) >= max_cycles:
                break
            if poll_interval_s > 0:
                if stop_event is None:
                    await asyncio.sleep(poll_interval_s)
                else:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_s)
                    except asyncio.TimeoutError:
                        pass
        return results

    async def aclose(self) -> None:
        await self.telemetry_merger.aclose()
