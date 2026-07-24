"""The v2 Team Game data-synthesis workflow.

This makes the teacher/student/reviewer/rectifier/judger hand-off observable
instead of hiding it inside one unstructured completion.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from .clients import AsyncChatClient
from .prompts import PromptCatalog
from .schemas import AgentMessage, AgentResult, TrainingRecord


_APPROVAL_FIELDS = ("approved", "is_approved", "boolean_approved", "verdict")


def _as_payload(raw_response: str) -> Dict[str, Any]:
    raw_response = raw_response.strip()
    if raw_response.startswith("```") and raw_response.endswith("```"):
        raw_response = raw_response.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(raw_response[start : end + 1])
            except json.JSONDecodeError:
                return {"text": raw_response}
        else:
            return {"text": raw_response}
    if not isinstance(value, Mapping):
        return {"text": raw_response}
    payload = dict(value)
    for wrapper in ("response", "result", "data"):
        nested = payload.get(wrapper)
        if isinstance(nested, Mapping):
            return dict(nested)
    return payload


def _field(payload: Mapping[str, Any], name: str, fallback: str = "") -> str:
    aliases = {
        "question": ("question", "query", "prompt"),
        "reasoning": ("reasoning", "analysis", "explanation"),
        "answer": ("answer", "final_answer", "response", "content", "text"),
        "final_answer": ("final_answer", "answer", "response", "content", "text"),
        "feedback": ("feedback", "review", "comment", "reason"),
    }
    for key in aliases.get(name, (name,)):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _approved(payload: Mapping[str, Any]) -> bool:
    for field in _APPROVAL_FIELDS:
        if field in payload:
            value = payload[field]
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"true", "approved", "approve", "yes", "pass"}
    return False


def _has_approval_signal(payload: Mapping[str, Any]) -> bool:
    return any(field in payload for field in _APPROVAL_FIELDS)


class TeamGame:
    """Runs the concrete QA loop described in the v2 prompt contract."""

    def __init__(self, client: AsyncChatClient, prompts: PromptCatalog, model: str, max_revisions: int = 1) -> None:
        if max_revisions < 0:
            raise ValueError("max_revisions cannot be negative")
        self.client = client
        self.prompts = prompts
        self.model = model
        self.max_revisions = max_revisions

    async def _call(self, role: str, prompt: str, trace: List[AgentResult], attempt: int = 1) -> Dict[str, Any]:
        raw_response = await self.client.complete(
            [
                AgentMessage("system", f"You are the {role} in a chemistry Team Game. Return valid JSON."),
                AgentMessage("user", prompt),
            ],
            model=self.model,
            role=role.lower(),
        )
        payload = _as_payload(raw_response)
        trace.append(AgentResult(role=role, raw_response=raw_response, payload=payload, attempt=attempt))
        return payload

    async def run(self, task_id: str, source_id: str, source: str) -> TrainingRecord:
        if not source.strip():
            raise ValueError("source must be non-empty")
        trace: List[AgentResult] = []
        teacher = await self._call(
            "Teacher",
            self.prompts.render("team_game", "teacher", source=source, source_id=source_id),
            trace,
        )
        question = _field(teacher, "question", _field(teacher, "text"))
        if not question:
            raise ValueError("Teacher did not return a question")
        student = await self._call(
            "Student",
            self.prompts.render("team_game", "student", source=source, question=question),
            trace,
        )
        reasoning = _field(student, "reasoning")
        answer = _field(student, "answer", _field(student, "text"))
        if not answer:
            raise ValueError("Student did not return an answer")
        reviewer = await self._call(
            "Reviewer",
            self.prompts.render(
                "team_game", "reviewer", source=source, question=question, reasoning=reasoning, answer=answer
            ),
            trace,
        )
        approved = _approved(reviewer)
        feedback = _field(reviewer, "feedback", "No reviewer feedback supplied.")

        for revision in range(1, self.max_revisions + 1):
            if approved:
                break
            rectified = await self._call(
                "Rectifier",
                self.prompts.render(
                    "team_game",
                    "rectifier",
                    source=source,
                    question=question,
                    reasoning=reasoning,
                    answer=answer,
                    feedback=feedback,
                ),
                trace,
                attempt=revision,
            )
            reasoning = _field(rectified, "reasoning", reasoning)
            answer = _field(rectified, "answer", _field(rectified, "text", answer))
            reviewer = await self._call(
                "Reviewer",
                self.prompts.render(
                    "team_game", "reviewer", source=source, question=question, reasoning=reasoning, answer=answer
                ),
                trace,
                attempt=revision + 1,
            )
            approved = _approved(reviewer)
            feedback = _field(reviewer, "feedback", feedback)

        judger = await self._call(
            "Judger",
            self.prompts.render(
                "team_game",
                "judger",
                source=source,
                question=question,
                reasoning=reasoning,
                answer=answer,
                feedback=feedback,
                approved=str(approved).lower(),
            ),
            trace,
        )
        final_answer = _field(judger, "final_answer", _field(judger, "answer", answer))
        if not final_answer:
            raise ValueError("Judger did not return a final answer")
        judger_approved = _approved(judger) if _has_approval_signal(judger) else approved
        judger_feedback = _field(judger, "feedback")
        approved = approved and judger_approved
        if judger_feedback:
            feedback = judger_feedback
        return TrainingRecord(
            task_id=task_id,
            source_id=source_id,
            source=source,
            question=question,
            reasoning=reasoning,
            answer=answer,
            reviewer_feedback=feedback,
            approved=approved,
            final_answer=final_answer,
            agent_trace=[item.to_dict() for item in trace],
        )
