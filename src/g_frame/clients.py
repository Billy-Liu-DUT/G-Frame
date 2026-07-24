"""Model client abstractions.

Only the OpenAI-compatible adapter imports ``openai``, and only when a real
request is made. This keeps all CPU-only tests independent of service SDKs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Union

from .schemas import AgentMessage
from .telemetry import DynamicSemaphore


class AsyncChatClient(Protocol):
    async def complete(self, messages: Sequence[AgentMessage], model: str, role: str) -> str:
        ...


class ScriptedChatClient:
    """Deterministic client used by offline smoke tests and unit tests."""

    def __init__(self, responses: Mapping[str, Union[Sequence[str], str]]) -> None:
        self._responses: Dict[str, List[str]] = {}
        for role, value in responses.items():
            self._responses[role] = [value] if isinstance(value, str) else list(value)
        self._positions = defaultdict(int)
        self.requests: List[Dict[str, object]] = []

    async def complete(self, messages: Sequence[AgentMessage], model: str, role: str) -> str:
        self.requests.append({"role": role, "model": model, "messages": [item.to_dict() for item in messages]})
        if role not in self._responses:
            raise KeyError(f"no scripted response for role {role}")
        position = self._positions[role]
        responses = self._responses[role]
        if position >= len(responses):
            position = len(responses) - 1
        self._positions[role] += 1
        return responses[position]


class LimiterBoundChatClient:
    """Apply a live :class:`DynamicSemaphore` to every completion.

    The wrapper is deliberately model-agnostic.  Team Game executive agents
    receive this client, so a runtime ``set_concurrency`` action affects their
    next request without coupling the workflow to a particular SDK or server.
    The decisional agent may use a separate client and limiter when it is
    hosted independently.
    """

    def __init__(self, client: AsyncChatClient, limiter: DynamicSemaphore) -> None:
        self.client = client
        self.limiter = limiter

    async def complete(self, messages: Sequence[AgentMessage], model: str, role: str) -> str:
        async with self.limiter:
            return await self.client.complete(messages=messages, model=model, role=role)


class OpenAICompatibleClient:
    """Lazy adapter for vLLM and other OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        temperature: float = 0.1,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self._client: Optional[object] = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("install the runtime extra to use an OpenAI-compatible client") from exc
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    async def complete(self, messages: Sequence[AgentMessage], model: str, role: str) -> str:
        client = self._get_client()
        request: Dict[str, object] = {
            "model": model,
            "temperature": self.temperature,
            "messages": [item.to_dict() for item in messages],
        }
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        response = await client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"{role} returned an empty completion")
        return content
