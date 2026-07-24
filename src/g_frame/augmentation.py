"""Versioned six-style source augmentation for the v2 data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Mapping, Sequence

from .clients import AsyncChatClient
from .prompts import PromptCatalog
from .schemas import AgentMessage


AUGMENTATION_STYLES = (
    "research_summary",
    "graduate_tutorial",
    "industrial_practice",
    "mechanistic_explanation",
    "question_led_review",
    "plain_language_bridge",
)


@dataclass(frozen=True)
class AugmentedSource:
    source_id: str
    style: str
    source: str
    augmented_text: str
    raw_response: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _extract_text(raw_response: str) -> str:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        payload = raw_response
    if isinstance(payload, Mapping):
        value = payload.get("augmented_text", payload.get("text"))
    else:
        value = payload
    if not isinstance(value, str) or not value.strip():
        raise ValueError("augmentation response must contain non-empty augmented_text")
    return value.strip()


class SixStyleAugmenter:
    """Run the six editable manuscript-derived augmentation prompts."""

    def __init__(self, client: AsyncChatClient, prompts: PromptCatalog, model: str) -> None:
        self.client = client
        self.prompts = prompts
        self.model = model

    async def augment(
        self, source_id: str, source: str, styles: Sequence[str] = AUGMENTATION_STYLES
    ) -> List[AugmentedSource]:
        if not source_id.strip() or not source.strip():
            raise ValueError("source_id and source must be non-empty")
        selected = tuple(styles)
        if not selected:
            raise ValueError("at least one augmentation style is required")
        unknown = set(selected) - set(AUGMENTATION_STYLES)
        if unknown:
            raise ValueError(f"unknown augmentation styles: {', '.join(sorted(unknown))}")
        results: List[AugmentedSource] = []
        for style in selected:
            prompt = self.prompts.render("augmentation", style, source_id=source_id, source=source)
            raw_response = await self.client.complete(
                [
                    AgentMessage("system", "You are a chemistry data augmentation agent. Return valid JSON."),
                    AgentMessage("user", prompt),
                ],
                model=self.model,
                role=f"augment_{style}",
            )
            results.append(
                AugmentedSource(
                    source_id=source_id,
                    style=style,
                    source=source,
                    augmented_text=_extract_text(raw_response),
                    raw_response=raw_response,
                )
            )
        return results
