"""Versioned prompt loading for the v2 agent workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class PromptCatalog:
    """Loads editable prompt assets without binding the core to a model SDK."""

    def __init__(self, asset_dir: Optional[Path] = None) -> None:
        self.asset_dir = asset_dir or Path(__file__).resolve().parent / "prompt_assets"
        self._cache: Dict[str, Dict[str, str]] = {}

    def _load_group(self, group: str) -> Dict[str, str]:
        if group not in self._cache:
            path = self.asset_dir / f"{group}.json"
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            templates = payload.get("templates", payload)
            if not isinstance(templates, dict):
                raise ValueError(f"prompt group {group} must contain an object of templates")
            self._cache[group] = {str(key): str(value) for key, value in templates.items()}
        return self._cache[group]

    def metadata(self, group: str) -> Dict[str, Any]:
        path = self.asset_dir / f"{group}.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {key: value for key, value in payload.items() if key != "templates"}

    def render(self, group: str, name: str, **values: Any) -> str:
        templates = self._load_group(group)
        if name not in templates:
            raise KeyError(f"prompt {group}/{name} was not found")
        try:
            return templates[name].format(**values)
        except KeyError as exc:
            raise ValueError(f"missing prompt value {exc.args[0]} for {group}/{name}") from exc
