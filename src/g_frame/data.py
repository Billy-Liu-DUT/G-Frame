"""Closed-data-safe providers and v2 SFT record serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping

from .schemas import TrainingRecord


class FileDatasetProvider:
    """Reads caller-supplied JSONL without copying source data into the repository."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path.expanduser().resolve()

    def iter_sources(self) -> Iterator[Dict[str, str]]:
        with self.source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"line {line_number} must be a JSON object")
                source_id = str(value.get("source_id", f"line-{line_number}"))
                source = value.get("source")
                if not isinstance(source, str) or not source.strip():
                    raise ValueError(f"line {line_number} is missing a non-empty source")
                yield {"source_id": source_id, "source": source}


class SFTDatasetBuilder:
    """Converts reviewed v2 records into the chat schema consumed by SFT."""

    @staticmethod
    def build_rows(records: Iterable[TrainingRecord], approved_only: bool = True) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for record in records:
            if approved_only and not record.approved:
                continue
            rows.append(
                {
                    "record_id": record.task_id,
                    "source_id": record.source_id,
                    "messages": record.to_chat_messages(),
                    "review": {"approved": record.approved, "feedback": record.reviewer_feedback},
                }
            )
        return rows

    @staticmethod
    def write_jsonl(records: Iterable[TrainingRecord], output_path: Path, approved_only: bool = True) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = SFTDatasetBuilder.build_rows(records, approved_only=approved_only)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)


def write_training_records(records: Iterable[TrainingRecord], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
