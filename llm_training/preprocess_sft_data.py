"""Prepare G-Frame SFT data without import-time ML side effects.

The v1 script only accepted a directory of private JSON chunks and executed as
soon as it was imported.  v2 keeps that conversion path and additionally
accepts the structured ``messages`` JSONL written by the Team Game workflow.
Tokenization is an explicit, local preprocessing step; the v2 runner can also
tokenize JSONL directly during training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


INPUT_FORMAT_LEGACY_JSON_DIR = "legacy-json-dir"
INPUT_FORMAT_V2_JSONL = "v2-jsonl"
SUPPORTED_INPUT_FORMATS = frozenset({INPUT_FORMAT_LEGACY_JSON_DIR, INPUT_FORMAT_V2_JSONL})
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


def create_chat_messages(question: str, cot_answer: str, final_answer: str) -> List[Dict[str, str]]:
    """Create the legacy OmniChem conversation format used by v1 data chunks."""

    system_prompt = (
        "You are a chemistry expert. Your task is to answer the user's question using "
        "the most academic and rigorous professor-level language in a structured format. "
        "Think step by step."
    )
    if cot_answer and final_answer:
        assistant_content = f"<think>{cot_answer}</think> {final_answer}"
    else:
        assistant_content = final_answer or cot_answer or ""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
    ]


def _as_string_list(value: Any, field_name: str, source: Path) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{source}: {field_name} must be a list of strings")
    return value


def collect_legacy_rows(input_directory: str | Path) -> List[Dict[str, Any]]:
    """Read v1 private JSON chunks into messages rows without copying source files."""

    directory = Path(input_directory).expanduser()
    if not directory.is_dir():
        raise ValueError(f"legacy JSON input directory does not exist: {directory}")
    rows: List[Dict[str, Any]] = []
    for source in sorted(directory.glob("*.json")):
        try:
            chunk_list = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}: invalid JSON: {error.msg}") from error
        if not isinstance(chunk_list, list):
            raise ValueError(f"{source}: expected a JSON list of chunk objects")
        for chunk_index, item in enumerate(chunk_list):
            if not isinstance(item, Mapping):
                raise ValueError(f"{source}: item {chunk_index} must be a JSON object")
            questions = _as_string_list(item.get("Questions"), "Questions", source)
            cot_answers = _as_string_list(item.get("COT_Answers"), "COT_Answers", source)
            answers = _as_string_list(item.get("Answers"), "Answers", source)
            for question_index, question in enumerate(questions):
                cot_answer = cot_answers[question_index] if question_index < len(cot_answers) else ""
                final_answer = answers[question_index] if question_index < len(answers) else ""
                if question.strip() and (cot_answer.strip() or final_answer.strip()):
                    rows.append(
                        {
                            "messages": create_chat_messages(question, cot_answer, final_answer),
                            "source_file": source.name,
                            "source_index": question_index,
                        }
                    )
    if not rows:
        raise ValueError(f"no usable legacy SFT conversations found in: {directory}")
    return rows


def _validate_v2_message(message: Any, line_number: int, message_index: int) -> Dict[str, str]:
    if not isinstance(message, Mapping):
        raise ValueError(f"line {line_number}, messages[{message_index}] must be an object")
    role = message.get("role")
    content = message.get("content")
    if role not in _ALLOWED_ROLES:
        allowed = ", ".join(sorted(_ALLOWED_ROLES))
        raise ValueError(
            f"line {line_number}, messages[{message_index}].role must be one of: {allowed}"
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"line {line_number}, messages[{message_index}].content must be non-empty")
    return {"role": role, "content": content}


def collect_v2_jsonl_rows(input_path: str | Path) -> List[Dict[str, Any]]:
    """Read Team Game JSONL rows and retain only the training-relevant fields."""

    source = Path(input_path).expanduser()
    if not source.is_file():
        raise ValueError(f"v2 messages JSONL does not exist: {source}")
    rows: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(item, Mapping):
                raise ValueError(f"line {line_number} must be a JSON object")
            raw_messages = item.get("messages")
            if not isinstance(raw_messages, list) or not raw_messages:
                raise ValueError(f"line {line_number} must contain a non-empty messages list")
            messages = [
                _validate_v2_message(message, line_number, index)
                for index, message in enumerate(raw_messages)
            ]
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"line {line_number} must end with an assistant response")
            row: Dict[str, Any] = {"messages": messages}
            if "record_id" in item:
                row["record_id"] = str(item["record_id"])
            if "source_id" in item:
                row["source_id"] = str(item["source_id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"v2 messages JSONL has no usable conversations: {source}")
    return rows


def collect_rows(input_path: str | Path, input_format: str) -> List[Dict[str, Any]]:
    if input_format == INPUT_FORMAT_LEGACY_JSON_DIR:
        return collect_legacy_rows(input_path)
    if input_format == INPUT_FORMAT_V2_JSONL:
        return collect_v2_jsonl_rows(input_path)
    supported = ", ".join(sorted(SUPPORTED_INPUT_FORMATS))
    raise ValueError(f"input_format must be one of: {supported}")


def load_and_process_raw_data(config: Mapping[str, Any]) -> Any:
    """v1-compatible wrapper that returns a Hugging Face Dataset lazily."""

    input_dir = config.get("input_dir")
    if not isinstance(input_dir, str) or not input_dir.strip():
        raise ValueError("config.input_dir must be a non-empty string")
    from datasets import Dataset

    return Dataset.from_list(collect_legacy_rows(input_dir))


def _format_and_tokenize_messages(
    messages: Sequence[Mapping[str, str]], tokenizer: Any, max_seq_length: int
) -> Dict[str, List[int]]:
    try:
        full_text = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            list(messages[:-1]), tokenize=False, add_generation_prompt=True
        )
    except AttributeError as error:
        raise RuntimeError(
            "the selected tokenizer has no chat template; choose a chat model for SFT"
        ) from error
    full_tokenized = tokenizer(
        full_text,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )
    prompt_tokenized = tokenizer(
        prompt_text,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )
    input_ids = list(full_tokenized["input_ids"])
    prompt_length = min(len(prompt_tokenized["input_ids"]), len(input_ids))
    labels = list(input_ids)
    labels[:prompt_length] = [-100] * prompt_length
    if all(label == -100 for label in labels):
        raise ValueError("a row was truncated before its assistant response")
    full_tokenized["labels"] = labels
    return full_tokenized


def tokenize_and_save_dataset(dataset: Any, tokenizer: Any, config: Mapping[str, Any]) -> Any:
    """v1-compatible tokenization helper with explicit configuration validation."""

    max_seq_length = config.get("max_seq_length")
    num_proc = config.get("num_proc", 1)
    output_dir = config.get("output_dir")
    if not isinstance(max_seq_length, int) or max_seq_length < 8:
        raise ValueError("config.max_seq_length must be an integer of at least 8")
    if not isinstance(num_proc, int) or num_proc < 1:
        raise ValueError("config.num_proc must be a positive integer")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("config.output_dir must be a non-empty string")
    target = Path(output_dir).expanduser()
    if target.exists():
        raise ValueError(f"refusing to overwrite existing tokenized dataset: {target}")

    def format_and_tokenize(example: Mapping[str, Any]) -> Dict[str, List[int]]:
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("every SFT row must contain a non-empty messages list")
        return _format_and_tokenize_messages(messages, tokenizer, max_seq_length)

    tokenized_dataset = dataset.map(
        format_and_tokenize,
        remove_columns=list(dataset.column_names),
        num_proc=num_proc,
        desc="Formatting and tokenizing SFT data",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(target))
    return tokenized_dataset


def preprocess_to_disk(
    input_path: str | Path,
    input_format: str,
    tokenizer_path: str,
    output_dir: str | Path,
    max_seq_length: int = 2048,
    num_proc: int = 1,
) -> int:
    """Load input rows, tokenize them, and write a legacy-compatible HF dataset."""

    rows = collect_rows(input_path, input_format)
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Preprocessing requires datasets and transformers. Install the v2 training "
            "environment before using --run."
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = Dataset.from_list(rows)
    tokenized = tokenize_and_save_dataset(
        dataset,
        tokenizer,
        {
            "max_seq_length": max_seq_length,
            "num_proc": num_proc,
            "output_dir": str(output_dir),
        },
    )
    return len(tokenized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G-Frame v2 SFT preprocessing for legacy chunks and v2 messages JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Legacy chunk directory or v2 messages JSONL")
    parser.add_argument(
        "--input-format",
        required=True,
        choices=sorted(SUPPORTED_INPUT_FORMATS),
        help="Input schema",
    )
    parser.add_argument("--tokenizer-path", help="Tokenizer used to create the saved dataset")
    parser.add_argument("--output-dir", help="New destination for the tokenized HF dataset")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate input rows only; does not import datasets or transformers",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Tokenize and save. This never launches model training.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.validate and not args.run:
        parser.error("choose --validate to inspect rows or --run to tokenize them")
    rows = collect_rows(args.input, args.input_format)
    if args.validate or args.run:
        print(json.dumps({"input_format": args.input_format, "rows": len(rows)}, sort_keys=True))
    if args.run:
        if not args.tokenizer_path or not args.output_dir:
            parser.error("--run requires --tokenizer-path and --output-dir")
        count = preprocess_to_disk(
            input_path=args.input,
            input_format=args.input_format,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            num_proc=args.num_proc,
        )
        print(json.dumps({"output_dir": args.output_dir, "tokenized_rows": count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
