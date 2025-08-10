import json
import os
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

# --- Configuration ---

# This script processes raw JSON data and saves it as a tokenized HF Dataset.
# Modify these paths and settings according to your project structure.
RAW_DATA_CONFIG = {
    # Directory containing your raw JSON files to process.
    "input_dir": "/path/to/your/raw_sft_json_chunks",
    # Path to the tokenizer for preparing the data.
    # IMPORTANT: This should be the same tokenizer used for training.
    "tokenizer_path": "/path/to/your/pretrained_checkpoint",
    # Path to save the final tokenized dataset.
    "output_dir": "/path/to/your/tokenized_sft_dataset",
}

TOKENIZATION_CONFIG = {
    "max_seq_length": 4096,
    "num_proc": 32,  # Number of CPU processes for parallel tokenization.
}


# --- Data Processing Logic ---

def create_chat_messages(question: str, cot_answer: str, final_answer: str) -> list:
    """Creates a message list in the OpenAI chat format."""
    system_prompt = "You are a chemistry expert. Your task is to answer the user's question using the most academic and rigorous professor-level language in a structured format. Think step by step."

    if cot_answer and final_answer:
        assistant_content = f"<think>{cot_answer}</think> {final_answer}"
    else:
        # Handle cases where CoT or final answer might be missing.
        assistant_content = final_answer or cot_answer or ""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content}
    ]


def load_and_process_raw_data(config: dict) -> Dataset:
    """Loads raw JSON chunks, processes them, and returns a Hugging Face Dataset."""
    all_messages = []
    base_dir = config["input_dir"]

    print(f"Scanning files in {base_dir}...")
    for filename in tqdm(os.listdir(base_dir)):
        file_path = os.path.join(base_dir, filename)
        if not filename.endswith(".json"):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunk_list = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read or parse {filename}. Error: {e}")
            continue

        for item in chunk_list:
            questions = item.get("Questions", [])
            cot_answers = item.get("COT_Answers", [])
            answers = item.get("Answers", [])

            # Ensure all lists have the same length for zipping
            min_len = min(len(questions), len(cot_answers), len(answers))

            for q, cota, a in zip(questions[:min_len], cot_answers[:min_len], answers[:min_len]):
                if q and (cota or a):  # Ensure there is content to process
                    messages = create_chat_messages(q, cota, a)
                    all_messages.append({"messages": messages})

    print(f"Processed a total of {len(all_messages)} conversations.")
    return Dataset.from_list(all_messages)


def tokenize_and_save_dataset(dataset: Dataset, tokenizer, config: dict):
    """Tokenizes the dataset and saves it to disk."""

    def format_and_tokenize(example):
        """Applies chat template, tokenizes, and creates labels."""
        messages = example["messages"]
        if not messages:
            return None

        # Format the full conversation
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_tokenized = tokenizer(
            full_text,
            max_length=config["max_seq_length"],
            truncation=True,
            padding=False,  # We will pad later with the data collator
        )

        # Format the prompt part to calculate its length
        prompt_messages = messages[:-1]
        if prompt_messages:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokenized = tokenizer(prompt_text, add_special_tokens=False)
            prompt_length = len(prompt_tokenized["input_ids"])
        else:
            prompt_length = 0

        # Create labels by masking the prompt tokens
        labels = full_tokenized["input_ids"].copy()
        labels[:prompt_length] = [-100] * prompt_length

        full_tokenized["labels"] = labels
        return full_tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        format_and_tokenize,
        remove_columns=list(dataset.column_names),
        num_proc=config["num_proc"],
        desc="Formatting and tokenizing SFT data",
    )

    print(f"Saving tokenized dataset to: {config['output_dir']}")
    tokenized_dataset.save_to_disk(config["output_dir"])
    print("Dataset successfully processed and saved.")


if __name__ == "__main__":
    # 1. Load raw data into a Hugging Face Dataset
    raw_dataset = load_and_process_raw_data(RAW_DATA_CONFIG)

    # 2. Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(RAW_DATA_CONFIG["tokenizer_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenize the dataset and save it to disk
    tokenize_and_save_dataset(raw_dataset, tokenizer, TOKENIZATION_CONFIG)