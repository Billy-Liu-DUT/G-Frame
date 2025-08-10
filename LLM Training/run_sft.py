import torch
import wandb
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

# --- 1. Configuration ---
# All user-configurable parameters for the SFT stage.
# IMPORTANT: Replace all placeholder paths with your actual paths.

SFT_MODEL_CONFIG = {
    # Path to the pre-trained model checkpoint to be fine-tuned.
    "model_path": "/path/to/your/pretrained_checkpoint",
    # Set to True if you added special tokens (e.g., <think>) during training.
    "add_special_tokens": True,
}

SFT_DATA_CONFIG = {
    # A list of paths to your final, tokenized SFT datasets.
    "dataset_paths": [
        "/path/to/your/tokenized_sft_dataset_1",
        "/path/to/your/tokenized_sft_dataset_2",
    ],
}

SFT_TRAINING_CONFIG = {
    "output_dir": "/path/to/your/sft_checkpoints",
    "deepspeed_config": "./ds_config_sft.json",
    "run_name": "omnichem-sft-stage-1",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "logging_steps": 10,
    "save_steps": 200,
    "bf16": True,
    "seed": 42,
}

# --- 2. Initialization ---

set_seed(SFT_TRAINING_CONFIG["seed"])

print(f"Loading tokenizer from: {SFT_MODEL_CONFIG['model_path']}")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_CONFIG["model_path"])

if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token, setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

if SFT_MODEL_CONFIG["add_special_tokens"]:
    special_tokens_dict = {"additional_special_tokens": ["<think>", "</think>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    print("Added special tokens to tokenizer.")

print(f"Loading model from: {SFT_MODEL_CONFIG['model_path']}")
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_CONFIG["model_path"],
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

if SFT_MODEL_CONFIG["add_special_tokens"]:
    model.resize_token_embeddings(len(tokenizer))
    print("Resized model token embeddings.")

# --- 3. Data Loading and Preparation ---

print("Loading and combining tokenized SFT datasets...")
loaded_datasets = [load_from_disk(path) for path in SFT_DATA_CONFIG["dataset_paths"]]
tokenized_dataset = concatenate_datasets(loaded_datasets).shuffle(seed=SFT_TRAINING_CONFIG["seed"])
print(f"Final SFT dataset size: {len(tokenized_dataset)} samples.")

# For SFT, DataCollatorForSeq2Seq is often used to handle padding for both inputs and labels.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

# --- 4. Training Setup ---

print("Initializing Weights & Biases for experiment tracking...")
wandb.init(mode="offline", project="omnichem-sft")

print("Setting up Training Arguments for SFT...")
training_args = TrainingArguments(
    output_dir=SFT_TRAINING_CONFIG["output_dir"],
    deepspeed=SFT_TRAINING_CONFIG["deepspeed_config"],
    run_name=SFT_TRAINING_CONFIG["run_name"],
    num_train_epochs=SFT_TRAINING_CONFIG["num_train_epochs"],
    per_device_train_batch_size=SFT_TRAINING_CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=SFT_TRAINING_CONFIG["gradient_accumulation_steps"],
    learning_rate=SFT_TRAINING_CONFIG["learning_rate"],
    logging_steps=SFT_TRAINING_CONFIG["logging_steps"],
    save_steps=SFT_TRAINING_CONFIG["save_steps"],
    bf16=SFT_TRAINING_CONFIG["bf16"],
    report_to="wandb",
    remove_unused_columns=False,
    seed=SFT_TRAINING_CONFIG["seed"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 5. Start Training ---

print("Starting supervised fine-tuning...")
train_result = trainer.train()

# Log and save metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print("Saving final model and tokenizer...")
trainer.save_model(SFT_TRAINING_CONFIG["output_dir"])
tokenizer.save_pretrained(SFT_TRAINING_CONFIG["output_dir"])
print("SFT finished and model saved successfully.")

if __name__ == "__main__":
    # This script is intended to be run directly.
    pass