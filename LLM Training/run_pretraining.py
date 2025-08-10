import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# --- 1. Configuration ---
# All user-configurable parameters for the pre-training stage.
# IMPORTANT: Replace all placeholder paths with your actual paths.

PRETRAIN_MODEL_CONFIG = {
    "model_path": "/path/to/your/base_model_or_checkpoint",
    "tokenizer_path": "/path/to/your/tokenizer",
}

PRETRAIN_DATA_CONFIG = {
    # Path to a single, large, pre-processed dataset for pre-training.
    "dataset_path": "/path/to/your/pretraining_dataset_in_arrow_format",
    "max_seq_length": 4096,
    "num_proc": 32,
    "map_batch_size": 1000,
}

PRETRAIN_TRAINING_CONFIG = {
    "output_dir": "/path/to/your/pretraining_checkpoints",
    "deepspeed_config": "./ds_config_pretrain.json",
    "run_name": "omnichem-pretraining-stage-1",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.01,
    "logging_steps": 10,
    "save_steps": 500,
    "bf16": True,
}

# --- 2. Data Preparation ---

def tokenize_dataset(dataset, tokenizer, config):
    """Tokenizes a dataset using the provided tokenizer."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["max_seq_length"],
            return_special_tokens_mask=True,
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=list(dataset.features),
        num_proc=config["num_proc"],
        batch_size=config["map_batch_size"],
    )

print(f"Loading pre-training dataset from: {PRETRAIN_DATA_CONFIG['dataset_path']}")
dataset = load_from_disk(PRETRAIN_DATA_CONFIG["dataset_path"])
shuffled_dataset = dataset.shuffle(seed=42)
print(f"Dataset loaded successfully with {len(shuffled_dataset)} samples.")

# --- 3. Model and Tokenizer Initialization ---

print(f"Loading tokenizer from: {PRETRAIN_MODEL_CONFIG['tokenizer_path']}")
tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_CONFIG["tokenizer_path"])
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token, setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from: {PRETRAIN_MODEL_CONFIG['model_path']}")
model = AutoModelForCausalLM.from_pretrained(PRETRAIN_MODEL_CONFIG["model_path"])

print("Tokenizing the pre-training dataset...")
tokenized_train_dataset = tokenize_dataset(shuffled_dataset, tokenizer, PRETRAIN_DATA_CONFIG)
print("Tokenization complete.")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# --- 4. Training Setup ---

print("Initializing Weights & Biases for experiment tracking...")
wandb.init(mode="offline", project="omnichem-pretraining")

print("Setting up Training Arguments for pre-training...")
training_args = TrainingArguments(
    output_dir=PRETRAIN_TRAINING_CONFIG["output_dir"],
    run_name=PRETRAIN_TRAINING_CONFIG["run_name"],
    num_train_epochs=PRETRAIN_TRAINING_CONFIG["num_train_epochs"],
    per_device_train_batch_size=PRETRAIN_TRAINING_CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=PRETRAIN_TRAINING_CONFIG["gradient_accumulation_steps"],
    learning_rate=PRETRAIN_TRAINING_CONFIG["learning_rate"],
    weight_decay=PRETRAIN_TRAINING_CONFIG["weight_decay"],
    warmup_ratio=PRETRAIN_TRAINING_CONFIG["warmup_ratio"],
    logging_steps=PRETRAIN_TRAINING_CONFIG["logging_steps"],
    save_steps=PRETRAIN_TRAINING_CONFIG["save_steps"],
    bf16=PRETRAIN_TRAINING_CONFIG["bf16"],
    gradient_checkpointing=True,
    deepspeed=PRETRAIN_TRAINING_CONFIG["deepspeed_config"],
    report_to="wandb",
    dataloader_drop_last=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=data_collator,
)

# --- 5. Start Training ---

print("Starting model pre-training...")
trainer.train()

print("Pre-training finished successfully.")