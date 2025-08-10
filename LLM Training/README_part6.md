# Generic Language Model Training Suite

This repository provides a suite of scripts for training and fine-tuning causal language models using the Hugging Face `transformers` ecosystem and Microsoft's DeepSpeed library. It includes scripts for both large-scale pre-training and supervised fine-tuning (SFT).

## Project Structure

```
.
├── run_pretraining.py          # Script for further pre-training
├── run_sft.py                  # Script for supervised fine-tuning
├── preprocess_sft_data.py      # Optional: Helper script to prepare SFT data
|
├── ds_config_pretrain.json     # DeepSpeed config for pre-training
├── ds_config_sft.json          # DeepSpeed config for fine-tuning
└── README.md                   # This documentation
```

## Workflow Overview

Both training and fine-tuning follow a two-step process:
1.  **Data Preparation**: Your raw dataset must be converted into a tokenized, ready-to-use Hugging Face `Dataset` saved to disk.
2.  **Training**: The training script loads the prepared dataset and starts the training run.

## Setup

### Step 1: Install Dependencies

Ensure you have Python 3.9+ and the required packages installed.

```bash
pip install transformers datasets torch wandb deepspeed numpy
```

### Step 2: Create DeepSpeed Configuration Files

This project requires separate DeepSpeed configuration files for each training stage. Create `ds_config_pretrain.json` and `ds_config_sft.json` in your project's root directory. You can use the templates provided in this repository.

## Stage 1: Pre-training

### Step 1.1: Data Preparation

Ensure you have a large-scale, pre-processed text dataset saved in the Hugging Face `arrow` format on disk. The dataset should have a single column named "text".

### Step 1.2: Configure the Script

Open `run_pretraining.py` and modify the configuration dictionaries at the top of the file. **You must replace all placeholder paths with your actual paths.**

-   `PRETRAIN_MODEL_CONFIG`: Set paths to your base model and tokenizer.
-   `PRETRAIN_DATA_CONFIG`: Provide the path to your pre-training dataset.
-   `PRETRAIN_TRAINING_CONFIG`: Define the output directory and hyperparameters.

### Step 1.3: Launch Pre-training

Use the `deepspeed` launcher to start.

```bash
deepspeed --num_gpus <number_of_gpus> run_pretraining.py
```

## Stage 2: Supervised Fine-Tuning (SFT)

### Step 2.1: Data Preparation

The `run_sft.py` script expects a dataset formatted in the OpenAI chat format (`[{"role": "system", ...}, {"role": "user", ...}]`). A helper script, `preprocess_sft_data.py`, is provided to convert raw JSON data into the required tokenized format.

1.  Modify `preprocess_sft_data.py` to point to your raw data source(s).
2.  Run the script to generate the final dataset:
    ```bash
    python preprocess_sft_data.py
    ```

### Step 2.2: Configure the Script

Open `run_sft.py` and modify the configuration dictionaries at the top.

-   `SFT_MODEL_CONFIG`: Set the path to the **pre-trained model checkpoint** you want to fine-tune.
-   `SFT_DATA_CONFIG`: Provide the path to your **final tokenized SFT dataset** (generated in the previous step).
-   `SFT_TRAINING_CONFIG`: Define the output directory and fine-tuning hyperparameters (e.g., smaller learning rate).

### Step 2.3: Launch SFT

Use the `deepspeed` launcher to start the fine-tuning run.

```bash
deepspeed --num_gpus <number_of_gpus> run_sft.py
```