# G-Frame & OmniChem: An Adaptive Multi-Agent Framework for Scientific Discovery in Chemistry

This repository contains the complete open-source suite for **G-Frame**, an adaptive multi-agent framework designed to overcome the reasoning deficiencies and factual "hallucinations" of lightweight Large Language Models (LLMs) in specialized scientific domains. The primary outcome of this framework is **OmniChem**, a 7B parameter model that demonstrates expert-level performance in chemistry, comparable to models like GPT-4o mini.

The core of G-Frame is a novel data synthesis pipeline that leverages **Bayesian game theory** to model the interaction between multiple LLM agents. This approach establishes an automated, closed-loop process where agents are incentivized to generate factually accurate and causally reasoned data, effectively internalizing the axiomatic constraints of science. The framework's **adaptive concurrency optimization** ensures maximum throughput and efficiency during this large-scale data generation process.

This work establishes a scalable and automated paradigm for creating domain-specific AI, providing a feasible path for deploying general AI to accelerate knowledge discovery in science.

## Abstract

> The application of lightweight Large Language Models (LLMs) in rule-based scientific domains, such as chemistry, is severely limited. This arises from their tendency to mimic linguistic patterns rather than reproduce the axiomatic, causal reasoning of experts, which results in a high propensity for "hallucinations". Here, we present G-Frame, an adaptive multi-agent framework that integrates Bayesian and team game principles to establish an automated closed-loop for high-quality data synthesis and model training. This process is designed to internalize domain constraints and structured reasoning. Using this framework, a specialized chemical corpus comprising 363,045 chains-of-thought (CoT) and 199,589 question-answer (QA) pairs was synthesized. The resulting 7B model, OmniChem, achieved a performance level on a custom benchmark comparable to that of GPT-4o mini, with a 90% reduction in hallucinations compared to its base model. The advanced capabilities of OmniChem in molecular design and synthesis planning are also demonstrated. This work establishes a scalable and automated paradigm, utilizing adaptive multi-agents and synthetic data to overcome the inherent reasoning deficiencies and hallucinations inherent in lightweight LLMs. A feasible path is thus presented for deploying general AI to accelerate knowledge discovery in specific scientific fields.

## The G-Frame Workflow: From Data to Deployment

This repository provides an end-to-end toolchain that covers the entire lifecycle of creating a specialized scientific LLM.

<p align="center">
  <img src="https://i.imgur.com/your-workflow-diagram.png" alt="G-Frame Workflow" width="800"/>
  <br/>
  <i><b>Figure 1:</b> The automated, closed-loop workflow of the G-Frame ecosystem.</i>
</p>

1.  **Stage 1: High-Quality Data Synthesis (The G-Frame Core)**: The `1_data_synthesis` module uses an adaptive concurrency manager to efficiently orchestrate multiple LLM agents. Modeled on Bayesian game principles, this system generates a vast corpus of high-quality, factually grounded chemical data.
2.  **Stage 2: Model Training**: The `2_model_training` scripts use the synthesized data to fine-tune a base model, creating the specialized OmniChem model. The suite supports both pre-training and supervised fine-tuning (SFT) with DeepSpeed.
3.  **Stage 3: Application & Interaction (RAG)**: The trained OmniChem model is deployed and integrated into a sophisticated Retrieval-Augmented Generation (RAG) system using the `4_rag_backend` and `5_chat_ui` components, enabling intuitive interaction via a Gradio interface.
4.  **Stage 4: Evaluation**: The final model's performance is rigorously tested on custom chemistry benchmarks provided in the `3_evaluation_benchmarks` directory.

## 🤗 Model and Dataset on Hugging Face

* **Model:** [**Billy-Liu-DUT/OmniChem-7B-v1**](https://huggingface.co/Billy-Liu-DUT/OmniChem-7B-v1)
* **Dataset:** [**Billy-Liu-DUT/OmniChem**](https://huggingface.co/datasets/Billy-Liu-DUT/OmniChem)

## 📂 Repository Structure

```
.
├── 1_data_synthesis/           # The G-Frame adaptive concurrency framework for data generation.
│   ├── adaptive_manager.py
│   ├── vllm_server.py
│   ├── task_executor.py
│   └── config.yaml
│
├── 2_model_training/           # Scripts and configs for pre-training and SFT.
│   ├── run_sft.py
│   └── ds_config_sft.json
│
├── 3_evaluation_benchmarks/    # Chemistry benchmark datasets.
│   ├── chemjudge.json
│   └── ...
│
├── 4_rag_backend/              # The custom nano_graphrag backend server.
│   ├── ingest.py
│   ├── serve.py
│   └── config.yaml
│
└── 5_chat_ui/                  # The Gradio-based local UI for chat and visualization.
    ├── app.py
    └── ...
```

## 🚀 Getting Started: A Step-by-Step Guide

This guide provides a high-level overview of how to use the components in this repository to replicate the research workflow. **Please refer to the detailed `README` file within each subdirectory for specific instructions.**

### Step 1: Data Synthesis with G-Frame (Core Innovation)

Generate the high-quality synthetic data corpus. This stage is the practical application of the Bayesian game theory mentioned in our paper.

1.  **Navigate to the directory:** `cd 1_data_synthesis/`
2.  **Configure `config.yaml`**: Set up your base model paths, data sources, API keys, and other parameters.
3.  **Launch a base LLM Server**: Ensure your base model is served via an OpenAI-compatible endpoint (e.g., using vLLM).
4.  **Run the Manager**: Start the automated data generation process: `python adaptive_manager.py`.

### Step 2: Model Training (Creating OmniChem)

Use the synthesized data to train the OmniChem model.

1.  **Navigate to the directory:** `cd 2_model_training/`
2.  **Prepare your Dataset**: Follow the instructions in the subdirectory's `README` to prepare your synthetic data for SFT.
3.  **Configure and Launch**: Edit `run_sft.py` to point to your dataset and base model, then launch with DeepSpeed: `deepspeed --num_gpus <N> run_sft.py`.

### Step 3: Deployment and Interaction (Using OmniChem)

Deploy the trained OmniChem model and interact with it through the custom RAG backend and Gradio UI.

1.  **Deploy OmniChem with vLLM**: Serve your newly fine-tuned OmniChem model.
2.  **Build the Knowledge Graph**: In `4_rag_backend/`, configure your source documents and run `python ingest.py`.
3.  **Launch the RAG Backend Server**: Start the custom server: `python serve.py`.
4.  **Launch the Gradio UI**: In `5_chat_ui/`, start the application: `gradio app.py`.
5.  **Connect UI to Backend**: In the Gradio UI's "LLM Settings," point the "API Base URL" to your RAG backend server (e.g., `http://localhost:8004/v1`).

### Step 4: Evaluation

Quantitatively measure your model's performance using the provided benchmarks.

1.  **Navigate to the directory:** `cd 3_evaluation_benchmarks/`
2.  Use the JSON files (`chemjudge.json`, etc.) as input for your evaluation scripts.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgements

This project was developed by the **Digital Chemistry Research Group** at **Dalian University of Technology**.

Key contributors include **Biquan Bie** and **Runzhe Liu**.

## Citing G-Frame & OmniChem

If you use the G-Frame framework or the OmniChem model in your research, please consider citing this repository. You can use the "Cite this repository" feature on the right sidebar of the GitHub page.
