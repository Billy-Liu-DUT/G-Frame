# Deep Research Agent Framework

This project is an automated, multi-agent framework designed to conduct in-depth academic literature research on a given topic. It follows a multi-turn process of retrieving, summarizing, and reflecting on scholarly articles to produce a comprehensive, high-quality research report.

The agent leverages the Clarivate Web of Science (WOS) API for literature discovery, web scraping for full-text retrieval, and Large Language Models (LLMs) for content summarization, outline generation, reflection, and final report polishing.

## Architecture

The framework is built on an asynchronous Python stack and is divided into several modular components:

1.  **`main.py` (Orchestrator)**: The main entry point that orchestrates the entire research workflow, from initial keyword search to the final report generation.
2.  **`data_retrieval.py` (Data Retrieval Module)**: Contains all functions related to external data acquisition, including interacting with the WOS API and scraping full-text content from DOIs.
3.  **`llm_services.py` (LLM Services Module)**: Centralizes all interactions with Large Language Models. It manages different LLM accounts, prompt templates, and API call logic, all driven by the central configuration file.
4.  **`agent_logic.py` (Agent Logic Module)**: Contains the core "intelligence" of the agent, including functions for summarizing articles, generating outlines, reflecting on research gaps, and polishing the final report.
5.  **`config.yaml` (Configuration File)**: A single, centralized file to manage all API keys, model names, file paths, prompts, and other parameters.

## Workflow

The agent operates in a sequential, multi-turn workflow:

1.  **Initial Search**: Retrieves an initial set of articles from Web of Science based on a primary keyword.
2.  **Content Retrieval**: Scrapes the full text of the retrieved articles using their DOIs.
3.  **Summarization & Outline**: Summarizes each article and generates an initial research outline based on the collected knowledge.
4.  **Reflection & Expansion**: The agent "reflects" on the initial outline to identify knowledge gaps and generates new, more specific keywords for a deeper search.
5.  **Deep Dive Loop**: The agent performs another round of search, retrieval, and summarization using the new keywords.
6.  **Report Generation & Polishing**: The agent synthesizes all collected information into a comprehensive research report and then polishes it to meet academic standards.

## Setup

### Step 1: Install Dependencies

Ensure you have Python 3.9+ and the required packages installed.

```bash
# Core dependencies
pip install pyyaml openai tqdm aiohttp clarivate.wos_starter.client beautifulsoup4

# For web scraping with Playwright
pip install playwright
python -m playwright install firefox
```

### Step 2: Configure the Project

All settings are managed in the `config.yaml` file.

1.  Create a file named `config.yaml` in the root of the project directory.
2.  Use the template provided in this repository to populate the file.
3.  **Crucially, you must fill in your API keys** for Clarivate Web of Science and your chosen LLM provider(s).
4.  Adjust file paths and model names as needed for your environment.

## How to Run

Once the `config.yaml` is configured, you can start the deep research process by running the main script.

1.  The script will prompt you to enter your research topic and an initial keyword.
2.  The agent will then begin the automated workflow.

```bash
python main.py
```

The process will generate intermediate and final files in the `results/` directory (or the directory you specify in the config), culminating in a polished research report in Markdown format.