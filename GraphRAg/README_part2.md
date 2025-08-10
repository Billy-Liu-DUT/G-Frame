# Custom GraphRAG Backend for GraphRAG-Local UI

This project provides a custom, high-performance backend for the [GraphRAG Local UI](https://github.com/Azure-Samples/graphrag-local-ui-community) ecosystem. It replaces the default indexing and querying logic with a custom implementation based on the `nano_graphrag` library, while still leveraging the excellent Gradio interface for chat and visualization.

The backend runs as a standalone server that mimics an OpenAI API endpoint. This allows the Gradio UI to connect to it seamlessly for performing complex graph-based Retrieval-Augmented Generation (RAG) queries.

## Architecture

1.  **`ingest.py`**: A one-time script used to read your source documents, process them, and build the knowledge graph using `nano_graphrag`.
2.  **`serve.py`**: A long-running HTTP server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint. It receives requests from the Gradio UI, translates them into `nano_graphrag` queries, and returns the results.
3.  **`config.yaml`**: A central configuration file for all settings, including model endpoints, paths, and system prompts.
4.  **GraphRAG-Local UI**: The user-facing Gradio applications (`app.py`, `index_app.py`, etc.) that act as the client.

## Setup and Installation

This guide combines the setup for both the backend service and the frontend UI.

1.  **Create and Activate a Conda Environment:**
    It's recommended to use a clean environment to avoid dependency conflicts.
    ```bash
    conda create -n graphrag-suite -y
    conda activate graphrag-suite
    ```

2.  **Install Required Packages:**
    This project requires dependencies for both the UI and the custom backend.
    ```bash
    pip install "uvicorn[standard]" fastapi openai numpy pyyaml gradio
    # You may need to install nano_graphrag or other custom libraries
    # pip install nano_graphrag
    ```

## Configuration

All settings for this project are managed in the `config.yaml` file. Before launching, you must create this file and populate it with your specific settings.

1.  **Create the `config.yaml` file** in the root of the project directory.
2.  **Populate it** using the template provided in this repository.
3.  **Modify the values**, especially `llm_endpoint`, `embedding_endpoint`, and paths under `data_paths` to match your local environment.

## Running the System

The system requires running your local models, this backend server, and the Gradio UI application.

### Step 1: Run Your Local LLMs

Ensure your OpenAI-compatible LLM and Embedding models are running. For example, if using Ollama, make sure the service is active.

### Step 2: Ingest Data to Build the Graph (One-Time Task)

Before you can query, you must build the knowledge graph from your source documents.

1.  Configure the `data_paths` section in your `config.yaml`.
2.  Run the ingestion script:
    ```bash
    python ingest.py
    ```
    This will process the documents specified in the config and save the graph to the `working_dir`.

### Step 3: Launch the Custom Backend Server

This server will handle all RAG queries from the UI.

```bash
python serve.py
```
By default, it will start on `http://localhost:8004`. You should see a confirmation message in your terminal.

### Step 4: Launch the GraphRAG-Local Gradio UI

Now, launch the user interface.

```bash
gradio app.py
```

### Step 5: Connect the UI to Your Backend

This is the final and most important step.

1.  Open the Gradio UI in your browser (usually `http://localhost:7860`).
2.  Navigate to the **"LLM Settings"** tab.
3.  In the **"LLM API Base URL"** field, enter the address of **your custom backend server**:
    ```
    http://localhost:8004/v1
    ```
4.  You can now use the "Chat" tab to interact with your knowledge graph. Select the **"direct"** query type to send requests to your backend.