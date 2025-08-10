# Advanced Chat UI for OpenAI-compatible APIs

This is a full-featured, local chat web interface built with Gradio. It is designed to connect to any Large Language Model (LLM) that is served via an OpenAI-compatible API endpoint (such as models served with vLLM, TGI, or local-ai).

It provides a robust and user-friendly platform for interacting with your models, complete with conversation history, parameter tuning, and dynamic persona management.

![Screenshot of the App](https://i.imgur.com/rS2hS0d.png)  ## ✨ Features

- **Connect to Any OpenAI-compatible API**: Easily configure the API base URL and key to point to your local or remote LLM service.
- **Streaming Responses**: Messages appear token-by-token for a real-time chat experience.
- **Persistent Chat History**: Conversations are automatically saved to disk and can be reloaded at any time, even after restarting the application.
- **Dynamic System Prompts**:
    - Choose from a dropdown of pre-defined "personas" or roles for the model.
    - A "Custom" option allows for on-the-fly editing of the system prompt.
- **Full Parameter Control**: Adjust advanced generation parameters like `temperature`, `top_p`, and `max_tokens` directly from the UI.
- **Stop Generation**: Immediately interrupt the model while it's generating a response.
- **Structured Output Parsing**: Automatically formats special tags (e.g., `<think>...</think>`) into collapsible sections for a cleaner interface.
- **Clean, Modern UI**: Built with Gradio for a responsive and intuitive user experience.

## 🚀 Getting Started

Follow these steps to get the chat interface up and running.

### Prerequisites

- Python 3.9+
- An available LLM served through an OpenAI-compatible API endpoint.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install the required dependencies:**
    A `requirements.txt` file is provided. Install it using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Gradio Web UI:**
    ```bash
    python chat-app.py
    ```

2.  **Open the Interface:**
    The console will output a local URL (usually `http://127.0.0.1:7860`). Open this URL in your web browser.

3.  **Configure and Chat:**
    - Use the sidebar to set your API endpoint URL and API key.
    - Select a model, persona, and adjust parameters as needed.
    - Start chatting!

## 🔧 Configuration

Default settings can be modified directly at the top of the `app.py` file:

- **`API_BASE_URL`**: The default API endpoint.
- **`API_KEY`**: The default API key.
- **`SYSTEM_PROMPTS`**: Add, remove, or edit the pre-defined system prompt personas in this dictionary.

---