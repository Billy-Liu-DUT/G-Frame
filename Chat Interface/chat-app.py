import gradio as gr
from openai import AsyncOpenAI, APIError, APIConnectionError
import os
import time
import json
import re

# --- Constants and Configuration ---

HISTORY_DIR = "chat_history"

# Default API configuration
API_BASE_URL = "http://localhost:8002/v1"
API_KEY = "token123"  # Default key for local models

# Pre-defined System Prompts (Personas)
SYSTEM_PROMPTS = {
    "OmniChem (Expert)": "You are a chemistry expert. Your task is to answer the user's problem using the most academic and rigorous professor-level language in a structured format. Think step by step.",
    "ThChem 1.0": "Which of the following options is correct? Use <answer> and </answer> to enclose the answer option. Think step by step.",
    "ThChem 2.0": "Your task is to complete the following multiple-choice questions. If there is a correct option, directly give the letter before the option. If there is no correct option, please output 'null'. Use <answer> and </answer> to enclose the answer option. Think step by step.",
    "Default Assistant": "You are a helpful assistant.",
    "Custom...": "Enter your custom system prompt below...",
}
DEFAULT_PROMPT_KEY = "OmniChem (Expert)"


# --- Persistent Chat History Functions ---

def load_sessions_from_disk():
    """Loads all chat session .json files from the history directory."""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    sessions = {}
    files = sorted([f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")], reverse=True)
    for filename in files:
        try:
            with open(os.path.join(HISTORY_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                session_id = os.path.splitext(filename)[0]
                sessions[session_id] = data
        except Exception as e:
            print(f"Warning: Skipping corrupted history file '{filename}': {e}")
            continue
    return sessions


def save_session_to_disk(history):
    """Saves the current chat history to a new .json file."""
    if not history:
        return

    timestamp_str = time.strftime('%Y-%m-%d %H:%M')
    first_query = history[0][0][:30].strip()
    display_title = f"{timestamp_str} - {first_query}"

    filename_ts = time.strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_ts}.json"

    session_data = {
        "title": display_title,
        "history": history,
        "timestamp": time.time()
    }

    with open(os.path.join(HISTORY_DIR, filename), 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)


# --- Global State ---
chat_sessions = load_sessions_from_disk()


# --- Gradio UI Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="Advanced Chat UI") as demo:
    gr.Markdown("# 🤖 OmniChem Chat Page")

    with gr.Row():
        # Sidebar for configuration
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Model & API Configuration")

            with gr.Accordion("API Configuration", open=True):
                base_url_box = gr.Textbox(label="API Base URL", value=API_BASE_URL)
                api_key_box = gr.Textbox(label="API Key", value=API_KEY, type="password")

            model_dd = gr.Dropdown(
                label="Select Model",
                choices=["omnichem", "qwen72b", "Qwen2.5-7B-Instruct", "llama8b", "chemdfm"],
                value="omnichem"
            )

            gr.Markdown("### System Prompt Settings")
            prompt_template_dd = gr.Dropdown(
                label="Select a Persona",
                choices=list(SYSTEM_PROMPTS.keys()),
                value=DEFAULT_PROMPT_KEY
            )
            system_prompt_box = gr.Textbox(
                label="System Prompt Content",
                value=SYSTEM_PROMPTS[DEFAULT_PROMPT_KEY],
                lines=8,
                interactive=False  # Disabled by default
            )

            with gr.Accordion("Advanced Parameters", open=False):
                temperature_slider = gr.Slider(0, 2.0, 0.6, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(0, 1.0, 0.1, step=0.05, label="Top-p")
                max_tokens_slider = gr.Slider(1, 32768, 16248, step=1, label="Max Tokens")
                presence_penalty_slider = gr.Slider(-2.0, 2.0, 1.2, step=0.1, label="Presence Penalty")

            gr.Markdown("---")
            gr.Markdown("## 📜 Chat History")
            new_chat_btn = gr.Button("➕ New Chat", variant="secondary")
            history_box = gr.Radio(
                label="Load Chat from Disk",
                choices=[data['title'] for data in chat_sessions.values()],
                value=None,
                interactive=True
            )

        # Main chat interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat Window", bubble_full_width=False, height=600, render_markdown=True)
            with gr.Row():
                msg_box = gr.Textbox(label="Message Input", placeholder="Type your message here...", scale=7, autofocus=True)
            with gr.Row():
                send_btn = gr.Button("🚀 Send", variant="primary", scale=2)
                stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)
                clear_btn = gr.Button("🗑️ Clear Current Chat", scale=1)


    # --- Backend Logic Functions ---

    def update_system_prompt(template_key):
        """Updates the system prompt text box based on dropdown selection."""
        prompt_text = SYSTEM_PROMPTS.get(template_key, "")
        is_custom = template_key == "Custom..."
        return gr.Textbox(value=prompt_text, interactive=is_custom)


    def parse_and_format_response(full_response):
        """Formats <think> tags into a collapsible HTML section."""
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = think_pattern.search(full_response)
        if match:
            thinking_process = match.group(1).strip()
            final_answer = think_pattern.sub("", full_response).strip()
            formatted_html = (
                f"<details style='border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;'>"
                f"<summary style='cursor: pointer; font-weight: bold;'>🧠 View Model's Thought Process</summary>"
                f"<p style='margin-top: 10px;'>{thinking_process}</p></details>{final_answer}"
            )
            return formatted_html
        return full_response


    async def stream_response(history, *args):
        """Main function to stream responses from the LLM API."""
        if not history:
            return

        (model, base_url, api_key, system_prompt, temperature, top_p, max_tokens, presence_penalty) = args
        if not base_url or not api_key:
            raise gr.Error("API Base URL and API Key are required!")

        client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        # Reconstruct message history for the API call
        messages = [{"role": "system", "content": system_prompt}]
        for user_msg, assistant_msg in history[:-1]:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                # Remove the formatted HTML to get the raw text for the API
                raw_assistant_msg = re.sub(r"<details.*?</details>", "", assistant_msg, flags=re.DOTALL).strip()
                messages.append({"role": "assistant", "content": raw_assistant_msg})
        messages.append({"role": "user", "content": history[-1][0]})

        request_params = {
            "model": model, "messages": messages, "temperature": temperature,
            "top_p": top_p, "max_tokens": int(max_tokens),
            "presence_penalty": presence_penalty, "stream": True
        }

        full_response = ""
        try:
            stream = await client.chat.completions.create(**request_params)
            history[-1][1] = ""
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_response += token
                history[-1][1] = full_response
                yield history
        except Exception as e:
            history[-1][1] = f"API Request Error: {str(e)}"
            yield history
            return

        # Post-process the final response for special tags
        history[-1][1] = parse_and_format_response(full_response)
        yield history


    def add_message(history, message):
        """Adds the user's message to the chat history."""
        if not message:
            gr.Warning("Message cannot be empty!")
            return history, ""
        if history is None:
            history = []
        history.append([message, None])
        return history, ""


    def start_new_chat(current_history):
        """Saves the current chat and starts a new one."""
        if current_history:
            save_session_to_disk(current_history)
        updated_sessions = load_sessions_from_disk()
        return None, gr.Radio(choices=[data['title'] for data in updated_sessions.values()], value=None, interactive=True), ""


    def load_chat_history(session_title):
        """Loads a selected chat session from the global state."""
        if not session_title:
            return None
        for data in chat_sessions.values():
            if data['title'] == session_title:
                return data['history']
        return None


    def clear_current_chat():
        """Clears the chatbot and message box."""
        return None, ""


    # --- Event Listeners ---

    prompt_template_dd.change(
        fn=update_system_prompt,
        inputs=[prompt_template_dd],
        outputs=[system_prompt_box]
    )

    api_inputs = [
        model_dd, base_url_box, api_key_box, system_prompt_box,
        temperature_slider, top_p_slider, max_tokens_slider, presence_penalty_slider
    ]

    # Event chaining for sending a message
    send_event = msg_box.submit(
        add_message, [chatbot, msg_box], [chatbot, msg_box], queue=False
    ).then(
        stream_response, [chatbot] + api_inputs, chatbot
    )
    send_btn_event = send_btn.click(
        add_message, [chatbot, msg_box], [chatbot, msg_box], queue=False
    ).then(
        stream_response, [chatbot] + api_inputs, chatbot
    )

    # Event listeners for other buttons
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[send_event, send_btn_event])
    clear_btn.click(clear_current_chat, outputs=[chatbot, msg_box])
    new_chat_btn.click(start_new_chat, inputs=[chatbot], outputs=[chatbot, history_box, msg_box])
    history_box.change(load_chat_history, inputs=[history_box], outputs=[chatbot])


# --- Application Entry Point ---

if __name__ == "__main__":
    demo.queue().launch()