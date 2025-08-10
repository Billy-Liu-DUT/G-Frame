import asyncio
import json
import yaml
from http.server import BaseHTTPRequestHandler, HTTPServer

from nano_graphrag import GraphRAG
from llm_services import get_llm_functions, get_embedding_function, single_llm_send


# This script runs a long-running server to handle RAG queries.

class RAGQueryHandler:
    """Handles the logic for processing RAG queries."""

    def __init__(self, config: dict):
        self.config = config
        rag_config = config.get('rag_config', {})
        data_config = config.get('data_paths', {})

        # Initialize LLM and embedding functions
        llm_func, self.single_llm_func = get_llm_functions(config)
        embedding_func = get_embedding_function(config)

        # Initialize GraphRAG. It will load the existing graph from the working_dir.
        print("Initializing GraphRAG for querying...")
        self.graph_rag = GraphRAG(
            working_dir=data_config.get('working_dir', './graph_data'),
            best_model_func=llm_func,
            embedding_func=embedding_func,
            always_create_working_dir=False,  # Important: Do not overwrite existing graph
        )
        print("GraphRAG query engine is ready.")

    def process_query(self, request_data: dict) -> dict:
        """Processes an incoming query and returns an OpenAI-compatible response."""
        try:
            user_input = request_data["messages"][-1]["content"]
            temperature = request_data.get("temperature", self.config.get('rag_config', {}).get('temperature', 0.0))

            # 1. Translate user input to English for the RAG system
            prompts_config = self.config.get('prompts', {})
            english_input = self.single_llm_func(
                prompt=user_input,
                system_prompt=prompts_config.get('translate_to_english', "Translate to English.")
            )

            # 2. Perform the GraphRAG query
            print(f"Performing RAG query with input: '{english_input}'")
            rag_output = asyncio.run(self.graph_rag.query(english_input))
            print(f"RAG response received: '{rag_output[:100]}...'")

            # 3. Translate the result back to Chinese for the user
            final_output = self.single_llm_func(
                prompt=rag_output,
                system_prompt=prompts_config.get('translate_to_chinese', "Translate to Chinese.")
            )

            # 4. Format the response as an OpenAI-compatible chat completion
            response = {
                "id": "rag-12345",
                "object": "chat.completion",
                "created": 1,
                "model": "GraphRAG",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_output},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            return response

        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "id": "error-12345",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": f"An internal error occurred: {e}"}}]
            }


def create_handler(query_handler: RAGQueryHandler):
    """Factory to create the HTTP request handler class with context."""

    class OpenAIProxyHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.query_handler = query_handler
            super().__init__(*args, **kwargs)

        def do_POST(self):
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data)

                response_data = self.query_handler.process_query(request_data)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.end_headers()
                error_response = {"error": {"message": f"Internal server error: {e}"}}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))

    return OpenAIProxyHandler


def run_server(config: dict):
    """Starts the HTTP server."""
    server_config = config.get('server', {})
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 8004)

    query_handler = RAGQueryHandler(config)
    handler_class = create_handler(query_handler)

    httpd = HTTPServer((host, port), handler_class)
    print(f"GraphRAG backend server starting on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    print("Loading configuration from config.yaml...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_server(config)