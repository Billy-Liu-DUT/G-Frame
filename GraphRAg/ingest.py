import asyncio
import json
import yaml
from tqdm import tqdm
import traceback

from nano_graphrag import GraphRAG
from llm_services import get_llm_functions, get_embedding_function


# This script is a one-time process to ingest documents and build the knowledge graph.

async def ingest_documents(config: dict):
    """
    Initializes GraphRAG and ingests all source documents specified in the config.
    """
    print("Initializing GraphRAG for data ingestion...")

    rag_config = config.get('rag_config', {})
    data_config = config.get('data_paths', {})

    # Get LLM and embedding functions based on config
    llm_func, _ = get_llm_functions(config)
    embedding_func = get_embedding_function(config)

    # Initialize GraphRAG
    graph_rag = GraphRAG(
        working_dir=data_config.get('working_dir', './graph_data'),
        best_model_func=llm_func,
        embedding_func=embedding_func,
        always_create_working_dir=True,  # Create dir if it doesn't exist
        cheap_model_func=llm_func,  # Use the same model for simplicity
        best_model_max_async=rag_config.get('llm_concurrency', 20),
        cheap_model_max_async=rag_config.get('llm_concurrency', 20),
    )

    source_files = data_config.get('source_documents', [])
    if not source_files:
        print("No source documents found in config.yaml. Nothing to ingest.")
        return

    print(f"Found {len(source_files)} documents to ingest.")
    error_log = []

    for doc_path in tqdm(source_files, desc="Ingesting Documents"):
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                print(f"Warning: Document {doc_path} is empty. Skipping.")
                continue

            # Ingest the document content. The library will handle chunking.
            await graph_rag.insert(text)
            print(f"Successfully ingested {doc_path}")

        except Exception as e:
            error_message = f"Failed to ingest {doc_path}. Error: {e}, Traceback: {traceback.format_exc()}"
            print(error_message)
            error_log.append(error_message)

    if error_log:
        with open("ingestion_errors.log", "w", encoding="utf-8") as f:
            for error in error_log:
                f.write(error + "\n")
        print(f"Ingestion complete with {len(error_log)} errors. See ingestion_errors.log for details.")
    else:
        print("Ingestion complete successfully.")


if __name__ == "__main__":
    print("Loading configuration from config.yaml...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    asyncio.run(ingest_documents(config))