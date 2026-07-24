import asyncio
import numpy as np
from openai import AsyncOpenAI, OpenAI
from nano_graphrag._utils import wrap_embedding_func_with_attrs


# This file centralizes LLM and embedding service interactions.

def get_llm_functions(config: dict):
    """Returns async and sync functions for LLM completions."""
    endpoints = config.get('endpoints', {})
    rag_config = config.get('rag_config', {})

    async_client = AsyncOpenAI(
        base_url=endpoints.get('llm_endpoint'),
        api_key="placeholder"
    )

    sync_client = OpenAI(
        base_url=endpoints.get('llm_endpoint'),
        api_key="placeholder"
    )

    async def llm_complete_async(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        completion = await async_client.chat.completions.create(
            model=endpoints.get('llm_model'),
            temperature=rag_config.get('temperature', 0.0),
            messages=messages,
        )
        return completion.choices[0].message.content

    def llm_complete_sync(prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = sync_client.chat.completions.create(
            model=endpoints.get('llm_model'),
            temperature=rag_config.get('temperature', 0.0),
            messages=messages,
            timeout=180
        )
        return completion.choices[0].message.content

    return llm_complete_async, llm_complete_sync


def single_llm_send(prompt: str, system_prompt: str = None) -> str:
    """A standalone sync function for simple, one-off LLM calls."""
    # This function is created here to avoid code duplication.
    # It will be assigned to the instance in the RAGQueryHandler.
    # This is a bit of a workaround for the fact that we can't easily get
    # the sync function from the async context of the server.
    # A cleaner solution would be a class that holds both clients.
    # For now, this is a pragmatic approach.
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _, sync_func = get_llm_functions(config)
    return sync_func(prompt, system_prompt)


def get_embedding_function(config: dict):
    """Returns a configured async embedding function."""
    endpoints = config.get('endpoints', {})

    async_client = AsyncOpenAI(
        base_url=endpoints.get('embedding_endpoint'),
        api_key="placeholder"
    )

    @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=1024)
    async def embed_texts_async(texts: list[str]) -> np.ndarray:
        response = await async_client.embeddings.create(
            model=endpoints.get('embedding_model'),
            input=texts,
            encoding_format="float"
        )
        arr = np.array([dp.embedding for dp in response.data])
        return arr

    return embed_texts_async