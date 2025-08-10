import asyncio
from functools import partial
from typing import Callable
from openai import AsyncOpenAI


class LLMService:
    """Manages configuration and interaction with a Large Language Model."""

    def __init__(self, config: dict):
        self.config = config
        active_model_name = config['active_model']
        model_details = config['models'][active_model_name]

        api_key = config['api_keys'][f"{model_details['nickname'].lower()}_api_key"]
        endpoint = config['llm_endpoints'][f"{model_details['nickname'].lower()}_endpoint"]

        self.model_name = model_details['model_name']
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
        self.prompt_templates = config['prompts']
        self.semaphore = asyncio.Semaphore(10)  # Concurrency limit

    def get_prompt(self, step_name: str, **kwargs) -> str:
        """Formats a prompt template with the given keyword arguments."""
        if step_name not in self.prompt_templates:
            raise ValueError(f"Prompt template for '{step_name}' not found in config.yaml")

        return self.prompt_templates[step_name].format(**kwargs)

    async def call_llm(self, prompt: str) -> str:
        """
        Calls the configured LLM API with a given prompt, respecting concurrency limits.
        """
        async with self.semaphore:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    temperature=0.1,  # Low temperature for factual, reproducible tasks
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM API call failed. Error: {e}")
                return f"Error: LLM call failed. {e}"