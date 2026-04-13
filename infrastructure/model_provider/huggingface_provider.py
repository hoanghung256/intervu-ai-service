import logging
import os
from typing import Optional

from .base_provider import BaseLLMProvider
from .model_constants import HUGGINGFACE_DEFAULT_MODEL


class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, model_name: str = HUGGINGFACE_DEFAULT_MODEL, client=None):
        self.model_name = model_name
        self._client = client

        if self._client is not None:
            return

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logging.warning("HF_TOKEN not found. HuggingFace services will be unavailable.")
            return

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
                timeout=90.0,
            )
        except Exception as e:
            logging.error(f"Failed to initialize HuggingFace client: {e}")
            self._client = None

    async def generate_content(self, prompt: str, model: Optional[str] = None) -> str:
        if not self._client:
            return "HuggingFace service is currently unavailable (API key missing or invalid)."

        target_model = model or self.model_name
        try:
            response = await self._client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"HuggingFace generation failed: {e}")
            return "HuggingFace service is currently unavailable (request failed)."
