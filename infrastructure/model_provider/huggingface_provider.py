import logging
import os
from typing import Optional, Tuple

from .base_provider import BaseLLMProvider
from .model_constants import HUGGINGFACE_DEFAULT_MODEL
from infrastructure.env_constants import ENV_HF_TOKEN


class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, model_name: str = HUGGINGFACE_DEFAULT_MODEL, client=None):
        self.model_name = model_name
        self._client = client

        if self._client is not None:
            return

        hf_token = os.getenv(ENV_HF_TOKEN)
        if not hf_token:
            logging.warning(f"{ENV_HF_TOKEN} not found. HuggingFace services will be unavailable.")
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

    async def generate_content(self, prompt: str, model: Optional[str] = None) -> Tuple[str, dict]:
        _zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not self._client:
            return "HuggingFace service is currently unavailable (API key missing or invalid).", _zero

        target_model = model or self.model_name
        try:
            response = await self._client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            usage = _zero
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = response.usage.total_tokens or (prompt_tokens + completion_tokens)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            return content, usage
        except Exception as e:
            logging.error(f"HuggingFace generation failed: {e}")
            return "HuggingFace service is currently unavailable (request failed).", _zero
