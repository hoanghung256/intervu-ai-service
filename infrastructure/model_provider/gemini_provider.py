import logging
import os
from typing import Optional, Tuple

from .base_provider import BaseLLMProvider
from .model_constants import GEMINI_DEFAULT_MODEL
from infrastructure.env_constants import ENV_GEMINI_API_KEY


class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_name: str = GEMINI_DEFAULT_MODEL, client=None):
        self.model_name = model_name
        self._client = client

        if self._client is not None:
            return

        gemini_api_key = os.getenv(ENV_GEMINI_API_KEY)
        if not gemini_api_key:
            logging.warning(f"{ENV_GEMINI_API_KEY} not found. Gemini services will be unavailable.")
            return

        try:
            from google import genai

            self._client = genai.Client(api_key=gemini_api_key).aio
        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}")
            self._client = None

    async def generate_content(self, prompt: str, model: Optional[str] = None) -> Tuple[str, dict]:
        _zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not self._client:
            return "Gemini service is currently unavailable (API key missing or invalid).", _zero

        target_model = model or self.model_name
        try:
            response = await self._client.models.generate_content(
                model=target_model,
                contents=[prompt],
            )
            usage = _zero
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0
                total_tokens = response.usage_metadata.total_token_count or (prompt_tokens + completion_tokens)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            return response.text, usage
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}")
            return "Gemini service is currently unavailable (request failed).", _zero
