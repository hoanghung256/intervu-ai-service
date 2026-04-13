import logging
import os
from typing import Optional

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

    async def generate_content(self, prompt: str, model: Optional[str] = None) -> str:
        if not self._client:
            return "Gemini service is currently unavailable (API key missing or invalid)."

        target_model = model or self.model_name
        try:
            response = await self._client.models.generate_content(
                model=target_model,
                contents=[prompt],
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}")
            return "Gemini service is currently unavailable (request failed)."
