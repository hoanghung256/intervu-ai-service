import os
import json
import re
import logging
from typing import Optional, List, Dict

from .base_provider import BaseLLMProvider
from .gemini_provider import GeminiProvider
from .huggingface_provider import HuggingFaceProvider
from .model_constants import GEMINI_DEFAULT_MODEL
from infrastructure.env_constants import ENV_LLM_PROVIDER

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    logging.warning("python-dotenv is not installed; skipping .env loading.")

class LLMProvider:
    def __init__(self, model_name: str = GEMINI_DEFAULT_MODEL, provider: Optional[BaseLLMProvider] = None):
        self.model_name = model_name
        self._provider = provider
        self._provider_name = ""
        self._config_error: Optional[str] = None

        if self._provider is not None:
            return

        provider_name = os.getenv(ENV_LLM_PROVIDER, "").strip().lower()
        self._provider_name = provider_name

        if not provider_name:
            self._config_error = f"LLM service is misconfigured ({ENV_LLM_PROVIDER} is required: gemini|huggingface)."
            logging.error(self._config_error)
            return

        if provider_name == "gemini":
            self._provider = GeminiProvider(model_name=model_name)
            return

        if provider_name == "huggingface":
            self._provider = HuggingFaceProvider()
            return

        self._config_error = f"LLM service is misconfigured ({ENV_LLM_PROVIDER} must be 'gemini' or 'huggingface')."
        logging.error(self._config_error)

    async def generate_content(self, prompt: str, model: Optional[str] = None) -> str:
        if self._config_error:
            return self._config_error

        if not self._provider:
            return "LLM service is currently unavailable (provider is not initialized)."

        return await self._provider.generate_content(prompt=prompt, model=model)

    async def chat_completion(self, messages: List[Dict], model: Optional[str] = None) -> Dict:
        prompt = self._messages_to_prompt(messages)
        content = await self.generate_content(prompt=prompt, model=model)
        return {
            "content": content,
            "role": "assistant"
        }

    @staticmethod
    def _messages_to_prompt(messages: List[Dict]) -> str:
        parts = []
        for message in messages or []:
            role = str(message.get("role", "user")).strip() or "user"
            raw_content = message.get("content", "")
            if isinstance(raw_content, (dict, list)):
                content = json.dumps(raw_content, ensure_ascii=False)
            else:
                content = str(raw_content)
            parts.append(f"{role}: {content}")
        return "\n".join(parts).strip()

    @staticmethod
    def clean_json_string(s: str) -> str:
        if not s:
            return ""

        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        s = s.strip()

        stack = []
        start_idx = None

        for idx, char in enumerate(s):
            if char in '{[':
                if not stack:
                    start_idx = idx
                stack.append(char)
            elif char in '}]':
                if stack:
                    top = stack[-1]
                    if (char == '}' and top == '{') or (char == ']' and top == '['):
                        stack.pop()
                    else:
                        continue

                    if not stack and start_idx is not None:
                        return s[start_idx:idx+1]

        # Fallback: try broad object slice if model returns extra text around JSON.
        first_obj = s.find("{")
        last_obj = s.rfind("}")
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            return s[first_obj:last_obj + 1]

        first_arr = s.find("[")
        last_arr = s.rfind("]")
        if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
            return s[first_arr:last_arr + 1]

        return s
