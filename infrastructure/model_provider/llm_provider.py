import json
import re
import logging
from typing import Optional, List, Dict, Tuple

from .base_provider import BaseLLMProvider
from .gemini_provider import GeminiProvider
from .huggingface_provider import HuggingFaceProvider
from .model_constants import (
    GEMINI_DEFAULT_MODEL,
    PROVIDER_GEMINI,
    PROVIDER_HUGGINGFACE,
    resolve_provider_name,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    logging.warning("python-dotenv is not installed; skipping .env loading.")


class LLMProvider:
    def __init__(self, model_name: str = GEMINI_DEFAULT_MODEL, provider: Optional[BaseLLMProvider] = None):
        self.model_name = model_name
        self._override: Optional[BaseLLMProvider] = provider
        self._gemini: Optional[GeminiProvider] = None
        self._hf: Optional[HuggingFaceProvider] = None

    def _get_provider(self, model: str) -> BaseLLMProvider:
        if self._override is not None:
            return self._override

        provider_name = resolve_provider_name(model)

        if provider_name == PROVIDER_GEMINI:
            if self._gemini is None:
                self._gemini = GeminiProvider(model_name=model)
            return self._gemini

        if provider_name == PROVIDER_HUGGINGFACE:
            if self._hf is None:
                self._hf = HuggingFaceProvider(model_name=model)
            return self._hf

        raise ValueError(f"Cannot resolve LLM provider for model '{model}'.")

    async def generate_content(self, prompt: str, model: Optional[str] = None) -> Tuple[str, dict]:
        _zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        effective_model = model or self.model_name
        try:
            provider = self._get_provider(effective_model)
        except ValueError as e:
            logging.error(str(e))
            return f"LLM service is misconfigured ({e}).", _zero

        return await provider.generate_content(prompt=prompt, model=effective_model)

    async def chat_completion(self, messages: List[Dict], model: Optional[str] = None) -> Dict:
        prompt = self._messages_to_prompt(messages)
        content, usage = await self.generate_content(prompt=prompt, model=model)
        return {
            "content": content,
            "role": "assistant",
            "usage": usage,
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
