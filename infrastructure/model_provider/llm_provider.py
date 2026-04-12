import os
import json
import re
import logging
from typing import Optional, List, Dict
from google import genai
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMProvider:
    def __init__(self, model_name: str = "gemma-3-27b-it"):
        self.model_name = model_name
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=gemini_api_key).aio
            except Exception as e:
                logging.error(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        else:
            logging.warning("GEMINI_API_KEY not found. Gemini services will be unavailable.")
            self.gemini_client = None

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                self.hg_client = AsyncOpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                    timeout=90.0,
                )
            except Exception as e:
                logging.error(f"Failed to initialize HuggingFace client: {e}")
                self.hg_client = None
        else:
            logging.warning("HF_TOKEN not found. HuggingFace services will be unavailable.")
            self.hg_client = None

    async def generate_content(self, prompt: str, model: Optional[str] = None) -> str:
        # Fallback to HuggingFace if Gemini is not available or if explicitly requested
        if self.hg_client:
            model = model or "meta-llama/Llama-3.1-8B-Instruct:novita"
            try:
                response = await self.hg_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"HuggingFace generation failed: {e}")

        if not self.gemini_client:
            return "LLM service is currently unavailable (HF_TOKEN and GEMINI_API_KEY missing or invalid)."
        
        model = model or self.model_name
        try:
            response = await self.gemini_client.models.generate_content(
                model=model,
                contents=[prompt]
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}")
            return "LLM service is currently unavailable (both HuggingFace and Gemini requests failed)."

    async def chat_completion(self, messages: List[Dict], model: str = "meta-llama/Llama-3.1-8B-Instruct:novita") -> Dict:
        if not self.hg_client:
            return {
                "content": "HuggingFace service is currently unavailable (API key missing or invalid).",
                "role": "assistant"
            }

        response = await self.hg_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return {
            "content": response.choices[0].message.content,
            "role": response.choices[0].message.role
        }

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
