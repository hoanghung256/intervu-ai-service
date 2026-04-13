from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_content(self, prompt: str, model: Optional[str] = None) -> str:
        raise NotImplementedError
