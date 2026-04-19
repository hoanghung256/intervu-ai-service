from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_content(self, prompt: str, model: Optional[str] = None) -> Tuple[str, dict]:
        raise NotImplementedError
