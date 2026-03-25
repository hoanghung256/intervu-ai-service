from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from domain.entities.interview import UserSession

class HistoryRepository(ABC):
    @abstractmethod
    async def get_history(self, user_id: str) -> List[Dict]:
        pass

    @abstractmethod
    async def append_message(self, user_id: str, message: Dict, max_messages: Optional[int] = None) -> None:
        pass

    @abstractmethod
    async def clear_history(self, user_id: str) -> None:
        pass

    @abstractmethod
    async def get_last_cv(self, user_id: str) -> Optional[Dict]:
        pass

    @abstractmethod
    async def set_last_cv(self, user_id: str, cv_json: Optional[Dict]) -> None:
        pass

    @abstractmethod
    async def get_last_cv_pdf_url(self, user_id: str) -> Optional[str]:
        pass

    @abstractmethod
    async def set_last_cv_pdf_url(self, user_id: str, cv_pdf_url: Optional[str]) -> None:
        pass

    @abstractmethod
    async def get_transcript(self, user_id: str) -> Optional[Dict]:
        pass

    @abstractmethod
    async def set_transcript(self, user_id: str, transcript_data: Dict) -> None:
        pass
