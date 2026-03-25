from typing import Optional, List, Dict
from .history_repository import HistoryRepository

class MemoryHistoryRepository(HistoryRepository):
    def __init__(self) -> None:
        self.histories: Dict[str, List[Dict]] = {}
        self.last_cvs: Dict[str, Optional[Dict]] = {}
        self.last_cv_pdf_urls: Dict[str, Optional[str]] = {}
        self.transcripts: Dict[str, Dict] = {}

    async def get_history(self, user_id: str) -> List[Dict]:
        return self.histories.get(user_id, [])

    async def append_message(self, user_id: str, message: Dict, max_messages: Optional[int] = None) -> None:
        messages = self.histories.setdefault(user_id, [])
        messages.append(message)
        if max_messages and len(messages) > max_messages:
            self.histories[user_id] = messages[-max_messages:]

    async def clear_history(self, user_id: str) -> None:
        self.histories[user_id] = []

    async def get_last_cv(self, user_id: str) -> Optional[Dict]:
        return self.last_cvs.get(user_id)

    async def set_last_cv(self, user_id: str, cv_json: Optional[Dict]) -> None:
        self.last_cvs[user_id] = cv_json

    async def get_last_cv_pdf_url(self, user_id: str) -> Optional[str]:
        return self.last_cv_pdf_urls.get(user_id)

    async def set_last_cv_pdf_url(self, user_id: str, cv_pdf_url: Optional[str]) -> None:
        self.last_cv_pdf_urls[user_id] = cv_pdf_url

    async def get_transcript(self, user_id: str) -> Optional[Dict]:
        return self.transcripts.get(user_id)

    async def set_transcript(self, user_id: str, transcript_data: Dict) -> None:
        self.transcripts[user_id] = transcript_data
