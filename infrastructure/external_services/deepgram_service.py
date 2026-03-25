import os
import asyncio
import logging
from typing import Optional
from deepgram import DeepgramClient

class DeepgramService:
    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        try:
            self.client = DeepgramClient(api_key=self.api_key) if self.api_key else None
        except Exception as e:
            logging.error(f"Deepgram client initialization failed: {e}")
            self.client = None

    async def transcribe_file(self, audio_data: bytes) -> Optional[str]:
        if not self.client:
            return None

        response = await asyncio.to_thread(
            self.client.listen.v1.media.transcribe_file,
            request=audio_data,
            model="nova-3",
            smart_format=True,
        )
        return response.results.channels[0].alternatives[0].transcript
