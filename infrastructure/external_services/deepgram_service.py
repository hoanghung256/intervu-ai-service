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
            logging.error("Deepgram client not initialized")
            return None

        try:
            response = await asyncio.to_thread(
                self.client.listen.v1.media.transcribe_file,
                request=audio_data,
                smart_format=True,
                diarize=True,
                punctuate=True,
                paragraphs=True
            )
            
            # Extract paragraphs which include speaker information
            paragraphs_data = response.results.channels[0].alternatives[0].paragraphs
            if paragraphs_data:
                transcript_parts = []
                for paragraph in paragraphs_data.paragraphs:
                    speaker = paragraph.speaker
                    # Join sentences in the paragraph
                    text = " ".join([s.text for s in paragraph.sentences])
                    transcript_parts.append(f"Speaker {speaker}: {text}")
                
                return "\n\n".join(transcript_parts)
            
            return response.results.channels[0].alternatives[0].transcript
        except Exception as e:
            logging.error(f"Deepgram transcription failed: {e}")
            return None
