import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from infrastructure.env_constants import ENV_ELEVENLABS_API_KEY

ELEVENLABS_SPEECH_TO_TEXT_URL = "https://api.elevenlabs.io/v1/speech-to-text?enable_logging=true"
ELEVENLABS_DEFAULT_MODEL = "scribe_v2"


class ElevenLabsService:
    provider_name = "elevenlabs"

    def __init__(self):
        self.api_key = os.getenv(ENV_ELEVENLABS_API_KEY)
        # Keep `client` for compatibility with existing caller checks.
        self.client = self.api_key if self.api_key else None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def transcribe_file(self, audio_data: bytes) -> Optional[str]:
        if not self.is_configured:
            logging.error("ElevenLabs service is not configured")
            return None

        if not audio_data:
            logging.error("ElevenLabs transcription received empty audio data")
            return None

        try:
            headers = {"xi-api-key": self.api_key}
            files = {
                # The API accepts major audio/video formats; file name is only metadata.
                "file": ("audio_input.webm", audio_data, "application/octet-stream"),
            }
            data = {
                "model_id": ELEVENLABS_DEFAULT_MODEL,
                "diarize": "true",
                "timestamps_granularity": "word",
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
                response = await client.post(
                    ELEVENLABS_SPEECH_TO_TEXT_URL,
                    headers=headers,
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                payload = response.json()

            diarized_transcript = self._format_diarized_transcript(payload)
            if diarized_transcript:
                return diarized_transcript

            return payload.get("text")
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response else "unknown"
            logging.error(f"ElevenLabs transcription failed with status {status_code}: {e}")
            return None
        except Exception as e:
            logging.error(f"ElevenLabs transcription failed: {e}")
            return None

    def _format_diarized_transcript(self, payload: Dict[str, Any]) -> Optional[str]:
        words = self._extract_words(payload)
        if not words:
            return None

        transcript_parts: List[str] = []
        current_speaker: Optional[str] = None
        current_tokens: List[str] = []
        found_speaker = False

        for word in words:
            token_text = str(word.get("text", ""))
            if not token_text:
                continue

            speaker_id = word.get("speaker_id")
            speaker_label = self._speaker_label(speaker_id)
            if speaker_label:
                found_speaker = True

            if current_speaker and speaker_label and speaker_label != current_speaker:
                merged_text = "".join(current_tokens).strip()
                if merged_text:
                    transcript_parts.append(f"{current_speaker}: {merged_text}")
                current_tokens = []

            if speaker_label:
                current_speaker = speaker_label

            current_tokens.append(token_text)

        merged_text = "".join(current_tokens).strip()
        if merged_text and current_speaker:
            transcript_parts.append(f"{current_speaker}: {merged_text}")

        if found_speaker and transcript_parts:
            return "\n\n".join(transcript_parts)

        return None

    @staticmethod
    def _extract_words(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        words = payload.get("words")
        if isinstance(words, list):
            return [word for word in words if isinstance(word, dict)]

        transcripts = payload.get("transcripts")
        if not isinstance(transcripts, list):
            return []

        combined_words: List[Dict[str, Any]] = []
        for transcript in transcripts:
            if not isinstance(transcript, dict):
                continue
            channel_words = transcript.get("words")
            if not isinstance(channel_words, list):
                continue
            combined_words.extend(word for word in channel_words if isinstance(word, dict))

        combined_words.sort(key=lambda word: word.get("start", 0))
        return combined_words

    @staticmethod
    def _speaker_label(speaker_id: Any) -> Optional[str]:
        if speaker_id is None:
            return None

        speaker_text = str(speaker_id).strip()
        if not speaker_text:
            return None

        if speaker_text.startswith("speaker_"):
            return f"Speaker {speaker_text.split('speaker_', 1)[1]}"

        if speaker_text.isdigit():
            return f"Speaker {speaker_text}"

        return f"Speaker {speaker_text.replace('_', ' ')}"
