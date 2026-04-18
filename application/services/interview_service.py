import re
import os
from typing import Optional, List, Tuple
from infrastructure.model_provider.llm_provider import LLMProvider
from infrastructure.repository.history_repository import HistoryRepository
from infrastructure.env_constants import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    DEFAULT_MIN_ANSWER_SENTENCES,
    DEFAULT_MIN_ANSWER_WORDS,
    ENV_MAX_HISTORY_MESSAGES,
    ENV_MIN_ANSWER_SENTENCES,
    ENV_MIN_ANSWER_WORDS,
)

class InterviewService:
    def __init__(self, llm_provider: LLMProvider, history_repository: HistoryRepository):
        self.llm_provider = llm_provider
        self.history_repository = history_repository
        self.max_history_messages = int(os.getenv(ENV_MAX_HISTORY_MESSAGES, DEFAULT_MAX_HISTORY_MESSAGES))
        self.min_answer_sentences = int(os.getenv(ENV_MIN_ANSWER_SENTENCES, DEFAULT_MIN_ANSWER_SENTENCES))
        self.min_answer_words = int(os.getenv(ENV_MIN_ANSWER_WORDS, DEFAULT_MIN_ANSWER_WORDS))

    def validate_candidate_answer(self, candidate_answer: Optional[str], previous_question: Optional[str]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if candidate_answer is None:
            return False, ["missing answer"]

        answer = candidate_answer.strip()
        if not answer:
            reasons.append("empty or whitespace")
            return False, reasons

        if self._sentence_count(answer) < self.min_answer_sentences:
            reasons.append(f"fewer than {self.min_answer_sentences} sentences")
        if len(answer.split()) < self.min_answer_words:
            reasons.append(f"fewer than {self.min_answer_words} words")
        if self._is_generic_answer(answer):
            reasons.append("too generic")

        if previous_question:
            q_tokens = {w for w in re.findall(r"[a-zA-Z]{4,}", previous_question.lower())}
            a_tokens = {w for w in re.findall(r"[a-zA-Z]{4,}", answer.lower())}
            if q_tokens and len(q_tokens.intersection(a_tokens)) == 0 and len(answer.split()) < 25:
                reasons.append("does not clearly address the previous question")

        return len(reasons) == 0, reasons

    def extract_single_question(self, text: str) -> str:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return "Could you briefly introduce yourself and your relevant background?"
        if "?" in cleaned:
            return cleaned.split("?")[0].strip() + "?"
        return cleaned

    async def get_next_question(self, user_id: str, user_answer: Optional[str] = None) -> str:
        last_parsed_cv = await self.history_repository.get_last_cv(user_id)
        if last_parsed_cv is None:
            raise ValueError("No CV has been parsed yet for this id")

        history = await self.history_repository.get_history(user_id)
        previous_question = self._extract_last_assistant_question(history)
        is_first_turn = previous_question is None

        if user_answer is not None:
            await self.history_repository.append_message(
                user_id,
                {"role": "user", "content": user_answer},
                max_messages=self.max_history_messages,
            )

        if is_first_turn:
            llm_messages = [{
                "role": "user",
                "content": f"""
            You are an AI interviewer. Ask the candidate to briefly introduce themselves and background based on CV.
            CV: {last_parsed_cv}
            """,
            }]
        else:
            is_valid, reasons = self.validate_candidate_answer(user_answer, previous_question)
            if not is_valid:
                llm_messages = [{
                    "role": "user",
                    "content": f"""
                Invalid reason(s): {", ".join(reasons)}
                Previous question: {previous_question}
                Candidate answer: {user_answer}
                Stay on the SAME topic. Ask the same question again or a close paraphrase.
                """,
                }]
            else:
                updated_history = await self.history_repository.get_history(user_id)
                llm_messages = [{
                    "role": "user",
                    "content": f"""
                You are an AI interviewer conducting a structured interview based on the CV.
                Move to the next suitable question based on the transcript.
                CV: {last_parsed_cv}
                Transcript: {updated_history}
                """,
                }]

        _zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt_text = self.llm_provider._messages_to_prompt(llm_messages)
        content, usage = await self.llm_provider.generate_content(prompt_text)
        assistant_text = self.extract_single_question(content)
        usage = usage or _zero_usage

        await self.history_repository.append_message(
            user_id,
            {"role": "assistant", "content": assistant_text},
            max_messages=self.max_history_messages,
        )
        return assistant_text, usage

    def _extract_last_assistant_question(self, messages: List[dict]) -> Optional[str]:
        for item in reversed(messages):
            if item.get("role") == "assistant":
                content = (item.get("content") or "").strip()
                if content:
                    return content
        return None

    def _sentence_count(self, text: str) -> int:
        parts = re.split(r"[.!?]+", text.strip())
        return len([p for p in parts if p.strip()])

    def _is_generic_answer(self, text: str) -> bool:
        normalized = text.strip().lower()
        generic_phrases = {
            "yes", "no", "ok", "okay", "sure", "maybe", "idk", "i don't know",
            "not sure", "n/a", "na", "none", "same", "all good", "fine",
        }
        return normalized in generic_phrases
