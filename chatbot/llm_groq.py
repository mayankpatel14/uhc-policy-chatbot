"""Groq API client with streaming chat completions."""

from typing import Generator

from groq import Groq

from chatbot.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
)


class GroqError(Exception):
    pass


class GroqClient:
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        if not api_key:
            raise GroqError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys"
            )
        self._client = Groq(api_key=api_key)
        self.model = model

    def check_ready(self) -> str | None:
        """Return an error message if not ready, else None."""
        try:
            self._client.models.list()
            return None
        except Exception as e:
            return f"Groq API error: {e}"

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        top_p: float = LLM_TOP_P,
    ) -> Generator[str, None, None]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            error_msg = str(e).lower()
            if "rate_limit" in error_msg or "429" in error_msg:
                raise GroqError(
                    "Groq rate limit reached. Please wait a moment and try again."
                )
            raise GroqError(f"Groq API error: {e}")

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Non-streaming convenience wrapper."""
        return "".join(self.chat_stream(messages, **kwargs))
