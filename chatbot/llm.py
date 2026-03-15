"""Ollama HTTP client for local LLM inference."""

import json
from typing import Generator

import requests

from chatbot.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
)


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    # -- Health checks --------------------------------------------------------

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def is_model_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            models = r.json().get("models", [])
            return any(
                m.get("name", "").startswith(self.model)
                for m in models
            )
        except (requests.ConnectionError, ValueError):
            return False

    def check_ready(self) -> str | None:
        """Return an error message if not ready, else None."""
        if not self.is_running():
            return (
                "Ollama is not running.\n"
                "  Start it with:  ollama serve\n"
                "  Or install:     brew install ollama"
            )
        if not self.is_model_available():
            return (
                f"Model '{self.model}' is not pulled.\n"
                f"  Pull it with:   ollama pull {self.model}"
            )
        return None

    # -- Chat -----------------------------------------------------------------

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        top_p: float = LLM_TOP_P,
    ) -> Generator[str, None, None]:
        """
        Send a chat completion request and yield tokens as they arrive.
        `messages` follows the OpenAI-style format:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            raise OllamaError(
                "Cannot reach Ollama. Is it running? (ollama serve)"
            )
        except requests.HTTPError as e:
            raise OllamaError(f"Ollama returned an error: {e}")

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

            if chunk.get("done", False):
                return

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Non-streaming convenience wrapper."""
        return "".join(self.chat_stream(messages, **kwargs))
