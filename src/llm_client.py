
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import ROOT

ENV_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class LLMUnavailable(Exception):
    pass


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False


def load_env(path: Path | None = None) -> None:
    path = path or (ROOT / ".env")

    if not path.exists():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'")
            )


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 30,
        max_output_tokens: int = 1200,
        temperature: float = 0.2,
        post=None,
    ):
        self.api_key = (
            api_key if api_key is not None
            else os.environ.get(ENV_KEY, "")
        ).strip()

        self.model = (
            model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        ).strip()

        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._post = post or requests.post

    @classmethod
    def from_env(cls, **kw) -> "GeminiClient":
        load_env()
        return cls(**kw)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _safe(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text

    def generate(self, system: str, messages: list[dict]) -> LLMResponse:
        if not self.available:
            raise LLMUnavailable(
                f"No API key found. Set {ENV_KEY} "
                "(see .env.example) to enable MosquiTech Intelligence."
            )

        body = {
            "systemInstruction": {
                "parts": [{"text": system}]
            },
            "contents": [
                {
                    "role": message["role"],
                    "parts": [{"text": message["text"]}]
                }
                for message in messages
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        url = f"{BASE_URL}/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            response = self._post(
                url,
                headers=headers,
                json=body,
                timeout=self.timeout,
            )

        except requests.Timeout as exc:
            raise LLMError(
                "The AI service timed out. Please try again."
            ) from exc

        except requests.RequestException as exc:
            raise LLMError(
                f"Could not reach the AI service "
                f"(network problem): {type(exc).__name__}."
            ) from exc

        if response.status_code != 200:
            try:
                error_data = response.json().get("error", {})
                message = error_data.get(
                    "message",
                    "No detailed error message provided."
                )
                status = error_data.get("status", "")

            except (ValueError, AttributeError):
                message = "Could not read the API error response."
                status = ""

            detail = f"{status}: {message}" if status else message

            raise LLMError(
                self._safe(
                    f"Gemini API error "
                    f"(HTTP {response.status_code}): {detail}"
                )
            )

        try:
            data = response.json()

        except ValueError as exc:
            raise LLMError(
                "The AI service returned an unreadable response."
            ) from exc

        block = (
            (data.get("promptFeedback") or {}).get("blockReason")
        )

        if block:
            raise LLMError(
                f"The AI service declined this request ({block}). "
                "Try rephrasing the question."
            )

        candidates = data.get("candidates") or []

        parts = (
            (candidates[0].get("content") or {}).get("parts") or []
            if candidates else []
        )

        text = "".join(
            part.get("text", "") for part in parts
        ).strip()

        if not text:
            raise LLMError(
                "The AI service returned an empty answer. "
                "Please try again."
            )

        usage = data.get("usageMetadata") or {}

        return LLMResponse(
            text=self._safe(text),
            model=self.model,
            prompt_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            truncated=(
                candidates[0].get("finishReason") == "MAX_TOKENS"
            ),
        )
