"""Minimal Google Gemini client (REST) for MosquiTech Intelligence.

* The API key is read ONLY from the environment (GEMINI_API_KEY), optionally loaded from a local .env file
  that is git-ignored. It is never written to code, logs, error messages or the UI.
* No key / network failure / quota error -> a clear exception. The caller shows an error; nothing is ever faked.
* Cost controls: low max output tokens, low temperature, no retries, short timeout.
"""
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
    """No API key configured (the ML dashboard keeps working without it)."""


class LLMError(Exception):
    """The API call failed. The message is safe to show to users (contains no secrets)."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (never overrides variables that are already set)."""
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
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 30,
                 max_output_tokens: int = 1200, temperature: float = 0.2, post=None):
        self.api_key = (api_key if api_key is not None else os.environ.get(ENV_KEY, "")).strip()
        self.model = (model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL).strip()
        self.timeout, self.max_output_tokens, self.temperature = timeout, max_output_tokens, temperature
        self._post = post or requests.post          # injectable so tests never hit the network

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
        """messages: [{"role": "user"|"model", "text": str}, ...]  (last one must be the user's question)."""
        if not self.available:
            raise LLMUnavailable(f"No API key found. Set {ENV_KEY} (see .env.example) to enable MosquiTech Intelligence.")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": m["role"], "parts": [{"text": m["text"]}]} for m in messages],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens},
        }
        url = f"{BASE_URL}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        try:
            r = self._post(url, headers=headers, json=body, timeout=self.timeout)
        except requests.Timeout as exc:
            raise LLMError("The AI service timed out. Please try again.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach the AI service (network problem): {type(exc).__name__}.") from exc

        if r.status_code != 200:
            hints = {400: "the request was rejected (check GEMINI_MODEL)", 401: "the API key was rejected", 403: "the API key was rejected or lacks access",
                     404: f"model '{self.model}' was not found (check GEMINI_MODEL)", 429: "the quota or rate limit was reached; try again later"}
            why = hints.get(r.status_code, "the AI service returned an error" if r.status_code >= 500 else "the request failed")
            raise LLMError(f"AI service error (HTTP {r.status_code}): {why}.")
        try:
            data = r.json()
        except ValueError as exc:
            raise LLMError("The AI service returned an unreadable response.") from exc

        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise LLMError(f"The AI service declined this request ({block}). Try rephrasing the question.")
        cands = data.get("candidates") or []
        parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise LLMError("The AI service returned an empty answer. Please try again.")
        usage = data.get("usageMetadata") or {}
        return LLMResponse(text=self._safe(text), model=self.model, prompt_tokens=usage.get("promptTokenCount"),
                           output_tokens=usage.get("candidatesTokenCount"),
                           truncated=(cands[0].get("finishReason") == "MAX_TOKENS"))
