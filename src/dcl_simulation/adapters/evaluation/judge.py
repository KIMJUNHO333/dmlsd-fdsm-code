from dataclasses import dataclass
import math
import os
import re

from ...records import ComponentInfo
from .._common import ConfigFile, positive


@dataclass(frozen=True)
class GeminiConfig(ConfigFile):
    model: str = "gemini-3.8-flash"
    temperature: float = 0.0
    thinking_level: str = "low"
    max_output_tokens: int = 10000
    api_key_env: str = "GEMINI_API_KEY"
    request_timeout_s: float = 180

    def __post_init__(self):
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", self.model):
            raise ValueError("model must be a Gemini model ID")
        if type(self.temperature) not in (int, float) or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        if self.thinking_level not in ("minimal", "low", "medium", "high"):
            raise ValueError("invalid thinking_level")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if not isinstance(self.api_key_env, str) or not self.api_key_env.strip():
            raise ValueError("api_key_env must be nonempty")
        positive(self.request_timeout_s, "request_timeout_s")


class GeminiJudge:
    def __init__(self, config: GeminiConfig = GeminiConfig()):
        import httpx
        key = os.environ.get(config.api_key_env)
        if not key:
            raise ValueError(f"missing API key environment variable: {config.api_key_env}")
        self.config = config
        self._client = httpx.Client(base_url="https://generativelanguage.googleapis.com/v1beta/",
            headers={"x-goog-api-key": key}, timeout=config.request_timeout_s)

    @property
    def provenance(self):
        return ComponentInfo("gemini", self.config.model, {
            "temperature": self.config.temperature, "thinking_level": self.config.thinking_level,
            "max_output_tokens": self.config.max_output_tokens,
        })

    def complete(self, task):
        response = self._client.post(f"models/{self.config.model}:generateContent", json={
            "systemInstruction": {"parts": [{"text": task.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": task.user_prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "thinkingConfig": {"thinkingLevel": self.config.thinking_level.upper()},
                "responseMimeType": "application/json", "responseJsonSchema": task.response_schema,
            },
        })
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("Judge response must contain one candidate")
        candidate = candidates[0]
        if not isinstance(candidate, dict) or candidate.get("finishReason") != "STOP":
            raise ValueError("Judge response did not finish normally")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise ValueError("Judge response has no content parts")
        text = "".join(part["text"] for part in parts if isinstance(part, dict)
                       and not part.get("thought") and isinstance(part.get("text"), str))
        if not text.strip():
            raise ValueError("Judge response has no judgment text")
        return text

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
