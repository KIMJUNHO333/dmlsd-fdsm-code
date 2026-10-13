from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

from ...records import ComponentInfo
from ...services import UtteranceRequest, utterance_messages


@dataclass(frozen=True)
class LLMConfig:
    model: str
    endpoint: str = "http://127.0.0.1:8000/v1"
    revision: str | None = None
    decoding: dict[str, object] = field(default_factory=dict)
    request_timeout_s: float = 120
    api_key_env: str | None = None

    def __post_init__(self):
        ComponentInfo(self.model, self.revision, self.decoding)
        if not isinstance(self.endpoint, str):
            raise ValueError("endpoint must be an HTTP URL")
        url = urlsplit(self.endpoint)
        if url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment:
            raise ValueError("endpoint must be an HTTP URL without query or fragment")
        if url.username is not None or url.password is not None:
            raise ValueError("use api_key_env for authentication")
        if type(self.request_timeout_s) not in (int, float) or not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive and finite")
        if self.api_key_env is not None and (not isinstance(self.api_key_env, str) or not self.api_key_env.strip()):
            raise ValueError("api_key_env must be nonempty or None")
        reserved = {"model", "messages", "stream", "n"} & self.decoding.keys()
        if reserved:
            raise ValueError(f"reserved decoding fields: {sorted(reserved)}")
        object.__setattr__(self, "decoding", deepcopy(self.decoding))

    @classmethod
    def from_file(cls, path: str | Path) -> "LLMConfig":
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("LLM config must be a JSON object")
        return cls(**value)


class ChatCompletionLLM:
    def __init__(self, config: LLMConfig):
        if not isinstance(config, LLMConfig):
            raise TypeError("config must be LLMConfig")
        import httpx

        self._config = deepcopy(config)
        headers = {}
        if config.api_key_env is not None:
            key = os.environ.get(config.api_key_env)
            if not key:
                raise ValueError(f"missing API key environment variable: {config.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(base_url=config.endpoint.rstrip("/") + "/",
                                    timeout=config.request_timeout_s, headers=headers)

    @property
    def config(self) -> LLMConfig:
        return deepcopy(self._config)

    @property
    def provenance(self) -> ComponentInfo:
        return ComponentInfo(self._config.model, self._config.revision,
                             {"decoding": deepcopy(self._config.decoding)})

    def connect(self) -> None:
        response = self._client.get("models")
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("LLM model list must contain a data array")
        if not any(isinstance(row, dict) and row.get("id") == self._config.model for row in rows):
            raise ValueError(f"LLM server does not provide model: {self._config.model}")

    def generate(self, request: UtteranceRequest) -> str:
        response = self._client.post("chat/completions", json={
            "model": self._config.model,
            "messages": utterance_messages(request),
            **deepcopy(self._config.decoding),
            "stream": False,
            "n": 1,
        })
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("LLM response must contain one choice")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response has no utterance text")
        return content.strip()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ChatCompletionLLM":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
