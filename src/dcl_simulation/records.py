from dataclasses import dataclass, field
import json
import math


@dataclass(frozen=True)
class SpeechSegment:
    id: int
    turn_id: int
    start_s: float
    end_s: float
    audio_sample_start: int
    audio_sample_end: int
    kind: str | None
    user_utterance_id: int | None
    observed_start_s: float
    observed_end_s: float | None
    truncated: bool


@dataclass(frozen=True)
class ResponseOpportunity:
    id: int
    user_utterance_id: int
    chain: int
    attempt: int
    start_s: float | None
    deadline_s: float | None
    outcome: str | None
    response_turn_id: int | None
    response_segment_id: int | None
    response_classification: str | None
    resolved_s: float | None
    observed_at_s: float | None


@dataclass(frozen=True)
class ComponentInfo:

    name: str
    revision: str | None = None
    settings: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("component name must be nonempty")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("component revision must be nonempty or None")
        _validate_settings(self.settings)


def _validate_settings(value):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("settings must be a JSON object with string keys")
    def check(item):
        if isinstance(item, dict):
            _validate_settings(item)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError("settings values must be JSON values")
    for item in value.values():
        check(item)
    json.dumps(value, allow_nan=False)


@dataclass(frozen=True)
class EvaluationProvenance:
    model: ComponentInfo | None = None
    llm: ComponentInfo | None = None
    tts: ComponentInfo | None = None
    vad: ComponentInfo | None = None
    asr: ComponentInfo | None = None
    vad_timestamp_resolution_s: float | None = None

    def __post_init__(self):
        for info in (self.model, self.llm, self.tts, self.vad, self.asr):
            if info is not None and not isinstance(info, ComponentInfo):
                raise TypeError("provenance components must be ComponentInfo or None")
        resolution = self.vad_timestamp_resolution_s
        if resolution is not None and (type(resolution) not in (int, float)
                or not math.isfinite(resolution) or resolution <= 0):
            raise ValueError("VAD timestamp resolution must be positive and finite")
