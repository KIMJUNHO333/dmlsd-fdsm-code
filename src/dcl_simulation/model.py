from dataclasses import dataclass
from typing import Protocol

from .audio import Audio, AudioSpec


@dataclass(frozen=True)
class ModelMeasurements:

    native_tokens: tuple[int, ...] | None = None
    native_text_token_ids: tuple[int, ...] | None = None
    sequence_length: int | None = None
    new_token_count: int | None = None

    def __post_init__(self):
        for value in (self.native_tokens, self.native_text_token_ids):
            if value is not None and (not isinstance(value, tuple)
                    or any(type(token) is not int or token < 0 for token in value)):
                raise ValueError("token IDs must be a tuple of nonnegative integers or None")
        for value in (self.sequence_length, self.new_token_count):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("token counts must be nonnegative integers or None")


@dataclass(frozen=True)
class ModelOutput:

    audio: Audio
    native_text: str | None = None
    generation_unit_id: str | None = None
    measurements: ModelMeasurements | None = None


class DuplexSession(Protocol):

    @property
    def audio_spec(self) -> AudioSpec:
        ...

    def process_tick(self, input_audio: Audio) -> ModelOutput:
        ...

    def close(self) -> None:
        ...


class DuplexModel(Protocol):

    def open_session(self) -> DuplexSession:
        ...
