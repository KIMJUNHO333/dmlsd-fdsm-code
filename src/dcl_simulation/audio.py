from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

Audio = NDArray[np.float32]


@dataclass(frozen=True)
class AudioSpec:

    input_sample_rate: int
    input_chunk_samples: int
    output_sample_rate: int
    output_chunk_samples: int

    def __post_init__(self) -> None:
        for name in (
            "input_sample_rate",
            "input_chunk_samples",
            "output_sample_rate",
            "output_chunk_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        output_duration = Fraction(self.output_chunk_samples, self.output_sample_rate)
        if self.tick_duration != output_duration:
            raise ValueError("input and output chunks must have equal duration")

    @property
    def tick_duration(self) -> Fraction:
        return Fraction(self.input_chunk_samples, self.input_sample_rate)

    def validate_input(self, audio: Audio) -> None:
        _validate_audio(audio, self.input_chunk_samples, "input")

    def validate_output(self, audio: Audio) -> None:
        _validate_audio(audio, self.output_chunk_samples, "output")


def _validate_audio(audio: Audio, samples: int, name: str) -> None:
    if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
        raise TypeError(f"{name} audio must be a float32 NumPy array")
    if audio.shape != (samples,):
        raise ValueError(f"{name} audio must have shape ({samples},), got {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{name} audio must contain only finite samples")
