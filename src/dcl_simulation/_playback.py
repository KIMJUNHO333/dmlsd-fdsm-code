from dataclasses import dataclass
import math
import re

from .audio import AudioSpec
from .services import SynthesizedSpeech, WordTiming


@dataclass
class Utterance:
    id: int
    text: str
    goal: str
    chain: int
    attempt: int
    speech: SynthesizedSpeech
    start_s: float
    words: tuple[WordTiming, ...]
    cursor: int = 0
    end_s: float | None = None
    completion: str | None = None

    @property
    def natural_end_s(self) -> float:
        return self.start_s + self.speech.audio.size / self.speech.sample_rate

    @property
    def alignment(self) -> str:
        return "word_timestamps" if self.words else "word_ratio"

    def prefix_end(self, at_s: float) -> int:
        samples = min(self.cursor, max(0, round((at_s - self.start_s) * self.speech.sample_rate)))
        if samples >= self.speech.audio.size:
            return len(self.text)
        elapsed = samples / self.speech.sample_rate
        if self.words:
            return max((word.char_end for word in self.words if word.end_s <= elapsed + 1e-9), default=0)
        words = list(re.finditer(r"\S+", self.text))
        count = samples * len(words) // self.speech.audio.size
        return words[count - 1].end() if count else 0

    @property
    def delivered_text(self) -> str:
        return self.text[:self.prefix_end(self.end_s or self.natural_end_s)].strip()


def prepare_speech(text: str, speech: SynthesizedSpeech, sample_rate: int) -> tuple[WordTiming, ...]:
    if not isinstance(speech, SynthesizedSpeech):
        raise TypeError("TTS must return SynthesizedSpeech")
    if speech.sample_rate != sample_rate:
        raise ValueError("TTS must return the requested sample rate")
    size = getattr(speech.audio, "size", 0)
    if size <= 0:
        raise ValueError("TTS returned empty audio")
    AudioSpec(sample_rate, size, sample_rate, size).validate_input(speech.audio)
    duration = size / sample_rate
    previous_char = 0
    previous_time = 0.0
    for word in speech.words:
        try:
            valid = (isinstance(word, WordTiming)
                     and type(word.char_start) is int and type(word.char_end) is int
                     and 0 <= previous_char <= word.char_start < word.char_end <= len(text)
                     and math.isfinite(word.start_s) and math.isfinite(word.end_s)
                     and 0 <= previous_time <= word.start_s <= word.end_s <= duration + 1e-9)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            return ()
        previous_char, previous_time = word.char_end, word.end_s
    return speech.words
