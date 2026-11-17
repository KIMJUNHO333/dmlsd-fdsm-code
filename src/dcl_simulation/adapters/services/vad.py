from dataclasses import dataclass, asdict
import math

import numpy as np

from ...records import ComponentInfo
from ...services import SpeechEvent, SpeechSpan, SpeechUpdate
from .._common import ConfigFile, audio_array, resample, sample_rate as check_rate, versions


@dataclass(frozen=True)
class SileroConfig(ConfigFile):
    threshold: float = 0.5
    min_silence_ms: int = 300
    speech_pad_ms: int = 30

    def __post_init__(self):
        if type(self.threshold) not in (int, float) or not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")
        for value in (self.min_silence_ms, self.speech_pad_ms):
            if type(value) is not int or value < 0:
                raise ValueError("silence and padding must be nonnegative integer milliseconds")
        if self.speech_pad_ms > 32 + self.min_silence_ms:
            raise ValueError("speech padding exceeds the detection delay")


class SileroVAD:
    timestamp_resolution_s = 512 / 16000

    def __init__(self, config: SileroConfig = SileroConfig()):
        self.config = config

    @property
    def provenance(self):
        packages = versions("silero-vad", "torch", "scipy")
        return ComponentInfo("silero-vad", packages["silero-vad"],
            {**asdict(self.config), "sample_rate": 16000, "frame_samples": 512,
             "packages": packages})

    def open_session(self, *, sample_rate: int):
        return SileroSession(self.config, sample_rate)


class SileroSession:
    def __init__(self, config, sample_rate):
        import torch
        threads = torch.get_num_threads()
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
        finally:
            torch.set_num_threads(threads)
        self._model.reset_states()
        self.config = config
        self._sample_rate = check_rate(sample_rate)
        self._buffer = np.empty(0, dtype=np.float32)
        self._processed = 0
        self._received = 0
        self._speaking = False
        self._pending_end = None
        self._last_event = 0
        self._finalized = False

    def process(self, audio):
        import torch
        if self._model is None or self._finalized:
            raise RuntimeError("VAD session is closed or finalized")
        audio = audio_array(audio)
        if audio.size * 16000 % self._sample_rate:
            raise ValueError("each VAD input chunk must map to an integer number of 16 kHz samples")
        signal = resample(audio, self._sample_rate, 16000)
        self._received += signal.size
        self._buffer = np.concatenate((self._buffer, signal))
        events, activity = [], []
        offset = 0
        with torch.inference_mode():
            while offset + 512 <= self._buffer.size:
                frame = self._buffer[offset:offset + 512]
                probability = float(self._model(torch.from_numpy(frame), 16000).item())
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("invalid VAD probability")
                offset += 512
                self._processed += 512
                if probability >= self.config.threshold:
                    activity.append(SpeechSpan((self._processed - 512) / 16000, self._processed / 16000))
                    self._pending_end = None
                    if not self._speaking:
                        self._speaking = True
                        events.append(self._event("start", self._processed - 512 - self.config.speech_pad_ms * 16))
                elif probability < max(0, self.config.threshold - 0.15) and self._speaking:
                    if self._pending_end is None:
                        self._pending_end = self._processed
                    if self._processed - self._pending_end >= self.config.min_silence_ms * 16:
                        events.append(self._event("end", self._pending_end - 512 + self.config.speech_pad_ms * 16))
                        self._speaking = False
                        self._pending_end = None
        self._buffer = self._buffer[offset:].copy()
        return SpeechUpdate(tuple(events), tuple(activity))

    def _event(self, kind, sample):
        self._last_event = min(self._received, max(self._last_event, sample, 0))
        return SpeechEvent(kind, self._last_event / 16000)

    def finalize(self):
        if self._model is None or self._finalized:
            raise RuntimeError("VAD session is closed or finalized")
        self._finalized = True
        if self._speaking and self._pending_end is not None:
            event = self._event("end", self._pending_end - 512 + self.config.speech_pad_ms * 16)
            self._speaking = False
            return SpeechUpdate((event,))
        return SpeechUpdate()

    def close(self):
        self._model = None
        self._buffer = np.empty(0, dtype=np.float32)
