from dataclasses import dataclass
import math

import numpy as np

from ...records import ComponentInfo
from ...services import SynthesizedSpeech, WordTiming
from .._common import ConfigFile, audio_array, model_path, positive, resample, sample_rate as check_rate, versions


@dataclass(frozen=True)
class KokoroConfig(ConfigFile):
    model_path: str
    revision: str | None = None
    voice: str = "am_fenrir"
    speed: float = 1.0
    device: str = "cpu"

    def __post_init__(self):
        model_path(self.model_path)
        ComponentInfo("hexgrad/Kokoro-82M", self.revision)
        if not isinstance(self.voice, str) or len(self.voice) < 2 or self.voice[0] not in ("a", "b") or not self.voice.replace('_', '').isalnum():
            raise ValueError("voice must be an American or British English Kokoro voice")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be nonempty")
        positive(self.speed, "speed")


class KokoroTTS:
    def __init__(self, config: KokoroConfig):
        from kokoro import KModel, KPipeline
        root = model_path(config.model_path)
        self._voice = root / "voices" / f"{config.voice}.pt"
        for path in (root / "config.json", root / "kokoro-v1_0.pth", self._voice):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.config = config
        self._model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(root / "config.json"),
                             model=str(root / "kokoro-v1_0.pth")).to(config.device).eval()
        self._pipeline = KPipeline(lang_code=config.voice[0], repo_id="hexgrad/Kokoro-82M",
                                   model=self._model, device=config.device)

    @property
    def provenance(self):
        return ComponentInfo("hexgrad/Kokoro-82M", self.config.revision,
            {"voice": self.config.voice, "speed": self.config.speed, "device": self.config.device,
             "sample_rate": 24000, "packages": versions("kokoro", "torch", "scipy")})

    def synthesize(self, text: str, *, sample_rate: int) -> SynthesizedSpeech:
        import torch
        check_rate(sample_rate)
        if self._pipeline is None:
            raise RuntimeError("TTS is closed")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be nonempty")
        parts, tokens, count = [], [], 0
        with torch.inference_mode():
            for item in self._pipeline(text, voice=str(self._voice), speed=self.config.speed):
                if item.audio is None:
                    continue
                value = item.audio.detach().cpu().numpy() if hasattr(item.audio, "detach") else item.audio
                audio = audio_array(np.asarray(value, dtype=np.float32))
                tokens.extend((token, count / 24000) for token in (item.tokens or ()))
                parts.append(audio.copy())
                count += audio.size
        if not parts:
            raise ValueError("Kokoro returned no audio")
        audio = resample(np.concatenate(parts), 24000, sample_rate)
        words = _word_timings(text, tokens, audio.size / sample_rate)
        return SynthesizedSpeech(audio, sample_rate, words)

    def close(self):
        self._pipeline = None
        self._model = None


def _word_timings(text, tokens, duration):
    words, cursor, previous_end = [], 0, 0.0
    for token, offset in tokens:
        value = getattr(token, "text", None)
        if not isinstance(value, str) or not value.strip():
            continue
        start = text.find(value, cursor)
        if start < 0 or text[cursor:start].strip():
            return ()
        cursor = start + len(value)
        left, right = getattr(token, "start_ts", None), getattr(token, "end_ts", None)
        if left is None or right is None:
            if any(char.isalnum() for char in value):
                return ()
            continue
        try:
            left, right = float(left) + offset, float(right) + offset
        except (ValueError, TypeError):
            return ()
        if not (math.isfinite(left) and math.isfinite(right) and previous_end <= left <= right <= duration + 1e-9):
            return ()
        words.append(WordTiming(start, cursor, left, right))
        previous_end = right
    return tuple(words) if not text[cursor:].strip() else ()
