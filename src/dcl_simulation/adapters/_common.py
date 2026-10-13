from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from string import Template

import numpy as np


class ConfigFile:
    path_fields = ("model_path",)

    @classmethod
    def from_file(cls, path):
        source = Path(path).expanduser().resolve()
        values = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(values, base=source.parent)

    @classmethod
    def from_dict(cls, values, *, base=None):
        if not isinstance(values, dict):
            raise ValueError("config must be a JSON object")
        values = dict(values)
        values.pop("adapter", None)
        for name in cls.path_fields:
            if values.get(name) is not None:
                values[name] = str(model_path(values[name], base=base))
        return cls(**values)


def model_path(value, *, base=None):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("model_path must be nonempty")
    variables = dict(os.environ)
    variables.setdefault("MODEL_ROOT", str(Path.home() / "model"))
    path = Path(Template(str(value)).substitute(variables)).expanduser()
    return path if path.is_absolute() or base is None else base / path


def audio_array(audio):
    if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1:
        raise ValueError("audio must be a mono float32 NumPy array")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("audio must be nonempty and finite")
    return np.ascontiguousarray(audio)


def sample_rate(value):
    if type(value) is not int or value <= 0:
        raise ValueError("sample rate must be a positive integer")
    return value


def resample(audio, source_rate, target_rate):
    audio = audio_array(audio)
    sample_rate(source_rate)
    sample_rate(target_rate)
    if source_rate == target_rate:
        return audio.copy()
    from scipy.signal import resample_poly
    factor = math.gcd(source_rate, target_rate)
    return np.ascontiguousarray(resample_poly(audio, target_rate // factor, source_rate // factor), dtype=np.float32)


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def versions(*packages):
    return {name: version(name) for name in packages}
