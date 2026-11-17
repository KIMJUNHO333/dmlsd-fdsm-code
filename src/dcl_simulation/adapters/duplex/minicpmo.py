from dataclasses import dataclass
import hashlib
import math

import numpy as np
import soundfile as sf

from ...audio import AudioSpec
from ...model import ModelMeasurements, ModelOutput
from ...records import ComponentInfo
from .._common import ConfigFile, model_path, positive, resample, versions


def create_model(config, *, config_dir):
    return MiniCPMOModel(MiniCPMOConfig.from_dict(config, base=config_dir))


@dataclass(frozen=True)
class MiniCPMOConfig(ConfigFile):
    path_fields = ("model_path", "reference_audio")
    model_path: str
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    reference_audio: str | None = None
    prefix_system_prompt: str = "Have a natural spoken conversation with the user in English."
    max_new_speak_tokens_per_chunk: int = 20
    decode_mode: str = "sampling"
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.8
    listen_prob_scale: float = 1.0
    listen_top_k: int | None = None
    text_repetition_penalty: float = 1.05
    text_repetition_window_size: int = 512

    def __post_init__(self):
        model_path(self.model_path)
        if self.reference_audio is not None:
            model_path(self.reference_audio)
        ComponentInfo("openbmb/MiniCPM-o-4_5", self.revision)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be nonempty")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("invalid dtype")
        if not isinstance(self.prefix_system_prompt, str) or not self.prefix_system_prompt.strip():
            raise ValueError("prefix_system_prompt must be nonempty")
        if self.decode_mode not in ("sampling", "greedy"):
            raise ValueError("decode_mode must be sampling or greedy")
        for name in ("max_new_speak_tokens_per_chunk", "top_k", "text_repetition_window_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.listen_top_k is not None and (type(self.listen_top_k) is not int or self.listen_top_k <= 0):
            raise ValueError("listen_top_k must be a positive integer or None")
        for name in ("temperature", "text_repetition_penalty"):
            positive(getattr(self, name), name)
        if type(self.top_p) not in (int, float) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between zero and one")
        if type(self.listen_prob_scale) not in (int, float) or not math.isfinite(self.listen_prob_scale) or self.listen_prob_scale < 0:
            raise ValueError("listen_prob_scale must be nonnegative and finite")

    @property
    def generation(self):
        return {key: getattr(self, key) for key in (
            "max_new_speak_tokens_per_chunk", "decode_mode", "temperature", "top_k", "top_p",
            "listen_prob_scale", "listen_top_k", "text_repetition_penalty", "text_repetition_window_size",
        )}


class MiniCPMOModel:
    audio_spec = AudioSpec(16000, 16000, 24000, 24000)

    def __init__(self, config: MiniCPMOConfig):
        import torch
        from transformers import AutoModel

        root = model_path(config.model_path)
        self._reference = model_path(config.reference_audio) if config.reference_audio else root / "assets/HT_ref_audio.wav"
        for path in (root / "config.json", root / "modeling_minicpmo.py", self._reference):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not (root / "assets/token2wav").is_dir():
            raise FileNotFoundError(root / "assets/token2wav")
        audio, rate = sf.read(self._reference, dtype="float32", always_2d=True)
        self._reference_audio = resample(audio.mean(axis=1), rate, 16000)
        self._reference_hash = hashlib.sha256(self._reference.read_bytes()).hexdigest()
        self.config = config
        self._session = None
        self._closed = False
        self._model = None
        try:
            self._model = AutoModel.from_pretrained(str(root), trust_remote_code=True, local_files_only=True,
                torch_dtype=getattr(torch, config.dtype), attn_implementation="sdpa").eval().to(config.device)
            self._model.init_tts(model_dir=str(root / "assets/token2wav"))
        except BaseException:
            self.close()
            raise

    @property
    def provenance(self):
        return ComponentInfo("openbmb/MiniCPM-o-4_5", self.config.revision, {
            "device": self.config.device, "dtype": self.config.dtype, "attention": "sdpa",
            "prefix_system_prompt": self.config.prefix_system_prompt,
            "reference_audio_sha256": self._reference_hash, "generation": self.config.generation,
            "input_sample_rate": 16000, "output_sample_rate": 24000, "tick_s": 1.0,
            "packages": versions("transformers", "torch", "minicpmo-utils"),
        })

    def open_session(self):
        if self._closed:
            raise RuntimeError("MiniCPM-o model is closed")
        if self._session is not None:
            raise RuntimeError("close the active MiniCPM-o session before opening another")
        try:
            self._session = MiniCPMOSession(self)
        except BaseException:
            self.close()
            raise
        return self._session

    def close(self):
        self._closed = True
        try:
            if self._session is not None:
                self._session.close()
        finally:
            self._model = None


class MiniCPMOSession:
    def __init__(self, model):
        import torch
        self._owner = model
        self.audio_spec = model.audio_spec
        self._closed = False
        self._buffer = np.empty(0, dtype=np.float32)
        self._generation_index = 0
        with torch.inference_mode():
            self._duplex = model._model.as_duplex(device=model.config.device)
            self._duplex.prepare(prefix_system_prompt=model.config.prefix_system_prompt,
                ref_audio=model._reference_audio.copy(), prompt_wav_path=str(model._reference))

    def process_tick(self, input_audio):
        import torch
        if self._closed:
            raise RuntimeError("MiniCPM-o session is closed")
        self.audio_spec.validate_input(input_audio)
        with torch.inference_mode():
            prefill = self._duplex.streaming_prefill(audio_waveform=input_audio.copy(), max_slice_nums=1)
            if not isinstance(prefill, dict) or prefill.get("success") is not True:
                raise RuntimeError("MiniCPM-o audio prefill failed")
            result = self._duplex.streaming_generate(prompt_wav_path=str(self._owner._reference),
                                                     **self._owner.config.generation)
        if not isinstance(result, dict) or type(result.get("is_listen")) is not bool or type(result.get("end_of_turn")) is not bool:
            raise ValueError("invalid MiniCPM-o generation result")
        text = result.get("text")
        if not isinstance(text, str):
            raise ValueError("MiniCPM-o generation has no text field")
        waveform = result.get("audio_waveform")
        if waveform is not None:
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError("invalid MiniCPM-o waveform")
            self._buffer = np.concatenate((self._buffer, waveform))
        output = np.zeros(24000, dtype=np.float32)
        take = min(output.size, self._buffer.size)
        output[:take] = self._buffer[:take]
        self._buffer = self._buffer[take:].copy()
        unit = str(self._generation_index)
        self._generation_index += 1
        count = result.get("n_tokens")
        return ModelOutput(output, native_text="" if result["is_listen"] else text,
            generation_unit_id=unit, measurements=ModelMeasurements(new_token_count=count))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._owner._model.reset_session()
        finally:
            self._owner._session = None
            self._owner = self._duplex = None
            self._buffer = np.empty(0, dtype=np.float32)
